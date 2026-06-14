"""
记忆管理器 — 协调短期记忆、长期记忆（Chroma+Qwen3-8B）、用户画像（MySQL）

负责：
  - 检索：从三层记忆 + 文档知识库获取上下文
  - 更新：对话结束后更新所有记忆层
  - 压缩：短期记忆过长时压缩为摘要存入长期记忆
  - 文档：文档上传、切片、向量化一站式处理
  - 知识问答：基于文档的 RAG 问答
  - 用户画像：判断何时需要更新（永久存储于 MySQL）
"""
import os
import time
import json
from typing import Optional

from state import AgentState
from memory.long_term import LongTermMemory
from memory.user_profile import UserProfileDB
from memory.document_handler import DocumentProcessor
from memory.rerank import QwenEmbeddingClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from config import MEMORY_CONFIG, LLM_CONFIG, RETRIEVAL_CONFIG


class MemoryManager:
    """
    三层记忆 + 文档知识库 协调器

    ┌──────────────────────────────────────────────┐
    │  短期记忆 (State.short_term_context)          │
    │  - 当前会话上下文                             │
    ├──────────────────────────────────────────────┤
    │  长期记忆 (Chroma + Qwen3-8B Embedding)       │
    │  - 历史对话摘要 (语义检索)                    │
    │  - 知识片段                                   │
    ├──────────────────────────────────────────────┤
    │  文档知识库 (Chroma + 多路召回 + Reranker)    │
    │  - 用户上传文档 (PDF/Word/Markdown/...)       │
    │  - 智能切片 (RecursiveCharacterTextSplitter)  │
    │  - 多路召回 + RRF 融合 + Reranker 精排        │
    ├──────────────────────────────────────────────┤
    │  用户画像 (MySQL — 永久存储)                  │
    │  - 偏好设置（饮食/出行/语言）                 │
    │  - 行为习惯                                   │
    │  - 个人信息                                   │
    └──────────────────────────────────────────────┘
    """

    _instance: Optional["MemoryManager"] = None
    _initialized: bool = False

    @classmethod
    def get_instance(cls) -> "MemoryManager":
        """单例 — 避免每次创建 ChromaDB 客户端 + MySQL 连接池"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # 防止重复初始化（直接的 MemoryManager() 调用也安全）
        if MemoryManager._initialized:
            return
        MemoryManager._initialized = True
        self.long_term = LongTermMemory()
        self.user_profile_db = UserProfileDB()
        self.qwen_embedding = QwenEmbeddingClient()
        self.doc_processor = DocumentProcessor(
            embedding_fn=self.qwen_embedding.embed_single
        )
        self.short_term_max = MEMORY_CONFIG["short_term_max_tokens"]
        self.top_k = MEMORY_CONFIG["long_term_top_k"]
        self.update_threshold = MEMORY_CONFIG["user_profile_update_threshold"]

    # ==================== 检索 ====================

    def retrieve_context(self, state: AgentState) -> dict:
        """
        检索所有相关上下文（含文档知识库）

        Returns:
            {
                "short_term": {...},
                "long_term": [...],
                "user_profile": {...},
                "document_knowledge": [...],   # ← 新增：文档检索结果
                "combined_context": str,
            }
        """
        user_id = state.get("user_id", "default_user")
        user_input = state.get("user_input", "")

        short_term = state.get("short_term_context", {})

        # 长期记忆检索
        long_term_memories = self.long_term.search_memories(
            query=user_input,
            user_id=user_id,
            top_k=self.top_k,
        )

        # 用户画像（永久存储于 MySQL）
        user_profile = self.user_profile_db.get_profile(user_id)
        if user_profile:
            profile_text = self._format_profile(user_profile)
            short_term["user_profile_summary"] = profile_text

        # 文档知识库检索（带相关性门控：先做廉价 BM25 检查，不相关则跳过昂贵的多路召回）
        document_knowledge = []
        doc_count = self.long_term.get_document_count()
        if doc_count > 0:
            # 冷启动：ChromaDB 持久化了文档，但 BM25 在内存中，重启后丢失 → 重建
            if self.long_term._bm25 is None:
                self.long_term.rebuild_bm25_index()

            # 阶段1：廉价门控 — BM25 关键词快速判断 query 是否与文档相关
            bm25 = self.long_term._bm25
            if bm25 and bm25._bm25:
                quick_hits = bm25.search(user_input, top_k=3)
                top_bm25_score = quick_hits[0].get("bm25_score", 0) if quick_hits else 0
                # BM25 分数阈值：关键词完全匹配不上则跳过全流程
                if top_bm25_score > 0.5:
                    document_knowledge = self.long_term.search_documents(
                        query=user_input,
                        top_k=3,
                        use_multirecall=True,
                    )
                else:
                    print(f"  ├─ 文档门控：BM25 top_score={top_bm25_score:.2f} ≤ 0.5，跳过多路召回")
            else:
                # BM25 未安装（rank-bm25 缺失），直接走简单向量检索
                document_knowledge = self.long_term.search_documents(
                    query=user_input,
                    top_k=3,
                    use_multirecall=False,
                )

        # 拼合上下文
        combined = self._build_combined_context(
            user_input=user_input,
            short_term=short_term,
            long_term_memories=long_term_memories,
            user_profile=user_profile,
            document_knowledge=document_knowledge,
        )

        return {
            "short_term": short_term,
            "long_term": long_term_memories,
            "user_profile": user_profile,
            "document_knowledge": document_knowledge,
            "combined_context": combined,
        }

    # ==================== 更新 ====================

    def update_after_turn(self, state: AgentState) -> AgentState:
        """对话回合结束后更新所有记忆层"""
        user_id = state.get("user_id", "default_user")
        key_info = self._extract_key_info(state)

        ctx = state.get("short_term_context", {})
        ctx["last_topic"] = key_info.get("topic", "")
        ctx["last_tool_used"] = key_info.get("tool_used", "")
        ctx["turn_count"] = ctx.get("turn_count", 0) + 1
        ctx["recent_results"] = (ctx.get("recent_results", []) + [key_info])[-5:]
        state["short_term_context"] = ctx

        # 压缩短期→长期
        if ctx["turn_count"] >= MEMORY_CONFIG["summary_trigger_turns"]:
            self._compress_short_term(state)

        # 记录行为
        action_type = key_info.get("tool_used", "chat")
        self.user_profile_db.log_behavior(user_id, action_type, {
            "topic": key_info.get("topic", ""),
            "summary": key_info.get("summary", ""),
        })

        # 智能更新用户画像（永久存储）
        self._smart_update_profile(state, key_info)

        state["memory_needs_update"] = False
        return state

    # ==================== 文档管理 ====================

    def upload_document(self, file_path: str) -> dict:
        """
        上传并处理文档

        流程：解析 → 智能切片 → 向量化 → 存储 → 重建 BM25 索引

        Returns:
            {"success": True, "chunks": 15, "source": "report.pdf", "file_type": ".pdf"}
        """
        try:
            # 处理文档（解析 + 切片）
            chunks = self.doc_processor.process(file_path)

            # 存入向量库
            for chunk in chunks:
                self.long_term.add_document_chunk(
                    content=chunk["content"],
                    metadata=chunk["metadata"],
                    chunk_id=chunk["chunk_id"],
                )

            # 重建 BM25 索引
            self.long_term.rebuild_bm25_index()

            return {
                "success": True,
                "chunks": len(chunks),
                "source": chunks[0]["metadata"]["source"] if chunks else os.path.basename(file_path),
                "file_type": chunks[0]["metadata"]["file_type"] if chunks else "",
            }
        except Exception as e:
            return {
                "success": False,
                "chunks": 0,
                "error": str(e),
            }

    def search_documents(self, query: str, top_k: int = 5) -> list[dict]:
        """搜索文档知识库"""
        return self.long_term.search_documents(query, top_k, use_multirecall=True)

    def list_documents(self) -> list[dict]:
        """列出已上传文档"""
        return self.long_term.list_documents()

    def delete_document(self, source: str) -> dict:
        """删除文档"""
        count = self.long_term.delete_document(source)
        return {"success": count > 0, "deleted_chunks": count, "source": source}

    # ==================== 知识问答 ====================

    def knowledge_qa(self, query: str) -> dict:
        """
        基于文档知识库的问答

        流程：
        1. 多路召回检索相关文档切片
        2. 合并为上下文
        3. 返回上下文（由 LLM 生成最终答案）

        Returns:
            {"query": "...", "sources": [...], "context": "..."}
        """
        results = self.long_term.search_documents(query, top_k=5, use_multirecall=True)

        if not results:
            return {
                "query": query,
                "sources": [],
                "context": "",
                "answerable": False,
            }

        sources = []
        context_parts = []
        for r in results:
            meta = r.get("metadata", {})
            sources.append({
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", ""),
                "rerank_score": r.get("rerank_score", 0),
            })
            context_parts.append(r["content"])

        return {
            "query": query,
            "sources": sources,
            "context": "\n\n---\n\n".join(context_parts),
            "answerable": True,
        }

    # ==================== 压缩 ====================

    def _compress_short_term(self, state: AgentState):
        """
        LLM 驱动的短期记忆压缩 → 长期记忆

        不再简单截断拼接消息，而是用 LLM 提取关键事实、决策和偏好变化，
        生成高信息密度的结构化摘要。
        """
        user_id = state.get("user_id", "default_user")
        ctx = state.get("short_term_context", {})
        messages = state.get("messages", [])

        if len(messages) > 4:
            # 收集最近对话内容
            dialog_parts = []
            for msg in messages[-8:]:
                role = "用户" if isinstance(msg, HumanMessage) else "助手"
                content = getattr(msg, "content", str(msg))[:400]
                dialog_parts.append(f"{role}: {content}")
            dialog_text = "\n".join(dialog_parts)

            importance = 0.5
            for kw in ["偏好", "喜欢", "常用", "总是", "生日", "地址", "过敏"]:
                if kw in dialog_text:
                    importance = 0.8
                    break

            # LLM 生成结构化摘要
            summary = self._llm_summarize(dialog_text)

            self.long_term.add_memory(
                content=summary,
                metadata={
                    "user_id": user_id,
                    "topic": ctx.get("last_topic", ""),
                    "turn_count": ctx["turn_count"],
                    "importance": importance,
                },
                memory_type="conversation_summary",
            )

        ctx["turn_count"] = 0
        ctx["recent_results"] = ctx.get("recent_results", [])[-3:]
        state["short_term_context"] = ctx

    def _llm_summarize(self, dialog_text: str) -> str:
        """
        使用 LLM 将对话压缩为结构化摘要

        提取：关键事实、用户偏好变化、未完成任务、重要决策
        """
        try:
            llm = ChatOpenAI(
                model=LLM_CONFIG["model"],
                temperature=0.2,
                api_key=LLM_CONFIG["api_key"],
                base_url=LLM_CONFIG["base_url"],
                max_tokens=512,
            )
            response = llm.invoke([
                SystemMessage(content=(
                    "你是对话摘要专家。将对话压缩为结构化摘要，提取以下信息：\n"
                    "1. 关键事实与数据\n"
                    "2. 用户偏好变化\n"
                    "3. 未完成的任务\n"
                    "4. 重要决策\n"
                    "用 2-4 句话中文输出，不超过 200 字。"
                )),
                HumanMessage(content=dialog_text),
            ])
            return response.content.strip()
        except Exception:
            # LLM 不可用时降级为截断拼接
            return dialog_text[:500]

    # ==================== 智能画像更新 ====================

    def _smart_update_profile(self, state: AgentState, key_info: dict):
        """智能判断是否更新用户画像（永久存储于 MySQL）"""
        user_id = state.get("user_id", "default_user")
        topic = key_info.get("topic", "")

        # 饮食偏好
        food_kw = ["吃", "食物", "菜", "饭", "口味", "辣", "甜", "素食", "过敏"]
        if any(kw in topic for kw in food_kw):
            for k, v in key_info.get("extracted_prefs", {}).items():
                if self.user_profile_db.should_update_profile(user_id, "food", k, v):
                    self.user_profile_db.upsert_preference(user_id, "food", k, v, confidence=0.6 if "过敏" in topic else 0.4)

        # 出行偏好
        travel_kw = ["出行", "旅行", "交通", "航班", "火车", "酒店", "路线"]
        if any(kw in topic for kw in travel_kw):
            for k, v in key_info.get("extracted_prefs", {}).items():
                if self.user_profile_db.should_update_profile(user_id, "travel", k, v):
                    self.user_profile_db.upsert_preference(user_id, "travel", k, v, confidence=0.5)

    # ==================== 工具方法 ====================

    @staticmethod
    def _extract_key_info(state: AgentState) -> dict:
        plan = state.get("plan") or {}
        tool_results = state.get("tool_results", [])
        return {
            "topic": plan.get("task_summary", state.get("user_input", "")),
            "tool_used": tool_results[-1]["tool_name"] if tool_results else "chat",
            "summary": state.get("observation", ""),
            "extracted_prefs": {},
        }

    @staticmethod
    def _format_profile(profile: dict) -> str:
        if not profile:
            return "暂无用户画像"
        parts = []
        prefs = profile.get("preferences", {})
        for cat, items in prefs.items():
            cat_labels = {"food": "🍽️ 饮食", "travel": "🗺️ 出行", "language": "🔤 语言", "display": "🖥️ 显示"}
            label = cat_labels.get(cat, cat)
            item_strs = []
            for k, v in items.items():
                if isinstance(v, dict):
                    if v.get("confidence", 0) >= 0.6:
                        item_strs.append(f"{k}: {v['value']}")
                else:
                    item_strs.append(f"{k}: {v}")
            if item_strs:
                parts.append(f"  {label}: {', '.join(item_strs)}")
        if not parts:
            return "用户画像: 暂无明确偏好"
        return "用户画像（永久记忆）:\n" + "\n".join(parts)

    @staticmethod
    def _build_combined_context(
        user_input: str,
        short_term: dict,
        long_term_memories: list[dict],
        user_profile: Optional[dict],
        document_knowledge: list[dict] = None,
    ) -> str:
        parts = [f"【用户输入】{user_input}"]

        if user_profile:
            parts.append(f"【用户画像】\n{MemoryManager._format_profile(user_profile)}")

        if long_term_memories:
            mem_texts = []
            for mem in long_term_memories[:3]:
                content = mem.get("content", "")[:300]
                score = mem.get("weighted_score", 0)
                mem_texts.append(f"  [{score:.2f}] {content}")
            parts.append(f"【历史相关记忆】\n" + "\n".join(mem_texts))

        # 文档知识库（仅注入相关度超过阈值的文档）
        if document_knowledge:
            min_score = RETRIEVAL_CONFIG.get("rerank_min_score", 0.3)
            doc_texts = []
            for doc in document_knowledge[:3]:
                score = doc.get("rerank_score", doc.get("relevance", 0))
                if score < min_score:
                    continue  # 低相关度文档不注入上下文，避免误导 LLM
                content = doc.get("content", "")[:400]
                source = doc.get("metadata", {}).get("source", "未知文档")
                doc_texts.append(f"  📄 {source} [{score:.3f}]:\n  {content}")
            if doc_texts:
                parts.append(f"【文档知识库】\n" + "\n".join(doc_texts))

        recent = short_term.get("recent_results", [])
        if recent:
            recent_text = " | ".join(
                r.get("summary", r.get("topic", ""))[:200] for r in recent[-2:]
            )
            parts.append(f"【最近上下文】{recent_text}")

        return "\n\n".join(parts)
