"""
长期记忆 + 文档知识库 — Chroma 向量数据库 + Qwen3-8B Embedding

功能：
  - 对话摘要存储与语义检索
  - 用户文档上传、切片、向量化存储
  - 多路召回 + RRF 融合 + Reranker 精排
  - 知识问答

Embedding: Qwen3-8B (xop3qwen8bembedding) via 讯飞 MaaS
Reranker:  Qwen3-8B (xop3qwen8breranker) via 讯飞 MaaS
"""
import json
import hashlib
import time
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import CHROMA_CONFIG, QWEN_MAAS_CONFIG
from memory.rerank import BM25Retriever, RetrievalEngine


# ==================== Chroma Qwen Embedding 适配器 ====================

class QwenEmbeddingFunction:
    """
    Chroma 兼容的 Qwen3-8B Embedding Function

    绕过 ChromaDB 内置 OpenAIEmbeddingFunction（存在内部调用兼容问题），
    直接使用 openai.OpenAI 客户端调用 Qwen API，确保 add/query 路径一致。

    ChromaDB 1.5.x 协议要求：
      - name() 方法返回 embedding function 名称
      - __call__(self, input: list[str]) → list[list[float]] (批量)
      - embed_query(self, input: str) → list[float] (单条查询)
      - embed_documents(self, documents: list[str]) → list[list[float]] (批量文档)
      所有方法参数名必须是 'input'
    """

    def __init__(self):
        import openai as _openai
        self._client = _openai.OpenAI(
            api_key=QWEN_MAAS_CONFIG["api_key"],
            base_url=QWEN_MAAS_CONFIG["base_url"],
        )
        self._model = QWEN_MAAS_CONFIG["embedding_model"]
        self._dim = QWEN_MAAS_CONFIG["embedding_dim"]

    def name(self) -> str:
        return "qwen3-8b-embedding"

    def _call_api(self, input: list[str]) -> list[list[float]]:
        """底层 API 调用：list[str] → list[list[float]]"""
        if not input:
            return [[0.0] * self._dim for _ in input]
        try:
            resp = self._client.embeddings.create(model=self._model, input=input)
            # 按 index 排序保证顺序
            sorted_data = sorted(resp.data, key=lambda x: x.index)
            return [d.embedding for d in sorted_data]
        except Exception:
            # API 失败时让上层（memory node）处理降级
            raise

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Chroma 批量 embedding 接口"""
        return self._call_api(input)

    def embed_query(self, input) -> list[list[float]]:
        """
        ChromaDB query 路径专用

        注意：ChromaDB 1.5.x 实际传入的是 list[str] 并期望 2D 返回，
        与协议文档（单 str → 1D）不一致。这里按实际行为处理。
        """
        if isinstance(input, list):
            return self._call_api(input)          # list[str] → list[list[float]]
        else:
            return [self._call_api([input])[0]]   # str → 单条嵌入（包装为 2D）

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """ChromaDB add 路径专用：批量文档"""
        return self._call_api(documents)


# ==================== 长期记忆 + 文档存储 ====================

class LongTermMemory:
    """
    基于 Chroma + Qwen3-8B 的记忆与知识库

    三层 Collection:
    1. long_term_memory  — 对话摘要（长期记忆）
    2. knowledge_base     — 知识片段
    3. user_documents     — 用户上传文档

    检索策略（多路召回 + RRF + Reranker）：
    1. 原始查询向量召回
    2. 查询改写向量召回
    3. HyDE 向量召回
    4. BM25 关键词检索
    → RRF 融合 → Qwen3-8B Reranker 精排
    """

    def __init__(self, persist_dir: str = None):
        persist_dir = persist_dir or CHROMA_CONFIG["persist_directory"]

        # 使用 Qwen3-8B 作为 embedding 函数
        self.embedding_fn = QwenEmbeddingFunction()

        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # 创建 / 获取三个 collection（验证 embedding function 一致性）
        self.long_term_collection = self._get_or_validate_collection(
            name=CHROMA_CONFIG["collection_long_term"],
            metadata={"description": "长期对话记忆"},
        )

        self.knowledge_collection = self._get_or_validate_collection(
            name=CHROMA_CONFIG["collection_knowledge"],
            metadata={"description": "知识库"},
        )

        self.documents_collection = self._get_or_validate_collection(
            name=CHROMA_CONFIG["collection_documents"],
            metadata={"description": "用户上传文档"},
        )

        # BM25 索引（用于文档检索）
        self._bm25: Optional[BM25Retriever] = None
        self._bm25_docs: list[dict] = []

        # 检索引擎（延迟初始化）
        self._retrieval_engine: Optional[RetrievalEngine] = None

    def _get_or_validate_collection(self, name: str, metadata: dict = None):
        """
        获取或创建 collection，并验证 embedding function 一致性。

        如果 collection 已存在但 embedding function 不同，发出警告
        （旧数据需要 re-index 才能与新 embedding 兼容）。
        """
        try:
            existing = self.client.get_collection(name)
            existing_ef = getattr(existing, "embedding_function", None)
            if existing_ef is not None:
                # 安全获取名称：处理 name 可能是方法或属性的情况
                existing_ef_name = getattr(existing_ef, "name", None)
                if callable(existing_ef_name):
                    existing_ef_name = existing_ef_name()
                elif existing_ef_name is None:
                    existing_ef_name = type(existing_ef).__name__
                existing_ef_name = str(existing_ef_name)
            else:
                existing_ef_name = "none"

            current_ef_name = self.embedding_fn.name()
            if existing_ef_name != current_ef_name:
                print(
                    f"  ⚠️ Collection「{name}」的 embedding function 不匹配：\n"
                    f"     现有: {existing_ef_name}\n"
                    f"     当前: {current_ef_name}\n"
                    f"     如果更换了 embedding 模型，请删除 ./chroma_db 目录并重新上传文档。"
                )
            return existing
        except Exception:
            # Collection 不存在，创建
            return self.client.create_collection(
                name=name,
                embedding_function=self.embedding_fn,
                metadata=metadata,
            )

    # ==================== 对话记忆 CRUD ====================

    def add_memory(
        self,
        content: str,
        metadata: Optional[dict] = None,
        memory_type: str = "conversation_summary",
    ) -> str:
        """
        添加长期记忆

        Args:
            content: 记忆内容
            metadata: 元数据 {user_id, timestamp, topic, importance, ...}
            memory_type: conversation_summary | fact | preference

        Returns:
            memory_id
        """
        metadata = metadata or {}
        metadata.update({
            "memory_type": memory_type,
            "timestamp": metadata.get("timestamp", time.time()),
            "importance": metadata.get("importance", 0.5),
        })

        raw = f"{content}{json.dumps(metadata, sort_keys=True)}{time.time()}"
        memory_id = hashlib.md5(raw.encode()).hexdigest()[:16]

        collection = (
            self.long_term_collection
            if memory_type in ("conversation_summary", "fact")
            else self.knowledge_collection
        )

        collection.upsert(
            ids=[memory_id],
            documents=[content],
            metadatas=[{k: str(v) for k, v in metadata.items()}],
        )
        return memory_id

    def search_memories(
        self,
        query: str,
        user_id: Optional[str] = None,
        top_k: int = None,
        memory_type: Optional[str] = None,
    ) -> list[dict]:
        """
        语义检索长期记忆

        Args:
            query: 检索查询
            user_id: 按用户过滤
            top_k: 返回条数
            memory_type: 按类型过滤

        Returns:
            [{id, content, metadata, relevance, weighted_score}, ...]
        """
        top_k = top_k or 5

        where = {}
        if user_id:
            where["user_id"] = user_id
        if memory_type:
            where["memory_type"] = memory_type

        results = self.long_term_collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where if where else None,
            include=["documents", "metadatas", "distances"],
        )

        memories = []
        if results["ids"] and results["ids"][0]:
            for i, mem_id in enumerate(results["ids"][0]):
                memories.append({
                    "id": mem_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "relevance": 1.0 - results["distances"][0][i],
                })

        # 时间衰减加权
        current_time = time.time()
        for mem in memories:
            ts = float(mem["metadata"].get("timestamp", current_time))
            age_days = (current_time - ts) / 86400
            time_weight = max(0.3, 1.0 - age_days * 0.05)
            mem["weighted_score"] = mem["relevance"] * time_weight

        # 按时间衰减加权分数排序
        memories.sort(key=lambda m: m["weighted_score"], reverse=True)
        memories = memories[:top_k]

        return memories

    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        try:
            self.long_term_collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False

    # ==================== 用户文档存储 ====================

    def add_document_chunk(
        self,
        content: str,
        metadata: dict,
        chunk_id: str = None,
    ) -> str:
        """
        添加文档切片到向量库 + 追加到 BM25 语料

        注意：不在此处重建 BM25Okapi（避免 N 次重复构建）。
        调用方应在批量添加完成后调用 rebuild_bm25_index()。

        Args:
            content: 切片内容
            metadata: 元数据 {source, file_type, chunk_index, ...}
            chunk_id: 切片 ID（可选，自动生成）

        Returns:
            chunk_id
        """
        if not chunk_id:
            raw = f"{metadata.get('source', '')}_{metadata.get('chunk_index', 0)}_{content[:50]}"
            chunk_id = hashlib.md5(raw.encode()).hexdigest()[:16]

        meta = {k: str(v) for k, v in metadata.items()}
        meta["timestamp"] = str(time.time())

        self.documents_collection.upsert(
            ids=[chunk_id],
            documents=[content],
            metadatas=[meta],
        )

        # 追加到 BM25 语料（不重建 Okapi —— 由 rebuild_bm25_index 统一处理）
        self._bm25_docs.append({
            "content": content,
            "metadata": meta,
            "chunk_id": chunk_id,
        })

        return chunk_id

    def rebuild_bm25_index(self):
        """
        从内存语料 (_bm25_docs) 重建 BM25Okapi 对象 + 重置检索引擎。

        调用时机：文档上传完成 / 文档删除后。
        不重新从 ChromaDB 拉取 —— _bm25_docs 始终与 ChromaDB 保持同步。
        """
        self._bm25 = BM25Retriever()
        self._bm25.index(self._bm25_docs)
        # 关键：重置检索引擎，让它下次使用新的 BM25 引用
        self._retrieval_engine = None
        print(f"  📇 BM25 索引已重建：{len(self._bm25_docs)} 个文档切片 (检索引擎已重置)")

    def search_documents(
        self,
        query: str,
        top_k: int = 5,
        use_multirecall: bool = True,
    ) -> list[dict]:
        """
        文档知识检索（多路召回 + RRF + Reranker）

        Args:
            query: 查询文本
            top_k: 最终返回数量
            use_multirecall: 是否使用多路召回（否则仅向量检索）

        Returns:
            [{"content": "...", "metadata": {...}, "rerank_score": 0.95}, ...]
        """
        if use_multirecall:
            return self._multirecall_search(query, top_k)
        else:
            return self._simple_search(query, top_k)

    def _simple_search(self, query: str, top_k: int) -> list[dict]:
        """简单向量检索"""
        results = self.documents_collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                chunks.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "relevance": 1.0 - results["distances"][0][i],
                })
        return chunks

    def _multirecall_search(self, query: str, top_k: int) -> list[dict]:
        """
        多路召回 + RRF 融合 + Reranker 精排

        4 路召回：
        1. 原始查询向量召回
        2. 查询改写向量召回
        3. HyDE 向量召回
        4. BM25 关键词检索

        → RRF 融合 → Qwen3-8B Reranker 精排
        """
        # 惰性初始化 BM25
        if self._bm25 is None:
            self.rebuild_bm25_index()

        # 向量检索器
        def vector_searcher(q: str, k: int) -> list[dict]:
            return self._simple_search(q, k)

        # 创建/获取检索引擎
        if self._retrieval_engine is None:
            self._retrieval_engine = RetrievalEngine(
                vector_searcher=vector_searcher,
                bm25_retriever=self._bm25,
            )

        return self._retrieval_engine.retrieve(query, top_k=top_k)

    # ==================== 文档管理 ====================

    def list_documents(self) -> list[dict]:
        """列出所有已上传的文档（按 source 去重）"""
        try:
            results = self.documents_collection.get(include=["metadatas"])
            sources = {}
            if results["metadatas"]:
                for meta in results["metadatas"]:
                    source = meta.get("source", "unknown")
                    if source not in sources:
                        sources[source] = {
                            "source": source,
                            "file_type": meta.get("file_type", ""),
                            "chunk_count": 0,
                        }
                    sources[source]["chunk_count"] += 1
            return list(sources.values())
        except Exception:
            return []

    def delete_document(self, source: str) -> int:
        """
        删除指定文档的所有切片（ChromaDB + BM25 语料）

        Returns:
            删除的切片数量
        """
        try:
            # 查找该文档的所有切片 ID
            results = self.documents_collection.get(
                include=["metadatas"],
            )
            ids_to_delete = []
            if results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    meta = results["metadatas"][i] if results["metadatas"] else {}
                    if meta.get("source") == source:
                        ids_to_delete.append(doc_id)

            if ids_to_delete:
                self.documents_collection.delete(ids=ids_to_delete)
                # 同步清理 BM25 语料
                self._bm25_docs = [
                    d for d in self._bm25_docs
                    if d.get("metadata", {}).get("source") != source
                ]
                self.rebuild_bm25_index()

            return len(ids_to_delete)
        except Exception:
            return 0

    def get_document_count(self) -> int:
        """获取文档切片总数"""
        return self.documents_collection.count()

    # ==================== 知识库 ====================

    def add_knowledge(self, content: str, metadata: Optional[dict] = None) -> str:
        """添加知识片段"""
        metadata = metadata or {}
        metadata["timestamp"] = time.time()
        raw = f"{content}{json.dumps(metadata)}{time.time()}"
        kid = hashlib.md5(raw.encode()).hexdigest()[:16]

        self.knowledge_collection.upsert(
            ids=[kid],
            documents=[content],
            metadatas=[{k: str(v) for k, v in metadata.items()}],
        )
        return kid

    def search_knowledge(self, query: str, top_k: int = 5) -> list[dict]:
        """搜索知识库"""
        results = self.knowledge_collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        knowledge = []
        if results["ids"] and results["ids"][0]:
            for i, kid in enumerate(results["ids"][0]):
                knowledge.append({
                    "id": kid,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "relevance": 1.0 - results["distances"][0][i],
                })
        return knowledge

    # ==================== 统计 ====================

    def get_stats(self) -> dict:
        """获取存储统计"""
        return {
            "long_term_count": self.long_term_collection.count(),
            "knowledge_count": self.knowledge_collection.count(),
            "document_chunks": self.documents_collection.count(),
            "embedding_model": QWEN_MAAS_CONFIG["embedding_model"],
            "reranker_model": QWEN_MAAS_CONFIG["reranker_model"],
        }
