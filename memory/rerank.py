"""
多路召回 + RRF 融合 + Reranker 精排

完整检索流程：
  ┌─────────────────────────────────────────────────────┐
  │                    User Query                        │
  └────────┬────────────┬───────────┬──────────┬────────┘
           ↓            ↓           ↓          ↓
    ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
    │ 原始召回  │ │ 改写召回  │ │ HyDE召回 │ │ BM25召回 │
    │ (vector) │ │ (vector) │ │ (vector) │ │(keyword) │
    └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘
         ↓             ↓           ↓            ↓
    ┌──────────────────────────────────────────────────┐
    │              RRF 融合 (Reciprocal Rank Fusion)     │
    │    score(d) = Σ 1/(k + rank_i(d))                 │
    └──────────────────────┬───────────────────────────┘
                           ↓
    ┌──────────────────────────────────────────────────┐
    │           Qwen3-8B Reranker 精排                   │
    │    query × documents → relevance scores            │
    └──────────────────────┬───────────────────────────┘
                           ↓
                   Final Top-K Results
"""
import time
import json
import math
import re
import hashlib
from typing import Optional, Callable
from collections import defaultdict

import httpx
import openai

from config import (
    QWEN_MAAS_CONFIG,
    RETRIEVAL_CONFIG,
    LLM_CONFIG,
)


# ==================== Qwen3-8B Embedding 客户端 ====================

class QwenEmbeddingClient:
    """
    Qwen3-8B Embedding API 客户端

    用于文档向量化和查询向量化
    """

    def __init__(self):
        self.client = openai.OpenAI(
            api_key=QWEN_MAAS_CONFIG["api_key"],
            base_url=QWEN_MAAS_CONFIG["base_url"],
        )
        self.model = QWEN_MAAS_CONFIG["embedding_model"]
        self.timeout = QWEN_MAAS_CONFIG["timeout"]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        批量文本向量化

        Args:
            texts: 文本列表

        Returns:
            embedding 向量列表
        """
        max_batch = QWEN_MAAS_CONFIG["max_batch_size"]
        all_embeddings = []

        for i in range(0, len(texts), max_batch):
            batch = texts[i:i + max_batch]
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                )
                # 按 index 排序确保顺序
                batch_embeddings = sorted(response.data, key=lambda x: x.index)
                all_embeddings.extend([e.embedding for e in batch_embeddings])
            except Exception as e:
                # 不再静默返回零向量（会污染向量空间）
                # 让上层决定如何处理：重试 / 降级跳过 / 报错
                raise RuntimeError(
                    f"Embedding API 请求失败 (batch {i // max_batch}): {e}"
                ) from e

        return all_embeddings

    def embed_single(self, text: str) -> list[float]:
        """单文本向量化"""
        result = self.embed([text])
        return result[0] if result else [0.0] * QWEN_MAAS_CONFIG["embedding_dim"]


# ==================== Qwen3-8B Reranker 客户端 ====================

class QwenRerankerClient:
    """
    Qwen3-8B Reranker API 客户端（讯飞 MaaS）

    用于对召回结果进行精排。

    API 格式（讯飞 MaaS Rerank HTTP 协议）：
      POST https://maas-api.cn-huabei-1.xf-yun.com/v2/rerank
      Authorization: Bearer {API_KEY}
      Body: {"model": "xop3qwen8breranker", "query": "...", "documents": [...]}
      Response: {"results": [{"index": 0, "relevance_score": 0.95}, ...]}

    注意：使用 httpx 直连而非 openai.OpenAI SDK，
    因为 /rerank 不是 OpenAI 兼容端点，OpenAI SDK 的请求/响应处理会引入兼容性问题。
    """

    def __init__(self):
        self.api_key = QWEN_MAAS_CONFIG["api_key"]
        self.base_url = QWEN_MAAS_CONFIG["base_url"]
        self.model = QWEN_MAAS_CONFIG["reranker_model"]
        self.timeout = QWEN_MAAS_CONFIG["timeout"]

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int = None,
    ) -> list[dict]:
        """
        对文档列表重排序

        Args:
            query: 查询文本
            documents: 待重排的文档内容列表
            top_k: 返回前 K 个

        Returns:
            [{"index": 原始索引, "content": "...", "score": 0.95}, ...]
        """
        if not documents:
            return []

        top_k = top_k or RETRIEVAL_CONFIG["rerank_top_k"]

        try:
            url = f"{self.base_url}/rerank"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            payload = {
                "model": self.model,
                "query": query,
                "documents": documents,
            }

            response = httpx.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            result = response.json()

            # 解析结果（讯飞 MaaS 格式）
            # results: [{"index": 0, "relevance_score": 0.95}, ...]
            # 注意：API 返回不保证按分数排序，需客户端自行排序
            ranked = []
            for item in result.get("results", []):
                idx = item.get("index", 0)
                ranked.append({
                    "index": idx,
                    "content": documents[idx] if idx < len(documents) else "",
                    "score": item.get("relevance_score", 0.0),
                })

            # 按分数降序排列
            ranked.sort(key=lambda x: x["score"], reverse=True)
            return ranked

        except httpx.HTTPStatusError as e:
            print(f"  ⚠️ Reranker HTTP 错误 {e.response.status_code}: {e.response.text[:200]}")
            return _rerank_fallback(documents, top_k)
        except httpx.TimeoutException:
            print(f"  ⚠️ Reranker 请求超时 (>{self.timeout}s)")
            return _rerank_fallback(documents, top_k)
        except Exception as e:
            print(f"  ⚠️ Reranker 请求出错: {e}")
            return _rerank_fallback(documents, top_k)


# ==================== BM25 关键词检索 ====================

class BM25Retriever:
    """
    BM25 关键词检索器

    采用 rank-bm25 库实现，对文档库做关键词级别的精确匹配。
    与向量检索互补：向量检索擅长语义匹配，BM25 擅长精确关键词匹配。
    """

    def __init__(self):
        self._corpus: list[str] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25 = None
        self._doc_map: list[dict] = []  # 文档元数据映射

    def index(self, documents: list[dict]):
        """
        构建 BM25 索引（全量）

        Args:
            documents: [{"content": "...", "metadata": {...}}, ...]
        """
        self._doc_map = list(documents)
        self._corpus = [doc.get("content", "") for doc in documents]
        self._tokenized_corpus = [self._tokenize(text) for text in self._corpus]
        self._rebuild_bm25()

    def add_document(self, doc: dict):
        """
        增量添加单个文档到 BM25 索引

        避免每次上传文档都 O(n) 全量重建。
        内部仍需重建 BM25Okapi（IDF 依赖全语料），但不需重新从 ChromaDB 拉取。

        Args:
            doc: {"content": "...", "metadata": {...}}
        """
        content = doc.get("content", "")
        self._doc_map.append(doc)
        self._corpus.append(content)
        self._tokenized_corpus.append(self._tokenize(content))
        self._rebuild_bm25()

    def _rebuild_bm25(self):
        """内部：重建 BM25Okapi 对象（IDF 计算依赖全语料统计）"""
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        except ImportError:
            print("  ⚠️ rank-bm25 未安装，BM25 检索不可用")
            self._bm25 = None

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        BM25 关键词检索

        Args:
            query: 查询文本
            top_k: 返回数量

        Returns:
            [{"content": "...", "metadata": {...}, "bm25_score": 0.95}, ...]
        """
        if not self._bm25 or not self._tokenized_corpus:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # 按分数排序
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in indexed_scores[:top_k]:
            if score > 0:
                results.append({
                    "content": self._corpus[idx],
                    "metadata": self._doc_map[idx].get("metadata", {}),
                    "bm25_score": float(score),
                    "source": "bm25",
                })

        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """
        中文友好的分词

        策略：
        - 中文：按字符 2-gram + 单字
        - 英文/数字：按空格分词 + 小写
        """
        tokens = []

        # 提取中文字符做 2-gram
        chinese_chars = re.findall(r'[一-鿿]', text)
        for i in range(len(chinese_chars)):
            tokens.append(chinese_chars[i])  # 单字
            if i < len(chinese_chars) - 1:
                tokens.append(chinese_chars[i] + chinese_chars[i + 1])  # 2-gram

        # 英文/数字分词
        english_words = re.findall(r'[a-zA-Z0-9]+', text)
        tokens.extend(w.lower() for w in english_words)

        return tokens


# ==================== 查询改写 & HyDE ====================

class QueryProcessor:
    """
    查询处理器

    功能：
    1. 查询改写 — LLM 改写用户查询为更精准的检索 query
    2. HyDE — 生成假设性文档，用假设文档的 embedding 去检索
    """

    REWRITE_PROMPT = """你是一个查询改写专家。将用户的问题改写为更好的检索查询。

规则：
1. 提取核心关键词，去除口语化表达
2. 如果是长问题，拆分为 2-3 个短查询（用换行分隔）
3. 保留专业术语和实体名称
4. 输出纯文本查询，不要加引号或编号

用户问题：{query}
改写查询："""

    HYDE_PROMPT = """你是一个知识助手。根据用户的问题，写一段可能包含答案的假设性文档段落。

规则：
1. 不要直接回答问题，而是写一段"看起来像答案"的文档段落
2. 使用专业、正式的语气，类似百科或技术文档
3. 长度控制在 100-200 字
4. 如果问题有多个方面，覆盖主要方面

用户问题：{query}
假设文档："""

    def __init__(self):
        self.llm = openai.OpenAI(
            api_key=LLM_CONFIG["api_key"],
            base_url=LLM_CONFIG["base_url"],
        )
        # 查询改写和 HyDE 是检索辅助步骤，不需要深度推理
        # 使用 flash 模型节省成本和延迟
        self.model = LLM_CONFIG["model"]

    def rewrite_query(self, query: str) -> list[str]:
        """
        查询改写：生成 1-3 个改写后的查询

        Returns:
            ["改写查询1", "改写查询2", ...]
        """
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": self.REWRITE_PROMPT.format(query=query)},
                ],
                temperature=0.3,
                max_tokens=256,
            )
            content = response.choices[0].message.content.strip()
            # 按行拆分多个改写
            rewrites = [line.strip("- ").strip() for line in content.split("\n") if line.strip()]
            return rewrites[:3] if rewrites else [query]
        except Exception as e:
            print(f"  ⚠️ 查询改写失败: {e}")
            return [query]

    def generate_hyde_document(self, query: str) -> str:
        """
        HyDE (Hypothetical Document Embeddings)

        生成假设性文档，用「答案的 embedding」去检索「问题相关的文档」
        原理：问题和答案的语义空间可能有 gap，用假设答案做桥梁
        """
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": self.HYDE_PROMPT.format(query=query)},
                ],
                temperature=0.5,
                max_tokens=512,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  ⚠️ HyDE 生成失败: {e}")
            return query


# ==================== RRF 融合 ====================

def rrf_fusion(
    result_groups: list[list[dict]],
    k: int = 60,
    final_pool_size: int = 20,
) -> list[dict]:
    """
    RRF (Reciprocal Rank Fusion) — 多路召回结果融合

    公式:
        RRF_score(d) = Σ 1 / (k + rank_i(d))

    其中:
        k = 60 (平滑常数，降低排名靠前文档的权重优势)
        rank_i(d) = 文档 d 在第 i 路召回中的排名 (从 1 开始)

    优势:
        - 不需要分数归一化（各路召回分数尺度不同也没关系）
        - 排名靠前的文档自然获得更高权重
        - 出现在多路召回中的文档有加成

    Args:
        result_groups: 多路召回结果列表
            [[{"content": "...", "score": ..., "source": "original"}, ...], ...]
        k: RRF 平滑常数
        final_pool_size: 融合后保留的候选数

    Returns:
        融合后的结果列表（按 RRF 分数降序）
    """
    # 使用内容 hash 作为去重 key（避免前100字符相同的不同文档被误去重）
    def _dedup_key(content: str) -> str:
        return hashlib.md5(content.strip().encode()).hexdigest()

    rrf_scores: dict[str, float] = {}
    doc_map: dict[str, dict] = {}

    for group in result_groups:
        for rank, doc in enumerate(group, start=1):
            key = _dedup_key(doc.get("content", ""))
            rrf_score = 1.0 / (k + rank)

            if key in rrf_scores:
                rrf_scores[key] += rrf_score
                # 保留排名最好的 metadata
                if doc.get("score", 0) > doc_map[key].get("score", 0):
                    doc_map[key] = doc
            else:
                rrf_scores[key] = rrf_score
                doc_map[key] = doc

    # 按 RRF 分数排序
    sorted_keys = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

    fused = []
    for key in sorted_keys[:final_pool_size]:
        doc = doc_map[key].copy()
        doc["rrf_score"] = rrf_scores[key]
        fused.append(doc)

    return fused


# ==================== 完整检索引擎 ====================

class RetrievalEngine:
    """
    多路召回 + RRF 融合 + Reranker 精排 完整引擎

    使用方式:
        engine = RetrievalEngine(vector_searcher, bm25_retriever)
        results = engine.retrieve("如何使用 Python 处理数据？")
    """

    def __init__(
        self,
        vector_searcher: Callable,   # fn(query, top_k) → list[dict]
        bm25_retriever: Optional[BM25Retriever] = None,
        embedding_client: Optional[QwenEmbeddingClient] = None,
        reranker_client: Optional[QwenRerankerClient] = None,
    ):
        self.vector_searcher = vector_searcher
        self.bm25 = bm25_retriever
        self.embedding = embedding_client or QwenEmbeddingClient()
        self.reranker = reranker_client or QwenRerankerClient()
        self.query_processor = QueryProcessor()
        self.config = RETRIEVAL_CONFIG

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        enable_rewrite: bool = None,
        enable_hyde: bool = None,
        enable_bm25: bool = None,
    ) -> list[dict]:
        """
        完整检索流程：多路召回 → RRF 融合 → Reranker 精排

        Args:
            query: 用户查询
            top_k: 最终返回数量
            enable_rewrite: 是否启用查询改写
            enable_hyde: 是否启用 HyDE
            enable_bm25: 是否启用 BM25

        Returns:
            [{"content": "...", "metadata": {...}, "rerank_score": 0.95, "source": "..."}, ...]
        """
        top_k = top_k or self.config["rerank_top_k"]
        enable_rewrite = enable_rewrite if enable_rewrite is not None else self.config["enable_query_rewrite"]
        enable_hyde = enable_hyde if enable_hyde is not None else self.config["enable_hyde"]
        enable_bm25 = enable_bm25 if enable_bm25 is not None else self.config["enable_bm25"]

        all_result_groups = []

        # ========== 第 1 路：原始查询向量召回 ==========
        print(f"  ├─ 原始查询召回 (top_k={self.config['recall_original_top_k']})")
        original_results = self.vector_searcher(query, self.config["recall_original_top_k"])
        for r in original_results:
            r["source"] = "original"
        all_result_groups.append(original_results)
        print(f"  │   召回 {len(original_results)} 条")

        # ========== 第 2 路：查询改写召回 ==========
        if enable_rewrite:
            print(f"  ├─ 查询改写召回 (top_k={self.config['recall_rewrite_top_k']})")
            rewritten_queries = self.query_processor.rewrite_query(query)
            rewrite_results = []
            for rq in rewritten_queries[:3]:
                results = self.vector_searcher(rq, self.config["recall_rewrite_top_k"])
                for r in results:
                    r["source"] = f"rewrite"
                rewrite_results.extend(results)
            # 去重（按 content hash，与前 100 字符不同，避免模板头部碰撞）
            seen = set()
            deduped = []
            for r in rewrite_results:
                key = hashlib.md5(r["content"].strip().encode()).hexdigest()
                if key not in seen:
                    seen.add(key)
                    deduped.append(r)
            all_result_groups.append(deduped)
            print(f"  │   召回 {len(deduped)} 条（去重后，改写为：{rewritten_queries[0][:50]}...）")

        # ========== 第 3 路：HyDE 召回 ==========
        if enable_hyde:
            print(f"  ├─ HyDE 召回 (top_k={self.config['recall_hyde_top_k']})")
            hyde_doc = self.query_processor.generate_hyde_document(query)
            hyde_results = self.vector_searcher(hyde_doc, self.config["recall_hyde_top_k"])
            for r in hyde_results:
                r["source"] = "hyde"
            all_result_groups.append(hyde_results)
            print(f"  │   召回 {len(hyde_results)} 条（假设文档: {hyde_doc[:80]}...）")

        # ========== 第 4 路：BM25 关键词检索 ==========
        if enable_bm25 and self.bm25:
            print(f"  ├─ BM25 关键词召回 (top_k={self.config['recall_bm25_top_k']})")
            bm25_results = self.bm25.search(query, self.config["recall_bm25_top_k"])
            for r in bm25_results:
                r["source"] = r.get("source", "bm25")
            all_result_groups.append(bm25_results)
            print(f"  │   召回 {len(bm25_results)} 条")
        elif enable_bm25 and not self.bm25:
            print(f"  ├─ BM25 召回跳过（索引为空）")

        # ========== RRF 融合 ==========
        print(f"  ├─ RRF 融合 (k={self.config['rrf_k']}, pool={self.config['rrf_final_pool_size']})")
        fused = rrf_fusion(
            all_result_groups,
            k=self.config["rrf_k"],
            final_pool_size=self.config["rrf_final_pool_size"],
        )
        print(f"  │   融合后候选 {len(fused)} 条")

        if not fused:
            print(f"  └─ 无结果")
            return []

        # ========== Reranker 精排 ==========
        print(f"  ├─ Qwen3-8B Reranker 精排 (top_k={top_k})")
        documents = [doc["content"] for doc in fused]
        ranked = self.reranker.rerank(query, documents, top_k=top_k)

        # 组装最终结果
        final_results = []
        for item in ranked:
            idx = item["index"]
            if idx < len(fused):
                doc = fused[idx].copy()
                doc["rerank_score"] = item["score"]
                final_results.append(doc)

        # 过滤低分结果
        min_score = self.config["rerank_min_score"]
        filtered = [r for r in final_results if r.get("rerank_score", 0) >= min_score]

        print(f"  └─ 最终返回 {len(filtered)} 条（过滤前 {len(final_results)}，最低分 {min_score}）")

        return filtered


# ==================== 工具函数 ====================

def _rerank_fallback(documents: list[str], top_k: int) -> list[dict]:
    """Reranker 降级：返回原始顺序的文档列表"""
    return [
        {"index": i, "content": doc, "score": 0.5}
        for i, doc in enumerate(documents[:top_k])
    ]
