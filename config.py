"""
全局配置 — 个人全能助手 Agent
所有涉密信息从 .env 读取，不硬编码
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ==================== LLM 配置 ====================
LLM_CONFIG = {
    "model": "deepseek-v4-flash",
    "think_model": "deepseek-v4-pro",
    "vision_model": "deepseek-v4-flash",
    "temperature": 0.3,
    "max_tokens": 4096,
    "api_key": os.getenv("DEEPSEEK_API_KEY"),
    "base_url": "https://api.deepseek.com",
}

# ==================== 讯飞 MaaS — Embedding / Reranker / Vision ====================
QWEN_MAAS_CONFIG = {
    "api_key": os.getenv("QWEN_MAAS_API_KEY"),
    "base_url": "https://maas-api.cn-huabei-1.xf-yun.com/v2",
    "embedding_model": "xop3qwen8bembedding",
    "reranker_model": "xop3qwen8breranker",
    "vision_model": "xop3qwen32bvl",
    "embedding_dim": 768,
    "max_batch_size": 32,
    "timeout": 30,
}

# ==================== Tavily 搜索 ====================
TAVILY_CONFIG = {
    "api_key": os.getenv("TAVILY_API_KEY"),
    "search_depth": "advanced",
    "max_results": 5,
    "include_answer": True,
}

# ==================== MySQL — 用户画像 ====================
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "agent_memory"),
    "charset": "utf8mb4",
    "pool_size": 5,
}

# ==================== Chroma 向量库 ====================
CHROMA_CONFIG = {
    "persist_directory": "./chroma_db",
    "collection_long_term": "long_term_memory",
    "collection_knowledge": "knowledge_base",
    "collection_documents": "user_documents",
}

# ==================== 阿里云 OSS — 图片存储 ====================
OSS_CONFIG = {
    "access_key_id": os.getenv("OSS_ACCESS_KEY_ID"),
    "access_key_secret": os.getenv("OSS_ACCESS_KEY_SECRET"),
    "bucket_name": os.getenv("OSS_BUCKET_NAME"),
    "endpoint": os.getenv("OSS_ENDPOINT"),
    "base_url": f"https://{os.getenv('OSS_BUCKET_NAME', '')}.{os.getenv('OSS_ENDPOINT', '')}",
}

# ==================== 文档处理 ====================
DOCUMENT_CONFIG = {
    "chunk_method": "recursive",
    "chunk_size": 512,
    "chunk_overlap": 64,
    "separators": ["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    "semantic_threshold": 0.7,
    "supported_formats": [".pdf", ".txt", ".md", ".docx", ".csv", ".json", ".py", ".js", ".ts"],
    "max_file_size_mb": 50,
}

# ==================== 多路召回 + RRF + 重排序 ====================
RETRIEVAL_CONFIG = {
    "recall_original_top_k": 10,
    "recall_rewrite_top_k": 10,
    "recall_hyde_top_k": 10,
    "recall_bm25_top_k": 10,
    "enable_query_rewrite": True,
    "enable_hyde": True,
    "enable_bm25": True,
    "rrf_k": 60,
    "rrf_final_pool_size": 20,
    "rerank_top_k": 5,
    "rerank_min_score": 0.3,
}

# ==================== 执行配置 ====================
EXECUTION_CONFIG = {
    "max_retries": 3,
    "max_plan_revisions": 2,
    "max_steps": 10,
    "tool_timeout": 30,
    "stream_chunk_size": 50,
}

# ==================== 记忆压缩 ====================
MEMORY_CONFIG = {
    "short_term_max_tokens": 4000,
    "long_term_top_k": 5,
    "summary_trigger_turns": 10,
    "user_profile_update_threshold": 3,
}

# ==================== 兜底 / 降级 ====================
FALLBACK_CONFIG = {
    "enable_human_escalation": True,
    "degraded_tools": ["web_search", "calculator"],
    "cache_ttl_fallback": 300,
}

# ==================== 日志配置 ====================
LOG_CONFIG = {
    "log_dir": "./log",
    "level": "INFO",          # DEBUG | INFO | WARNING | ERROR
    "console": True,          # 是否同步输出到控制台
    "retention_days": 30,     # 日志保留天数
}

# ==================== 天气城市坐标 ====================
CITY_COORDINATES = {
    "北京": (39.9042, 116.4074),   "上海": (31.2304, 121.4737),
    "广州": (23.1200, 113.3271),   "深圳": (22.6272, 114.0737),
    "武汉": (30.5928, 114.3055),   "杭州": (30.2741, 120.1551),
    "成都": (30.5728, 104.0668),   "南京": (32.0603, 118.7969),
    "重庆": (29.4316, 106.9123),   "西安": (34.3416, 108.9398),
    "长沙": (28.2282, 112.9388),   "郑州": (34.7466, 113.6253),
    "天津": (39.3434, 117.3616),   "苏州": (31.2990, 120.5853),
}
