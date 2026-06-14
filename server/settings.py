"""
Web 层配置 — FastAPI 服务器
与 config.py 分离，仅包含 Web 相关配置
"""
import os

WEB_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "session_ttl_seconds": 3600,         # 空闲 session 自动过期
    "max_sessions": 20,                   # 最大 session 数
    "max_upload_size_mb": 50,            # 最大上传文件
    "static_dir": os.path.join(os.path.dirname(__file__), "static"),
    "cors_origins": ["*"],               # 生产环境应限制
    "cleanup_interval": 300,             # 过期 session 清理间隔（秒）
}
