"""
FastAPI 主应用 — 组装路由、中间件、静态文件
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from server.settings import WEB_CONFIG
from server.dependencies import get_agent, get_session_manager
from server.routes import chat, sessions, documents, knowledge
from logger import init_logger, get_logger
from config import LOG_CONFIG


@asynccontextmanager
async def lifespan(app: FastAPI):
    # === 启动 ===
    # 初始化日志系统
    init_logger(
        log_dir=LOG_CONFIG.get("log_dir", "./log"),
        level=LOG_CONFIG.get("level", "INFO"),
    )
    logger = get_logger()

    print("🚀 启动 All-Around Assistant Web Server...")
    if logger:
        logger.info("Web 服务器启动中...")

    agent = await get_agent()
    print(f"  ├─ Agent: {agent.user_id}")
    print(f"  ├─ 工具: {len(agent.tool_registry.list_tools())} 个")
    print(f"  └─ Graph: 就绪")

    if logger:
        logger.info(
            f"Web 服务器已启动 │ agent={agent.user_id} "
            f"tools={len(agent.tool_registry.list_tools())} "
            f"log_dir={LOG_CONFIG.get('log_dir')}"
        )

    sm = await get_session_manager()
    # 启动后台清理任务
    cleanup_task = asyncio.create_task(sm._cleanup_loop())

    yield

    # === 关闭 ===
    cleanup_task.cancel()
    if logger:
        logger.info("Web 服务器关闭")
    print("👋 服务器关闭")


app = FastAPI(
    title="All-Around Assistant",
    description="个人全能助手 Web API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=WEB_CONFIG["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 路由
app.include_router(chat.router, prefix="/api")
app.include_router(sessions.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(knowledge.router, prefix="/api")

# 静态文件（必须在 API 路由之后，否则会覆盖）
import os
static_dir = os.path.abspath(WEB_CONFIG["static_dir"])

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}

# 挂载静态文件
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
