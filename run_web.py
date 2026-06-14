#!/usr/bin/env python3
"""
启动 All-Around Assistant Web 服务器

使用方式:
  python run_web.py              # 默认 http://0.0.0.0:8000
  python run_web.py --port 9000  # 自定义端口
"""
import sys
import os

# Windows 终端编码修复
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import uvicorn
from server.settings import WEB_CONFIG


def main():
    parser = argparse.ArgumentParser(description="All-Around Assistant Web Server")
    parser.add_argument("--host", default=WEB_CONFIG["host"], help="绑定地址")
    parser.add_argument("--port", type=int, default=WEB_CONFIG["port"], help="端口号")
    parser.add_argument("--reload", action="store_true", help="代码热重载（开发用）")
    args = parser.parse_args()

    print(f"""
  All-Around Assistant Web Server
  ================================
  API:  http://{args.host}:{args.port}/api/health
  UI:   http://{args.host}:{args.port}
  Docs: http://{args.host}:{args.port}/docs
""")

    uvicorn.run(
        "server.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
