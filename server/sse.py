"""
SSE (Server-Sent Events) 格式化工具

事件类型：
  - node_event: 图节点转换事件
  - token: 流式文本块
  - done: 完成
  - error: 异常
"""
import json
from typing import Optional


def sse_event(event_type: str, data: dict) -> str:
    """格式化为一条 SSE 事件"""
    payload = {"type": event_type, **data}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def sse_node_event(node: str, content: str) -> str:
    """节点转换事件"""
    return sse_event("node_event", {"node": node, "content": content})


def sse_token(content: str) -> str:
    """流式 token"""
    return sse_event("token", {"content": content})


def sse_done(content: str) -> str:
    """完成事件"""
    return sse_event("done", {"content": content})


def sse_error(message: str, detail: Optional[str] = None) -> str:
    """错误事件"""
    return sse_event("error", {"content": message, "detail": detail})
