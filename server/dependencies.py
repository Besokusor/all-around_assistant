"""
FastAPI 依赖注入 — 管理 agent 和 session manager 单例
"""
from typing import Optional

from agent import AllAroundAssistant
from server.session_manager import SessionManager

_agent: Optional[AllAroundAssistant] = None
_session_manager: Optional[SessionManager] = None


async def get_agent() -> AllAroundAssistant:
    """获取 agent 单例"""
    global _agent
    if _agent is None:
        _agent = AllAroundAssistant(stream_enabled=True)
    return _agent


async def get_session_manager() -> SessionManager:
    """获取 session manager 单例"""
    global _session_manager, _agent
    if _session_manager is None:
        if _agent is None:
            _agent = AllAroundAssistant(stream_enabled=True)
        _session_manager = SessionManager(_agent)
    return _session_manager
