"""
会话管理路由
"""
from fastapi import APIRouter, Depends, HTTPException

from server.dependencies import get_session_manager
from server.session_manager import SessionManager

router = APIRouter(tags=["sessions"])


@router.get("/sessions")
async def list_sessions(sm: SessionManager = Depends(get_session_manager)):
    """列出所有会话"""
    sessions = await sm.list_sessions()
    return {"sessions": sessions}


@router.post("/sessions")
async def create_session(
    user_id: str = "default_user",
    sm: SessionManager = Depends(get_session_manager),
):
    """创建新会话"""
    try:
        session = await sm.create_session(user_id)
        return {
            "session_id": session.session_id,
            "user_id": session.user_id,
            "created_at": session.created_at,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    sm: SessionManager = Depends(get_session_manager),
):
    """删除会话"""
    ok = await sm.delete_session(session_id)
    if not ok:
        raise HTTPException(404, "Session not found")
    return {"deleted": True, "session_id": session_id}
