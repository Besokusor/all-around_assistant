"""
聊天路由 — SSE 流式返回 + OSS 图片上传
"""
import asyncio as _asyncio
import uuid
from fastapi import APIRouter, Depends, HTTPException, Form, UploadFile, File
from fastapi.responses import StreamingResponse
from typing import Optional

from server.dependencies import get_session_manager
from server.session_manager import SessionManager
from server.sse import sse_node_event, sse_error
from config import OSS_CONFIG

router = APIRouter(tags=["chat"])


def _upload_to_oss(file_bytes: bytes, filename: str) -> str:
    """上传图片到阿里云 OSS，返回带签名的临时 URL（1小时有效）"""
    import oss2

    auth = oss2.Auth(OSS_CONFIG["access_key_id"], OSS_CONFIG["access_key_secret"])
    bucket = oss2.Bucket(auth, OSS_CONFIG["endpoint"], OSS_CONFIG["bucket_name"])

    ext = filename.rsplit(".", 1)[-1] if "." in filename else "jpg"
    object_name = f"chat-images/{uuid.uuid4().hex}.{ext}"

    bucket.put_object(object_name, file_bytes)

    # 生成签名 URL（1小时有效），确保外部可访问
    return bucket.sign_url("GET", object_name, 3600)


@router.post("/chat")
async def chat(
    message: str = Form(...),
    session_id: str = Form(...),
    image: Optional[UploadFile] = File(None),
    sm: SessionManager = Depends(get_session_manager),
):
    """聊天接口 — SSE 流式返回"""
    session = await sm.get_session(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")

    # 图片上传到 OSS
    image_source = None
    if image and image.filename:
        try:
            content = await image.read()
            if content:
                image_source = _upload_to_oss(content, image.filename)
        except Exception as e:
            raise HTTPException(400, f"图片上传失败: {e}")

    # 入队
    try:
        response_queue = await sm.enqueue_request(session_id, message, image_source)
    except ValueError as e:
        raise HTTPException(404, str(e))

    # SSE 流式响应
    async def event_generator():
        yield sse_node_event("处理中", "请求已接收")
        while True:
            try:
                event = await _asyncio.wait_for(response_queue.get(), timeout=120)
                yield event
                if '"type":"done"' in event or '"type":"error"' in event:
                    break
            except _asyncio.TimeoutError:
                yield sse_error("请求处理超时，请重试")
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
