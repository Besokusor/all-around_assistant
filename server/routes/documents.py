"""
文档管理路由
"""
import os
import tempfile
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File

from memory.manager import MemoryManager
from config import DOCUMENT_CONFIG

router = APIRouter(tags=["documents"])


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档到知识库"""
    if not file.filename:
        raise HTTPException(400, "No file provided")

    # 检查扩展名
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in DOCUMENT_CONFIG["supported_formats"]:
        raise HTTPException(
            400,
            f"不支持的文件格式「{ext}」，支持：{', '.join(DOCUMENT_CONFIG['supported_formats'])}",
        )

    # 检查大小
    content = await file.read()
    max_size = DOCUMENT_CONFIG["max_file_size_mb"] * 1024 * 1024
    if len(content) > max_size:
        raise HTTPException(
            400,
            f"文件过大（{len(content) / 1024 / 1024:.1f}MB），最大 {DOCUMENT_CONFIG['max_file_size_mb']}MB",
        )

    # 写入临时文件
    suffix = ext if ext else ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        manager = MemoryManager.get_instance()
        result = manager.upload_document(tmp_path)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        # 清理临时文件
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@router.get("/documents")
async def list_documents():
    """列出知识库中所有文档"""
    try:
        manager = MemoryManager.get_instance()
        docs = manager.list_documents()
        return {"documents": docs}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/documents/{name:path}")
async def delete_document(name: str):
    """删除指定文档"""
    try:
        manager = MemoryManager.get_instance()
        result = manager.delete_document(name)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))
