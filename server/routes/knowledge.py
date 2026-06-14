"""
知识问答路由
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from memory.manager import MemoryManager
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import LLM_CONFIG

router = APIRouter(tags=["knowledge"])


class KnowledgeQARequest(BaseModel):
    query: str


@router.post("/knowledge/qa")
async def knowledge_qa(req: KnowledgeQARequest):
    """基于已上传文档的知识问答"""
    try:
        manager = MemoryManager.get_instance()

        if manager.long_term.get_document_count() == 0:
            return {
                "answerable": False,
                "query": req.query,
                "answer": "知识库为空。请先上传文档。",
                "sources": [],
            }

        qa_result = manager.knowledge_qa(req.query)

        if not qa_result["answerable"]:
            return {
                "answerable": False,
                "query": req.query,
                "answer": "未找到相关内容，请尝试换一种表述。",
                "sources": [],
            }

        # LLM 生成回答
        llm = ChatOpenAI(
            model=LLM_CONFIG["model"],
            temperature=0.5,
            api_key=LLM_CONFIG["api_key"],
            base_url=LLM_CONFIG["base_url"],
            max_tokens=2048,
        )

        system_prompt = (
            "你是一个知识助手。根据提供的文档内容回答用户问题。\n"
            "严格基于文档内容回答，不要编造信息。在回答末尾标注来源。"
        )

        msg = HumanMessage(content=(
            f"【文档内容】\n{qa_result['context']}\n\n"
            f"【用户问题】{req.query}\n\n请基于以上文档内容回答问题。"
        ))

        response = llm.invoke([SystemMessage(content=system_prompt), msg])

        return {
            "answerable": True,
            "query": req.query,
            "answer": response.content.strip(),
            "sources": qa_result["sources"],
        }

    except Exception as e:
        raise HTTPException(500, str(e))
