"""
会话管理器 — 内存存储 + 异步队列

每个 session 一个 asyncio.Queue，确保同一会话的请求串行执行，
避免 LangGraph checkpoint 状态冲突。
"""
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from langchain_core.messages import HumanMessage, AIMessage

from state import create_initial_state
from server.sse import sse_node_event, sse_token, sse_done, sse_error
from config import EXECUTION_CONFIG
from logger import get_logger, set_trace_id

if TYPE_CHECKING:
    from agent import AllAroundAssistant


@dataclass
class Session:
    session_id: str
    user_id: str
    created_at: float
    last_access: float
    messages: list = field(default_factory=list)
    turn_count: int = 0              # 对话轮次（触发长期记忆压缩）

    @property
    def message_count(self) -> int:
        return len(self.messages) // 2


class SessionManager:
    """
    会话管理器

    - _sessions: session_id → Session
    - _queues: session_id → asyncio.Queue（请求队列）
    - _workers: session_id → asyncio.Task（消费者协程）
    """

    def __init__(self, agent: "AllAroundAssistant"):
        self._agent = agent
        self._sessions: dict[str, Session] = {}
        self._queues: dict[str, asyncio.Queue] = {}
        self._workers: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None

    # ==================== Session CRUD ====================

    async def create_session(self, user_id: str = "default_user") -> Session:
        async with self._lock:
            if len(self._sessions) >= 20:
                # 清理最老的 session
                oldest = min(self._sessions.values(), key=lambda s: s.last_access)
                await self.delete_session(oldest.session_id)

            sid = str(uuid.uuid4())[:8]
            session = Session(
                session_id=sid,
                user_id=user_id,
                created_at=time.time(),
                last_access=time.time(),
            )
            self._sessions[sid] = session
            self._queues[sid] = asyncio.Queue()
            self._workers[sid] = asyncio.create_task(self._session_worker(sid))
            return session

    async def get_session(self, session_id: str) -> Optional[Session]:
        session = self._sessions.get(session_id)
        if session:
            session.last_access = time.time()
        return session

    async def list_sessions(self) -> list[dict]:
        result = []
        for s in self._sessions.values():
            preview = ""
            for msg in reversed(s.messages):
                if isinstance(msg, HumanMessage):
                    preview = getattr(msg, "content", "")[:50]
                    break
            result.append({
                "session_id": s.session_id,
                "user_id": s.user_id,
                "created_at": s.created_at,
                "message_count": s.message_count,
                "preview": preview,
            })
        result.sort(key=lambda x: x["created_at"], reverse=True)
        return result

    async def delete_session(self, session_id: str) -> bool:
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            if not session:
                return False
            queue = self._queues.pop(session_id, None)
            worker = self._workers.pop(session_id, None)
            if worker:
                worker.cancel()
            if queue:
                # 清空队列，避免生产者挂起
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            return True

    # ==================== 请求队列 ====================

    async def enqueue_request(
        self, session_id: str, message: str, image_source: Optional[str]
    ) -> asyncio.Queue:
        """将 chat 请求入队，返回一个 response_queue 用于接收 SSE 事件"""
        response_queue: asyncio.Queue = asyncio.Queue()
        request = {
            "message": message,
            "image_source": image_source,
            "response_queue": response_queue,
        }
        queue = self._queues.get(session_id)
        if not queue:
            raise ValueError(f"Session {session_id} not found")
        await queue.put(request)
        return response_queue

    # ==================== 会话工作协程 ====================

    async def _session_worker(self, session_id: str):
        """每个 session 一个 worker，串行消费请求队列"""
        queue = self._queues.get(session_id)
        if not queue:
            return

        session = self._sessions.get(session_id)
        if not session:
            return

        node_icons = {
            "memory_retrieve": "🧠 记忆检索",
            "planner": "💡 思考规划",
            "executor": "⚡ 执行",
            "observer": "👁️ 观察评估",
            "memory_update": "💾 记忆更新",
            "response": "💬 生成回复",
            "fallback": "🆘 兜底",
        }

        while True:
            try:
                request = await queue.get()
                message = request["message"]
                image_source = request.get("image_source")
                response_queue: asyncio.Queue = request["response_queue"]

                # 为本轮请求设置 trace_id
                import uuid as _uuid
                trace_id = str(_uuid.uuid4())[:8]
                set_trace_id(trace_id)

                web_logger = get_logger()
                print(f"[Worker {session_id[:6]}] 收到请求: {message[:40]}...")
                if web_logger:
                    web_logger.trace_request_start(
                        user_input=message,
                        user_id=session.user_id,
                        image_source=image_source,
                    )

                try:
                    # === 快速分类：简单对话 or 需要工具/复杂推理 ===
                    # 有图片/对话历史/工具关键词 → 走完整 graph
                    has_history = len(session.messages) >= 2
                    needs_tools = bool(image_source) or has_history or self._needs_tools(message)

                    if not needs_tools:
                        # 简单闲聊 → 跳过 graph，直接流式 LLM 回应（快）
                        print(f"[Worker {session_id[:6]}] 简单对话，直接回应")
                        if web_logger:
                            web_logger.info(
                                f"快速路径 [trace={trace_id}] │ "
                                f"session={session_id[:6]} msg={message[:40]}"
                            )
                        collected_response = ""
                        stream_llm = self._make_stream_llm()
                        async for chunk in stream_llm.astream([
                            HumanMessage(content=(
                                "【系统】你是个人全能助手。简短友好地回复用户。用中文。\n"
                                f"【用户】{message}"
                            )),
                        ]):
                            token = chunk.content if hasattr(chunk, "content") else str(chunk)
                            if token:
                                collected_response += token
                                await response_queue.put(sse_token(token))

                        await response_queue.put(sse_done(collected_response or "嗯？"))
                        session.messages.append(HumanMessage(content=message))
                        session.messages.append(AIMessage(content=collected_response))
                        if len(session.messages) > 40:
                            session.messages = session.messages[-40:]
                        session.last_access = time.time()
                        print(f"[Worker {session_id[:6]}] 快速回应完成，{len(collected_response)} 字符")
                        if web_logger:
                            web_logger.trace_request_end(
                                total_ms=(time.time() - trace_start) * 1000 if 'trace_start' in dir() else 0,
                                steps=0,
                                tools_called=0,
                                response_len=len(collected_response),
                                final_response=collected_response,
                            )
                        continue

                    # === 复杂任务 → 完整 LangGraph 流程（含 token 级流式） ===
                    # 构建 AgentState，把图片 URL 注入到用户输入中
                    msg_with_image = message
                    if image_source:
                        msg_with_image = f"{message}\n[用户上传的图片URL：{image_source}]"

                    # 注入对话轮次 + 图片 URL 到短期记忆
                    session.turn_count += 1
                    ctx = {"turn_count": session.turn_count}
                    if image_source:
                        ctx["image_source"] = image_source

                    state = create_initial_state(
                        user_input=msg_with_image,
                        user_id=session.user_id,
                        stream_enabled=True,
                        max_retries=EXECUTION_CONFIG["max_retries"],
                    )
                    state["messages"] = list(session.messages) + [
                        HumanMessage(content=msg_with_image)
                    ]
                    state["short_term_context"] = ctx

                    config = {"configurable": {"thread_id": session_id}}

                    # === 统一 astream：进度事件 + 逐 token 流式 ===
                    import time as _time
                    trace_start = _time.time()
                    trace_nodes = []
                    collected_response = ""

                    async for event in self._agent.graph.astream(
                        state, config,
                        stream_mode=["messages", "updates"],
                    ):
                        # LangGraph 1.x 多 mode stream 格式：
                        #   3-tuple: (namespace, mode, data) — namespace 通常是 ()
                        #   2-tuple: (mode, data) — 旧版兼容
                        if len(event) == 3:
                            _, mode, data = event
                        elif len(event) == 2:
                            mode, data = event
                        else:
                            continue

                        if mode == "updates":
                            # 节点完成事件 → SSE 进度通知
                            if isinstance(data, dict):
                                for node_name, node_output in data.items():
                                    if node_name == "response":
                                        # response 节点不发送进度事件（token 流式推送）
                                        continue

                                    label = node_icons.get(node_name, node_name)
                                    key = ""
                                    if isinstance(node_output, dict):
                                        logs = node_output.get("execution_log", [])
                                        tr = node_output.get("tool_results") or []
                                        key = str(logs[-1])[:100] if logs else ""
                                        trace_nodes.append({
                                            "node": node_name,
                                            "tools": [t.get("tool_name", "?") for t in tr if isinstance(t, dict)],
                                        })
                                    await response_queue.put(sse_node_event(label, key))

                        elif mode == "messages":
                            # LLM token chunk → 仅推送 response 节点的 token
                            if isinstance(data, (list, tuple)) and len(data) >= 2:
                                chunk, metadata = data[0], data[1]
                                node_name = metadata.get("langgraph_node", "")
                                if node_name == "response":
                                    token = chunk.content if hasattr(chunk, "content") else ""
                                    if token:
                                        collected_response += token
                                        await response_queue.put(sse_token(token))

                    # === Trace 日志 ===
                    total_ms = round((_time.time() - trace_start) * 1000)
                    tool_calls = []
                    for n in trace_nodes:
                        tool_calls.extend(n["tools"])
                    trace = {
                        "sid": session_id[:6],
                        "total_ms": total_ms,
                        "nodes": [f"{n['node']}" + (f" {'|'.join(n['tools'])}" if n["tools"] else "") for n in trace_nodes],
                        "tools_called": len(tool_calls),
                        "input": message[:40],
                    }
                    print(f"[Trace] {json.dumps(trace, ensure_ascii=False)}")

                    if web_logger:
                        web_logger.trace_request_end(
                            total_ms=total_ms,
                            steps=len(trace_nodes),
                            tools_called=len(tool_calls),
                            response_len=len(collected_response),
                            final_response=collected_response,
                        )

                    # 兜底：如果没有收集到流式 token（如 fallback 直接设置了 final_response）
                    if not collected_response:
                        final_state = self._agent.graph.get_state(config)
                        if final_state and hasattr(final_state, 'values'):
                            collected_response = final_state.values.get("final_response", "") or "（未收到回复）"
                        else:
                            collected_response = "（未收到回复）"

                    print(f"[Worker {session_id[:6]}] 完成，回复 {len(collected_response)} 字符")
                    await response_queue.put(sse_done(collected_response))

                    # 同步压缩后的 turn_count 回 session
                    final_state = self._agent.graph.get_state(config)
                    if final_state and hasattr(final_state, 'values'):
                        updated_ctx = final_state.values.get("short_term_context", {})
                        session.turn_count = updated_ctx.get("turn_count", session.turn_count)
                        if session.turn_count == 0:
                            print(f"[Worker {session_id[:6]}] 长期记忆已压缩")

                    # 更新 session
                    session.messages.append(HumanMessage(content=message))
                    session.messages.append(AIMessage(content=collected_response))
                    if len(session.messages) > 40:
                        session.messages = session.messages[-40:]
                    session.last_access = time.time()

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    if web_logger:
                        web_logger.error(
                            f"Worker 异常 [trace={trace_id}] │ "
                            f"session={session_id[:6]} error={e}"
                        )
                    try:
                        await response_queue.put(sse_error(str(e)))
                    except Exception:
                        print(f"[Worker] 发送错误事件失败: {e}")

            except asyncio.CancelledError:
                break
            except Exception:
                import traceback
                traceback.print_exc()

    # ==================== 快速分类 + 工具方法 ====================

    @staticmethod
    def _has_tool_keywords(message: str) -> bool:
        """关键词预检：检测是否明确需要工具或外部信息"""
        tool_kw = [
            # 搜索类
            "搜索", "查一下", "帮我找", "搜", "最新",
            # 计算类
            "计算", "等于", "多少", "等于几", "是多少",
            "+", "*", "/",
            # 天气类
            "天气", "气温", "下雨", "下雪", "温度",
            # 时间类（LLM 无法知道实时时间，需要工具）
            "今天星期几", "今天几号", "现在几点", "几点了",
            "今天日期", "今天是什么日子",
            # 出行类
            "出行", "怎么走", "怎么去", "高铁", "飞机", "火车",
            "多远", "多久到", "路线",
            # 菜谱类
            "菜谱", "做什么菜", "怎么做", "食材", "食谱",
            # CLI 类
            "ls ", "dir ", "git ", "pip ", "python ",
            # 文档类
            "上传", "文档", "知识库",
        ]
        msg = message.lower()
        return any(kw in msg for kw in tool_kw)

    @staticmethod
    def _is_chitchat(message: str) -> bool:
        """关键词预检：是否明显是闲聊（不需要工具也不需要外部知识）"""
        chitchat_patterns = [
            "你好", "嗨", "hello", "hi", "hey",
            "谢谢", "辛苦了", "再见", "拜拜", "晚安", "早安",
            "你是谁", "你叫什么", "你能做什么", "你会什么",
            "哈哈", "有意思", "不错", "好的", "ok",
            "无聊", "讲个笑话", "笑话",
        ]
        msg = message.lower().strip()
        return any(p in msg for p in chitchat_patterns) and len(msg) < 30

    def _needs_tools(self, message: str) -> bool:
        """
        快速判断是否需要工具/复杂推理。
        - 明确工具关键词 → True（需要 graph）
        - 明显日常闲聊 → False（直接 LLM 回应）
        - 其他 → False（默认走快速路径，避免 graph 开销）
        """
        if self._has_tool_keywords(message):
            return True
        if self._is_chitchat(message):
            return False
        # 默认走快速路径（LLM 直接回答大多数简单问题也足以应付）
        return False

    def _make_stream_llm(self):
        """创建流式 LLM 实例（用于简单对话快速路径）"""
        from langchain_openai import ChatOpenAI
        from config import LLM_CONFIG as _LLM
        return ChatOpenAI(
            model=_LLM["model"],
            temperature=0.7,
            api_key=_LLM["api_key"],
            base_url=_LLM["base_url"],
            max_tokens=2048,
            streaming=True,
        )

    # ==================== 清理 ====================

    async def _cleanup_loop(self):
        """后台清理过期 session"""
        from server.settings import WEB_CONFIG

        ttl = WEB_CONFIG["session_ttl_seconds"]
        interval = WEB_CONFIG["cleanup_interval"]

        while True:
            await asyncio.sleep(interval)
            now = time.time()
            expired = []
            async with self._lock:
                for sid, s in self._sessions.items():
                    if now - s.last_access > ttl:
                        expired.append(sid)
            for sid in expired:
                await self.delete_session(sid)
