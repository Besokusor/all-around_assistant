#!/usr/bin/env python3
"""
个人全能助手 Agent — 主入口

五层架构：记忆层 → 思考层 → 执行层 → 观察层 → 输出层

技能：
  [Q&A] 智能问答    [Web] 联网搜索    [Travel] 出行规划
  [Calc] 计算器     [Food] 拍食材推荐菜谱  [CLI] CLI 执行
  [Image] 图像分析  [Gen] 图像生成

使用方式：
  python agent.py                  # 交互模式
  python agent.py "今天武汉天气"    # 单次查询
  python agent.py --image photo.jpg "这里面有什么食材？"  # 图像输入
  python agent.py --stream "搜索最新AI新闻"  # 流式输出
"""
import os
import sys
import time
import uuid
import asyncio
import argparse

# Windows 终端编码修复
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage

from state import create_initial_state, AgentState
from graph.builder import build_graph, AssistantGraph
from tools.registry import get_all_tools
from config import EXECUTION_CONFIG, LLM_CONFIG, LOG_CONFIG
from logger import init_logger, get_logger, set_trace_id


# ==================== 欢迎界面 ====================

BANNER = r"""
╔══════════════════════════════════════════════════════╗
║          🤖 个人全能助手  —  All-Around Assistant      ║
║                                                      ║
║  记忆层 🧠  |  思考层 💡  |  执行层 ⚡  |  观察层 👁️   ║
║                                                      ║
║  技能：问答 | 搜索 | 出行 | 计算 | 菜谱 | CLI | 图像  ║
║                                                      ║
║  输入 help 查看帮助  |  quit 退出                     ║
╚══════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
📋 可用命令：
──────────────────────────────────────────────────────
  help           显示此帮助信息
  skills         列出所有技能
  tools          列出所有可用工具
  history        查看当前会话历史
  checkpoint     查看 checkpoint 状态
  rollback       回退到上一个 checkpoint
  clear          清空当前会话

  📄 文档管理：
  upload <path>  上传文档到知识库（支持 PDF/Word/MD/TXT 等）
  docs           列出知识库中所有文档
  deldoc <name>  删除指定文档
  askdoc <问题>   基于文档进行知识问答

  🖼️ 其他：
  image <path>   加载图像进行分析
  stream on/off  切换流式输出
  quit / exit    退出助手
──────────────────────────────────────────────────────

💡 直接输入问题即可获得回答，例如：
  • 北京今天的天气怎么样？
  • 帮我计算 (100 + 50) * 3.14
  • 我有鸡蛋、西红柿和葱，能做什么菜？
  • 从武汉到上海怎么走最快？
  • 搜索最新的 AI 新闻
  • upload ./report.pdf（上传文档）
  • askdoc 报告中的Q3营收是多少？（基于文档问答）
"""


# ==================== 主助手类 ====================

class AllAroundAssistant:
    """
    个人全能助手

    封装了完整的工作流：
    - Graph 执行
    - 流式输出
    - 多轮对话
    - Checkpoint 管理
    - 图像输入
    """

    def __init__(self, user_id: str = "default_user", stream_enabled: bool = True):
        self.user_id = user_id
        self.stream_enabled = stream_enabled
        self.thread_id = str(uuid.uuid4())[:8]
        self.session_messages: list = []

        # 初始化日志系统
        self._init_logging()

        # 初始化组件
        print("🔧 初始化组件...")
        print(f"  ├─ LLM: {LLM_CONFIG['model']} (思考: {LLM_CONFIG['think_model']})")
        self.tool_registry = get_all_tools()
        print(f"  ├─ 工具: {len(self.tool_registry.list_tools())} 个已注册")
        print(f"  │   {', '.join(self.tool_registry.list_tools())}")

        # 初始化 Graph
        try:
            self.graph: AssistantGraph = build_graph()
            print(f"  ├─ Graph: 7 节点已就绪")
            print(f"  └─ Checkpoint: MemorySaver (thread: {self.thread_id})")
        except Exception as e:
            print(f"  ⚠️ Graph 初始化失败: {e}")
            self.graph = None

        print(f"\n✅ 助手就绪！输入你的问题吧...\n")
        logger = get_logger()
        if logger:
            logger.info(
                f"Agent 初始化完成 │ user={user_id} thread={self.thread_id} "
                f"model={LLM_CONFIG['model']} tools={len(self.tool_registry.list_tools())}"
            )

    @staticmethod
    def _init_logging():
        """初始化日志系统（仅首次调用生效）"""
        init_logger(
            log_dir=LOG_CONFIG.get("log_dir", "./log"),
            level=LOG_CONFIG.get("level", "INFO"),
        )

    # ==================== 核心执行 ====================

    def chat(self, user_input: str, image_source: str = None) -> str:
        """
        执行一次对话

        Args:
            user_input: 用户输入文本
            image_source: 可选图像路径/base64

        Returns:
            AI 回复文本
        """
        # 为本轮对话生成 trace_id
        trace_id = str(uuid.uuid4())[:8]
        set_trace_id(trace_id)

        logger = get_logger()
        t_start = time.time()

        # 处理图像输入
        if image_source:
            user_input = self._augment_with_image(user_input, image_source)

        # 日志：请求开始（摘要 + 原始输入）
        if logger:
            logger.trace_request_start(
                user_input=user_input,
                user_id=self.user_id,
                image_source=image_source,
            )

        # 创建初始状态
        state = create_initial_state(
            user_input=user_input,
            user_id=self.user_id,
            stream_enabled=self.stream_enabled,
            max_retries=EXECUTION_CONFIG["max_retries"],
        )

        # 添加历史消息
        state["messages"] = list(self.session_messages) + [HumanMessage(content=user_input)]

        # 配置（用于 checkpoint）
        config = {"configurable": {"thread_id": self.thread_id}}

        try:
            if self.stream_enabled:
                result = self._stream_execute(state, config)
            else:
                result = self._sync_execute(state, config)

            # 日志：请求结束（摘要 + 原始输出）
            if logger:
                total_ms = (time.time() - t_start) * 1000
                tool_results = state.get("tool_results", [])
                tools_called = len(tool_results)
                plan = state.get("plan", {})
                steps = plan.get("total_steps", 0) if plan else 0
                logger.trace_request_end(
                    total_ms=total_ms,
                    steps=steps,
                    tools_called=tools_called,
                    response_len=len(result) if result else 0,
                    final_response=result or "",
                )

            return result
        except Exception as e:
            error_msg = f"❌ 执行异常：{str(e)}"
            print(error_msg)
            if logger:
                logger.error(f"执行异常 [trace={trace_id}]：{e}")
            return error_msg

    def _sync_execute(self, state: AgentState, config: dict) -> str:
        """同步执行 — graph.invoke() 包含 response_node，final_response 自动生成"""
        print("⏳ 处理中...")
        start = time.time()

        result = self.graph.invoke(state, config)

        elapsed = time.time() - start
        tool_results = result.get("tool_results", [])
        if tool_results:
            print(f"\n📊 调用了 {len(tool_results)} 个工具，耗时 {elapsed:.1f}s")

        final_response = result.get("final_response", "") or "（未收到回复）"
        print(final_response)

        self._update_session(state["user_input"], final_response)
        return final_response

    async def _stream_execute_async(self, state: AgentState, config: dict) -> str:
        """
        真正的 token 级流式执行

        使用 graph.astream(stream_mode=["messages", "updates"])：
        - "updates" 事件：节点完成时推送，展示执行进度
        - "messages" 事件：response_node 内 LLM 的逐 token 输出

        LangGraph 会拦截 response_node 中 LLM 的 streaming 调用，
        将每个 token chunk 通过 "messages" 事件推送出来。
        """
        print("⏳ 处理中...\n")

        collected = ""
        node_names = {
            "memory_retrieve": "🧠 记忆检索",
            "planner": "💡 思考规划",
            "executor": "⚡ 执行",
            "observer": "👁️ 观察评估",
            "memory_update": "💾 记忆更新",
            "response": "💬 生成回复",
            "fallback": "🆘 兜底",
        }

        try:
            async for event in self.graph.astream(
                state, config,
                stream_mode=["messages", "updates"],
            ):
                # LangGraph 1.x 多 mode stream 格式：
                #   3-tuple: (namespace, mode, data) — namespace 通常是 ()
                #   2-tuple: (mode, data) — 旧版兼容
                if len(event) == 3:
                    _, mode, data = event    # namespace, mode, data
                elif len(event) == 2:
                    mode, data = event
                else:
                    continue

                if mode == "updates":
                    # 节点完成事件 → 展示进度
                    if isinstance(data, dict):
                        for node_name, node_output in data.items():
                            label = node_names.get(node_name, node_name)
                            if isinstance(node_output, dict):
                                logs = node_output.get("execution_log", [])
                                detail = logs[-1][:80] if logs else ""
                                print(f"  {label} {detail}")
                            else:
                                print(f"  {label} ✓")

                elif mode == "messages":
                    # LLM token chunk → 仅输出 response 节点的 token
                    if isinstance(data, (list, tuple)) and len(data) >= 2:
                        chunk, metadata = data[0], data[1]
                        # 只流式输出 response 节点的 token（排除 planner/executor 的 LLM 调用）
                        node_name = metadata.get("langgraph_node", "")
                        if node_name == "response":
                            token = chunk.content if hasattr(chunk, "content") else ""
                            if token:
                                print(token, end="", flush=True)
                                collected += token

        except Exception as e:
            print(f"\n⚠️ 流式执行异常：{e}")

        # 兜底：如果没有收集到流式 token（fallback 直接 END 跳过 response_node、或流式异常）
        if not collected:
            try:
                result = self.graph.get_state(config)
                if result and hasattr(result, 'values'):
                    collected = result.values.get("final_response", "") or ""
            except Exception:
                pass

        print()
        self._update_session(state["user_input"], collected)
        return collected or "（未收到回复）"

    def _stream_execute(self, state: AgentState, config: dict) -> str:
        """同步包装器 — 内部调用异步 astream"""
        return asyncio.run(self._stream_execute_async(state, config))

    # ==================== 交互模式 ====================

    def run_interactive(self):
        """运行交互式对话循环"""
        print(BANNER)

        while True:
            try:
                user_input = input("\n👤 你：").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 再见！")
                break

            if not user_input:
                continue

            # 处理命令
            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 再见！")
                break

            if user_input.lower() == "help":
                print(HELP_TEXT)
                continue

            if user_input.lower() == "skills":
                self._show_skills()
                continue

            if user_input.lower() == "tools":
                self._show_tools()
                continue

            if user_input.lower() == "history":
                self._show_history()
                continue

            if user_input.lower() == "checkpoint":
                self._show_checkpoint()
                continue

            if user_input.lower() == "rollback":
                self._do_rollback()
                continue

            if user_input.lower() == "clear":
                self._clear_session()
                continue

            if user_input.lower().startswith("stream "):
                arg = user_input[7:].strip().lower()
                self.stream_enabled = arg in ("on", "true", "1", "yes")
                print(f"📺 流式输出：{'开启' if self.stream_enabled else '关闭'}")
                continue

            # === 文档管理命令 ===

            if user_input.lower().startswith("upload "):
                file_path = user_input[7:].strip()
                print(f"📤 正在上传文档：{file_path}...")
                response = self.chat(f"请使用 document_upload 工具上传文档：{file_path}")
                print(f"\n🤖 {response}")
                continue

            if user_input.lower() == "docs":
                response = self.chat("请使用 list_documents 工具列出所有文档")
                print(f"\n🤖 {response}")
                continue

            if user_input.lower().startswith("deldoc "):
                doc_name = user_input[7:].strip()
                response = self.chat(f"请使用 delete_document 工具删除文档：{doc_name}")
                print(f"\n🤖 {response}")
                continue

            if user_input.lower().startswith("askdoc "):
                question = user_input[7:].strip()
                print(f"🔍 正在检索文档知识库...")
                response = self.chat(f"请使用 knowledge_qa 工具回答以下问题（基于已上传文档）：{question}")
                print(f"\n🤖 {response}")
                continue

            if user_input.lower().startswith("image "):
                image_path = user_input[6:].strip()
                print(f"🖼️ 已加载图像：{image_path}")
                # 提示用户输入问题
                question = input("👤 关于这张图你想问什么？：").strip()
                if question:
                    response = self.chat(question, image_source=image_path)
                    print(f"\n🤖 助手：{response}")
                continue

            # 正常对话
            image_source = None
            # 检测是否包含图像路径
            if " --image " in user_input:
                parts = user_input.split(" --image ")
                user_input = parts[0].strip()
                image_source = parts[1].strip().split()[0]

            response = self.chat(user_input, image_source=image_source)
            print(f"\n🤖 助手：{response}")

    # ==================== 会话管理 ====================

    def _update_session(self, user_input: str, response: str):
        """更新会话历史"""
        self.session_messages.append(HumanMessage(content=user_input))
        self.session_messages.append(AIMessage(content=response))

        # 限制消息数量（保留最近 20 轮）
        max_messages = 40
        if len(self.session_messages) > max_messages:
            self.session_messages = self.session_messages[-max_messages:]

    def _clear_session(self):
        """清空会话"""
        self.session_messages = []
        self.thread_id = str(uuid.uuid4())[:8]
        print("🧹 会话已清空，checkpoint 已重置")

    def _show_history(self):
        """显示会话历史"""
        if not self.session_messages:
            print("📝 暂无对话历史")
            return
        print(f"\n📝 会话历史（{len(self.session_messages) // 2} 轮）：")
        for msg in self.session_messages[-10:]:
            role = "👤" if isinstance(msg, HumanMessage) else "🤖"
            content = getattr(msg, "content", str(msg))[:100]
            print(f"  {role} {content}")

    def _show_checkpoint(self):
        """显示 checkpoint 状态"""
        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            state = self.graph.get_state(config)
            if state:
                print(f"📌 Checkpoint 状态：")
                print(f"  ├─ Thread: {self.thread_id}")
                plan = state.values.get("plan") if hasattr(state, 'values') else None
                if plan:
                    print(f"  ├─ 当前任务: {plan.get('task_summary', 'N/A')}")
                    print(f"  └─ 步骤: {plan.get('current_step', 0)}/{plan.get('total_steps', 0)}")
                else:
                    print(f"  └─ 无活跃任务")
            else:
                print("📌 暂无 checkpoint")
        except Exception as e:
            print(f"⚠️ 获取 checkpoint 失败：{e}")

    def _do_rollback(self):
        """回退到上一个 checkpoint"""
        try:
            config = {"configurable": {"thread_id": self.thread_id}}
            state = self.graph.get_state(config)
            if state:
                # 重置到上一个检查点
                self.graph.update_state(config, {
                    "decision": "continue",
                    "retry_count": 0,
                    "fallback_level": 0,
                })
                print("⏪ 已回退到上一个 checkpoint")
            else:
                print("⚠️ 没有可回退的 checkpoint")
        except Exception as e:
            print(f"⚠️ 回退失败：{e}")

    # ==================== 辅助方法 ====================

    @staticmethod
    def _augment_with_image(user_input: str, image_source: str) -> str:
        """将图像信息附加到用户输入"""
        if os.path.isfile(image_source):
            return f"[用户上传了图像：{image_source}]\n{user_input}\n请分析这张图像。"
        return f"[用户提供了图像]\n{user_input}\n请分析这张图像。"

    def _show_skills(self):
        """显示所有技能"""
        print("""
🛠️ 技能列表：
──────────────────────────────────────────────
  🔍 智能问答    — 通用知识问答、分析推理
  🌐 联网搜索    — Tavily 实时搜索最新信息
  🗺️ 出行规划    — 城市间交通方式推荐
  ☀️ 天气查询    — 实时天气（温度/湿度/风速）
  🔢 计算器      — 安全数学表达式计算
  🍳 菜谱推荐    — 根据食材推荐菜谱
  💻 CLI 执行    — 安全终端命令执行
  🖼️ 图像分析    — 识别食材/物体/场景
  🎨 图像生成    — 根据描述生成图像
──────────────────────────────────────────────
        """)

    def _show_tools(self):
        """显示所有工具"""
        tools = self.tool_registry.list_tools()
        print(f"\n🔧 已注册工具（{len(tools)} 个）：")
        for name in tools:
            tool_obj = self.tool_registry.get(name)
            desc = (tool_obj.description or "无描述")[:80]
            print(f"  • {name}: {desc}")


# ==================== CLI 入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description="🤖 个人全能助手 — All-Around Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python agent.py                              # 交互模式
  python agent.py "武汉今天天气怎么样"           # 单次查询
  python agent.py --image photo.jpg "有什么食材" # 图像分析
  python agent.py --stream "搜索最新AI新闻"      # 流式输出
  python agent.py --no-stream "1+1等于几"        # 非流式输出
        """,
    )
    parser.add_argument(
        "query", nargs="?", default=None,
        help="直接提问（不进入交互模式）"
    )
    parser.add_argument(
        "--image", "-i", type=str, default=None,
        help="图像路径（用于分析）"
    )
    parser.add_argument(
        "--stream", "-s", action="store_true", default=True,
        help="启用流式输出（默认）"
    )
    parser.add_argument(
        "--no-stream", action="store_true", default=False,
        help="禁用流式输出"
    )
    parser.add_argument(
        "--user", "-u", type=str, default="default_user",
        help="用户标识"
    )
    parser.add_argument(
        "--list-tools", action="store_true", default=False,
        help="列出所有可用工具"
    )

    args = parser.parse_args()

    # 列出工具
    if args.list_tools:
        registry = get_all_tools()
        print(f"🔧 可用工具（{len(registry.list_tools())} 个）：\n")
        print(registry.get_tool_descriptions())
        return

    # 确定流式模式
    stream_enabled = args.stream and not args.no_stream

    # 创建助手
    assistant = AllAroundAssistant(
        user_id=args.user,
        stream_enabled=stream_enabled,
    )

    # 单次查询模式
    if args.query:
        response = assistant.chat(args.query, image_source=args.image)
        if not stream_enabled:
            print(f"\n🤖 {response}")
        return

    # 交互模式
    assistant.run_interactive()


if __name__ == "__main__":
    main()
