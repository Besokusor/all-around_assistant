"""
Graph 构建器 — 组装 LangGraph 工作流

流程:
  START → memory_retrieve → planner → executor ↔ observer
                                              ↓
                                    ┌─────────┴──────────┐
                                    ↓         ↓          ↓
                                continue    retry    revise_plan
                                    ↓         ↓          ↓
                                executor  executor   planner
                                    ↓
                                (observer→end)
                                    ↓
                            memory_update → response → END
                                    ↓
                                fallback → (retry/end/human)
"""
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import AgentState
from graph.nodes import (
    memory_retrieve_node,
    planner_node,
    executor_node,
    observer_node,
    memory_update_node,
    response_node,
    fallback_node,
)


# ==================== 路由函数 ====================

def route_after_planner(state: AgentState) -> str:
    """规划完成后 → 执行 or 兜底"""
    decision = state.get("decision", "continue")
    if decision == "fallback":
        return "fallback"
    return "executor"


def route_after_executor(state: AgentState) -> str:
    """执行完成后 → 观察"""
    return "observer"


def route_after_observer(state: AgentState) -> str:
    """
    观察评估后的路由决策

    continue     → executor（继续下一步）
    retry        → executor（重试当前步骤）
    revise_plan  → planner（修订计划）
    fallback     → fallback（兜底处理）
    end          → memory_update（完成）
    """
    decision = state.get("decision", "end")

    routes = {
        "continue": "executor",
        "retry": "executor",
        "revise_plan": "planner",
        "fallback": "fallback",
        "end": "memory_update",
    }
    return routes.get(decision, "memory_update")


def route_after_fallback(state: AgentState) -> str:
    """兜底处理后的路由"""
    decision = state.get("decision", "end")
    if decision == "retry":
        return "executor"
    return END


# ==================== Graph 构建 ====================

class AssistantGraph:
    """
    个人全能助手 LangGraph 工作流

    特性：
    - 7 个节点覆盖五层架构
    - 条件路由实现灵活决策
    - MemorySaver 提供 checkpoint 支持
    - 支持流式输出
    """

    def __init__(self, checkpointer=None):
        self.checkpointer = checkpointer or MemorySaver()
        self.graph = self._build()

    def _build(self) -> StateGraph:
        """构建并编译 Graph"""
        wf = StateGraph(AgentState)

        # === 添加节点 ===
        wf.add_node("memory_retrieve", memory_retrieve_node)
        wf.add_node("planner", planner_node)
        wf.add_node("executor", executor_node)
        wf.add_node("observer", observer_node)
        wf.add_node("memory_update", memory_update_node)
        wf.add_node("response", response_node)
        wf.add_node("fallback", fallback_node)

        # === 设置入口 ===
        wf.set_entry_point("memory_retrieve")

        # === 添加边 ===
        wf.add_edge("memory_retrieve", "planner")

        wf.add_conditional_edges(
            "planner", route_after_planner,
            {"executor": "executor", "fallback": "fallback"},
        )

        wf.add_edge("executor", "observer")

        wf.add_conditional_edges(
            "observer", route_after_observer,
            {
                "executor": "executor",
                "planner": "planner",
                "fallback": "fallback",
                "memory_update": "memory_update",
            },
        )

        # 兜底 → 重试 or 结束
        wf.add_conditional_edges(
            "fallback",
            route_after_fallback,
            {"executor": "executor", END: END},
        )

        # 记忆更新 → 回复生成 → 结束
        wf.add_edge("memory_update", "response")
        wf.add_edge("response", END)

        # === 编译（带 checkpoint） ===
        return wf.compile(checkpointer=self.checkpointer)

    def invoke(self, state: AgentState, config: dict = None) -> AgentState:
        """
        同步执行（非流式）

        Args:
            state: 初始 AgentState
            config: LangGraph 配置（包含 thread_id 用于 checkpoint）

        Returns:
            最终 AgentState
        """
        config = config or {"configurable": {"thread_id": "default"}}
        return self.graph.invoke(state, config)

    def stream(self, state: AgentState, config: dict = None):
        """
        流式执行

        Yields:
            每个节点的输出事件
        """
        config = config or {"configurable": {"thread_id": "default"}}
        return self.graph.stream(state, config)

    async def astream(self, state: AgentState, config: dict = None, stream_mode=None):
        """
        异步流式执行

        Args:
            state: 初始 AgentState
            config: LangGraph 配置
            stream_mode: 流式模式，如 ["messages", "updates"]
                - "updates": 节点完成时推送状态更新
                - "messages": 拦截节点内 LLM token chunk

        Yields:
            (mode, data) 元组
        """
        config = config or {"configurable": {"thread_id": "default"}}
        if stream_mode:
            async for event in self.graph.astream(state, config, stream_mode=stream_mode):
                yield event
        else:
            async for event in self.graph.astream(state, config):
                yield event

    def get_state(self, config: dict = None) -> AgentState:
        """获取当前 checkpoint 状态（用于回退）"""
        config = config or {"configurable": {"thread_id": "default"}}
        return self.graph.get_state(config)

    def update_state(self, config: dict, values: dict):
        """手动更新状态（用于回退/修正）"""
        self.graph.update_state(config, values)


# ==================== 工厂函数 ====================

def build_graph(checkpointer=None) -> AssistantGraph:
    """
    构建 AssistantGraph 实例

    Args:
        checkpointer: 可选自定义 checkpointer（默认 MemorySaver）

    Returns:
        AssistantGraph 实例
    """
    return AssistantGraph(checkpointer=checkpointer)
