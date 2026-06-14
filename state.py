"""
AgentState — 个人全能助手核心状态定义

五层架构状态：
  - 记忆层: short_term_context, retrieved_long_term, user_profile
  - 工具层: pending_tool_calls, tool_results
  - 思考层: plan, plan_steps, current_step_index
  - 执行层: execution_log, retry_count
  - 观察层: observation, decision

兜底机制: requires_human, fallback_level, checkpoint_data
"""
from typing import TypedDict, Annotated, Sequence, Optional, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class PlanStep(TypedDict):
    """计划中的单个步骤"""
    step_id: int
    description: str           # 步骤描述
    tool_name: Optional[str]   # 需要调用的工具名（None 表示纯推理）
    tool_params: Optional[dict]  # 工具参数
    expected_output: str       # 预期产出
    status: str                # pending | running | completed | failed | skipped


class ToolCallRecord(TypedDict):
    """工具调用记录"""
    tool_name: str
    params: dict
    result: Optional[str]
    error: Optional[str]
    duration_ms: float
    timestamp: str


class Plan(TypedDict):
    """执行计划"""
    task_summary: str           # 任务摘要
    steps: list[PlanStep]       # 步骤列表
    current_step: int           # 当前步骤索引
    total_steps: int
    revision_count: int         # 修订次数
    created_at: str


class UserProfile(TypedDict):
    """用户画像"""
    user_id: str
    name: Optional[str]
    preferences: dict           # 偏好设置 {food, travel, language, ...}
    habits: dict                # 行为习惯
    frequently_used: list[str]  # 常用功能
    dietary_restrictions: list[str]  # 饮食限制
    travel_preferences: dict    # 出行偏好
    personal_info: dict         # 个人信息（脱敏）
    updated_at: str


class AgentState(TypedDict):
    """
    全能助手 Agent 完整状态

    === 会话信息 ===
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_input: str                     # 当前用户输入
    user_id: str                        # 用户标识

    # ==================== 记忆层 ====================
    short_term_context: dict            # 短期记忆（当前会话上下文）
    retrieved_long_term: list[dict]     # 长期记忆检索结果
    user_profile: Optional[UserProfile] # 用户画像
    memory_needs_update: bool           # 是否需要更新记忆

    # ==================== 思考层（规划） ====================
    plan: Optional[Plan]                # 当前执行计划
    plan_revision_count: int            # 计划修订次数

    # ==================== 工具层 & 执行层 ====================
    pending_tool_calls: list[dict]      # 待执行的工具调用
    tool_results: list[ToolCallRecord]  # 工具执行结果记录
    execution_log: list[str]            # 执行日志
    retry_count: int                    # 当前步骤重试次数
    max_retries: int                    # 最大重试次数（从 config 注入）

    # ==================== 观察层 ====================
    observation: str                    # 当前观察结果
    decision: str                       # continue | retry | revise_plan | fallback | end
    observation_history: list[str]      # 观察历史

    # ==================== 输出 ====================
    final_response: Optional[str]       # 最终回复
    stream_enabled: bool                # 是否启用流式输出
    error_message: Optional[str]        # 错误信息

    # ==================== 兜底 & 降级 ====================
    requires_human: bool                # 是否需要人工介入
    fallback_level: int                 # 降级级别 0=正常 1=降级 2=最小模式 3=人工
    checkpoint_data: Optional[dict]     # 检查点快照（用于回退）
    degradation_reason: Optional[str]   # 降级原因


# ==================== State 工具函数 ====================

def create_initial_state(
    user_input: str,
    user_id: str = "default_user",
    stream_enabled: bool = True,
    max_retries: int = 3,
) -> AgentState:
    """创建初始状态"""
    return AgentState(
        messages=[],
        user_input=user_input,
        user_id=user_id,
        # 记忆层
        short_term_context={},
        retrieved_long_term=[],
        user_profile=None,
        memory_needs_update=False,
        # 思考层
        plan=None,
        plan_revision_count=0,
        # 执行层
        pending_tool_calls=[],
        tool_results=[],
        execution_log=[],
        retry_count=0,
        max_retries=max_retries,
        # 观察层
        observation="",
        decision="continue",
        observation_history=[],
        # 输出
        final_response=None,
        stream_enabled=stream_enabled,
        error_message=None,
        # 兜底
        requires_human=False,
        fallback_level=0,
        checkpoint_data=None,
        degradation_reason=None,
    )


def create_checkpoint(state: AgentState) -> dict:
    """创建检查点快照（保存可回退的关键状态）"""
    return {
        "plan": state.get("plan"),
        "current_step_index": state["plan"]["current_step"] if state.get("plan") else 0,
        "retry_count": state["retry_count"],
        "tool_results": list(state["tool_results"]),
        "execution_log": list(state["execution_log"]),
        "fallback_level": state["fallback_level"],
    }


def restore_from_checkpoint(state: AgentState, checkpoint: dict) -> AgentState:
    """从检查点恢复状态"""
    if checkpoint.get("plan"):
        state["plan"] = checkpoint["plan"]
    state["retry_count"] = checkpoint.get("retry_count", 0)
    state["tool_results"] = checkpoint.get("tool_results", [])
    state["execution_log"] = checkpoint.get("execution_log", [])
    state["execution_log"].append("[CHECKPOINT] 已回退到上一个检查点")
    state["fallback_level"] = checkpoint.get("fallback_level", 0) + 1
    return state
