"""
LangGraph 节点 — 五层架构的节点实现

节点流程：
  1. memory_retrieve_node  → 记忆检索
  2. planner_node          → 思考规划（with_structured_output 严格 JSON）
  3. executor_node         → 执行工具调用（bind_tools 严格工具选择）
  4. observer_node         → 观察评估
  5. memory_update_node    → 记忆更新
  6. response_node         → 生成回复（流式）
  7. fallback_node         → 兜底/降级处理

关键改进：
  - planner_node 使用 llm.with_structured_output(PlanOutput) 替代 prompt JSON
  - executor_node 使用 llm.bind_tools() 替代 prompt 约束工具选择
"""
import json
import time
import traceback
from typing import Optional

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from state import (
    AgentState, Plan, PlanStep, ToolCallRecord,
    create_checkpoint, restore_from_checkpoint,
)
from memory.manager import MemoryManager
from tools.registry import get_all_tools
from config import LLM_CONFIG, EXECUTION_CONFIG, FALLBACK_CONFIG
from logger import get_logger, set_trace_id


# ==================== Pydantic Schema：规划器结构化输出 ====================

class PlanStepSchema(BaseModel):
    """单个计划步骤的 Schema（LLM 结构化输出）"""
    step_id: int = Field(description="步骤序号，从1开始")
    description: str = Field(description="步骤描述，说明这一步骤要做什么")
    tool_name: Optional[str] = Field(
        default=None,
        description="需要调用的工具名。可用工具：web_search, calculator, get_weather, "
                    "plan_travel, recommend_recipe, execute_cli, document_upload, "
                    "list_documents, delete_document, knowledge_qa。无需工具时设为 null",
    )
    tool_params: dict = Field(
        default_factory=dict,
        description="工具参数，如 {'expression': '1+2'}。为每个参数提供明确的键值对。",
    )
    expected_output: str = Field(description="预期此步骤完成后的产出")


class PlanOutput(BaseModel):
    """规划器完整输出 Schema"""
    task_summary: str = Field(description="一句话任务摘要")
    steps: list[PlanStepSchema] = Field(description="执行步骤列表，按顺序排列")


# ==================== 日志工具函数 ====================

def _log():
    """获取 trace logger 实例"""
    return get_logger()


# ==================== 节点 1：记忆检索 ====================

def memory_retrieve_node(state: AgentState) -> dict:
    """
    记忆检索节点（记忆层入口）

    从三层记忆检索相关上下文：
    - 短期记忆（当前会话）
    - 长期记忆（Chroma 语义检索）
    - 用户画像（MySQL）
    """
    t0 = time.time()
    print("\n" + "=" * 50)
    print("🧠 [记忆层] 检索相关记忆...")

    try:
        # 记录原始输入
        logger = _log()
        if logger:
            logger.trace_input("memory", {
                "user_input": state.get("user_input", "")[:200],
                "user_id": state.get("user_id", ""),
                "messages_count": len(state.get("messages", [])),
                "fallback_level": state.get("fallback_level", 0),
            })

        manager = MemoryManager.get_instance()
        context = manager.retrieve_context(state)

        print(f"  ├─ 短期记忆：{len(context['short_term'])} 项上下文")
        print(f"  ├─ 长期记忆：检索到 {len(context['long_term'])} 条相关历史")
        print(f"  └─ 用户画像：{'已加载' if context['user_profile'] else '暂无'}")

        if logger:
            dur = (time.time() - t0) * 1000
            logger.memory(
                f"检索完成 ({dur:.0f}ms) │ 短期={len(context['short_term'])}项 "
                f"长期={len(context['long_term'])}条 画像={'有' if context['user_profile'] else '无'}"
            )
            # 记录原始输出
            long_term_preview = [
                {"content": m.get("content", "")[:150], "score": m.get("score")}
                for m in context.get("long_term", [])[:3]
            ]
            profile_preview = {}
            if context.get("user_profile"):
                up = context["user_profile"]
                profile_preview = {
                    "user_id": up.get("user_id"),
                    "preferences": up.get("preferences", {}),
                    "frequently_used": up.get("frequently_used", [])[:5],
                }
            logger.trace_output("memory", {
                "short_term_keys": list(context.get("short_term", {}).keys()),
                "long_term_count": len(context.get("long_term", [])),
                "long_term_top3": long_term_preview,
                "has_user_profile": bool(context.get("user_profile")),
                "user_profile": profile_preview,
            })

        return {
            "short_term_context": context["short_term"],
            "retrieved_long_term": context["long_term"],
            "user_profile": context["user_profile"],
            "execution_log": ["[记忆层] 完成记忆检索"],
        }
    except Exception as e:
        print(f"  ⚠️ 记忆检索降级：{e}")
        if logger:
            logger.warning(f"记忆检索降级：{e}", layer="memory")
        return {
            "short_term_context": {},
            "retrieved_long_term": [],
            "user_profile": None,
            "execution_log": [f"[记忆层] 检索降级：{e}"],
            "fallback_level": state.get("fallback_level", 0) + 1,
        }


# ==================== 节点 2：规划（with_structured_output 严格约束） ====================

PLANNER_SYSTEM_PROMPT = """你是一个任务规划专家。根据用户输入和上下文，将任务分解为可执行的步骤。

## 可用工具
{tool_descriptions}

## 规划原则
1. 分析用户真实意图，不要过度拆解简单任务
2. 每个步骤应明确：需要什么工具？参数是什么？预期产出是什么？
3. 信息检索类任务（搜索、天气）通常 1-2 步即可
4. 复杂任务（出行规划+天气+菜谱）可能需要 3-5 步
5. 优先使用工具获取实时数据，不要凭空生成
6. 不需要工具时，tool_name 设为 null，表示纯文本推理回答
"""


def planner_node(state: AgentState) -> dict:
    """
    规划节点（思考层）— 使用 with_structured_output 严格约束输出格式

    分析用户意图 → 拆解任务 → 生成执行计划（Pydantic 强制 JSON Schema）
    支持计划修订（observer 触发 revise_plan 时）
    """
    t0 = time.time()
    print("\n" + "=" * 50)
    print("💡 [思考层] 生成执行计划 (with_structured_output)...")

    user_input = state.get("user_input", "")
    revision_count = state.get("plan_revision_count", 0)
    max_revisions = EXECUTION_CONFIG["max_plan_revisions"]

    # 记录原始输入
    logger = _log()
    if logger:
        profile = state.get("user_profile")
        logger.trace_input("think", {
            "user_input": user_input[:300],
            "revision_count": revision_count,
            "has_profile": bool(profile),
            "has_long_term": bool(state.get("retrieved_long_term")),
            "messages_count": len(state.get("messages", [])),
        })

    if revision_count >= max_revisions:
        print(f"  ⚠️ 已达最大修订次数 ({max_revisions})，触发降级")
        logger = _log()
        if logger:
            logger.fallback(f"计划修订超限 ({revision_count}/{max_revisions})，触发降级")
        return {
            "decision": "fallback",
            "degradation_reason": f"计划修订超过 {max_revisions} 次",
            "requires_human": True,
            "execution_log": ["[思考层] 超过最大修订次数，降级处理"],
        }

    try:
        # ===== 使用 json_mode 在 API 层面强制合法 JSON 输出 =====
        # json_mode 原理：response_format={"type": "json_object"} → API 层约束 token 输出
        # DeepSeek 不支持 tool_choice（function_calling），但支持 json_mode
        # 注意：prompt 中必须包含 "json" 字样（OpenAI/DeepSeek 的硬性要求）
        from langchain_core.output_parsers import PydanticOutputParser

        parser = PydanticOutputParser(pydantic_object=PlanOutput)

        planner_llm = ChatOpenAI(
            model=LLM_CONFIG.get("think_model", "deepseek-v4-pro"),
            temperature=0.3,
            api_key=LLM_CONFIG["api_key"],
            base_url=LLM_CONFIG["base_url"],
            max_tokens=2048,
        )

        # json_mode: API 层强制输出 JSON（不依赖 tool_choice），Pydantic 校验 Schema
        structured_llm = planner_llm.with_structured_output(PlanOutput, method="json_mode")

        # 获取工具描述
        tool_registry = get_all_tools()
        tool_descriptions = tool_registry.get_tool_descriptions()

        # 构建规划上下文
        context_parts = [f"【用户输入】{user_input}"]

        # 对话历史（最近的对话，帮助理解指代和上下文）
        history_text = _build_history_from_messages(state.get("messages", []))
        if history_text:
            context_parts.append(history_text)

        profile = state.get("user_profile")
        if profile:
            prefs = profile.get("preferences", {})
            if prefs:
                context_parts.append(f"【用户偏好】{json.dumps(prefs, ensure_ascii=False)}")

        long_term = state.get("retrieved_long_term", [])
        if long_term:
            mem_text = "\n".join(f"  - {m['content'][:200]}" for m in long_term[:3])
            context_parts.append(f"【相关历史】\n{mem_text}")

        if revision_count > 0:
            observation = state.get("observation", "")
            context_parts.append(f"【上次执行反馈】{observation}")
            context_parts.append("请根据反馈重新修订计划。")

        full_context = "\n\n".join(context_parts)
        system_prompt = PLANNER_SYSTEM_PROMPT.format(tool_descriptions=tool_descriptions)

        # 注入格式指令 + 确保包含 "json" 字样（json_mode 要求 prompt 含 "json"）
        format_instructions = parser.get_format_instructions()

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"{full_context}\n\n"
                f"请按以下 JSON Schema 输出 json 格式的计划：\n"
                f"{format_instructions}\n"
                f"注意：只输出纯 JSON 对象，不要用 markdown 代码块包裹。"
            )),
        ]

        # json_mode 双重保障：
        #   1. API 层 response_format={"type":"json_object"} → 强制 token 级合法 JSON
        #   2. Pydantic 自动校验 Schema → 类型/字段/结构完全匹配
        # structured_llm.invoke() 直接返回 PlanOutput（无需额外 parser）
        try:
            plan_data: PlanOutput = structured_llm.invoke(messages)
        except Exception as parse_error:
            # API 保证 JSON 合法 → 这里只可能是 Schema 不匹配
            print(f"  ⚠️ Schema 校验失败 ({parse_error})，重试...")
            retry_msg = HumanMessage(content=(
                f"上次输出的 JSON Schema 不正确，请严格按格式重新输出 json。\n"
                f"校验错误：{str(parse_error)[:200]}"
            ))
            messages.append(retry_msg)
            plan_data: PlanOutput = structured_llm.invoke(messages)

        if plan_data and plan_data.steps:
            steps = []
            for s in plan_data.steps:
                steps.append(PlanStep(
                    step_id=s.step_id,
                    description=s.description,
                    tool_name=s.tool_name,
                    tool_params=s.tool_params or {},
                    expected_output=s.expected_output,
                    status="pending",
                ))

            plan = Plan(
                task_summary=plan_data.task_summary,
                steps=steps,
                current_step=0,
                total_steps=len(steps),
                revision_count=revision_count,
                created_at=str(time.time()),
            )

            print(f"  ├─ 任务：{plan['task_summary']}")
            print(f"  ├─ 步骤数：{plan['total_steps']}")
            for step in steps:
                tool_info = f"🔧 {step['tool_name']}" if step["tool_name"] else "💬 纯推理"
                print(f"  │   {step['step_id']}. [{tool_info}] {step['description'][:60]}")
            print(f"  └─ 修订次数：{revision_count}")

            if logger:
                dur = (time.time() - t0) * 1000
                step_list = ", ".join(
                    f"{s['step_id']}.{'🔧' if s.get('tool_name') else '💬'}{s['description'][:30]}"
                    for s in steps
                )
                logger.think(
                    f"计划生成 ({dur:.0f}ms) │ 任务=\"{plan['task_summary']}\" "
                    f"步骤={plan['total_steps']} 修订={revision_count} │ [{step_list}]"
                )
                # 记录原始输出：完整计划
                logger.trace_output("think", {
                    "task_summary": plan["task_summary"],
                    "total_steps": plan["total_steps"],
                    "revision_count": plan["revision_count"],
                    "steps": [
                        {
                            "step_id": s["step_id"],
                            "description": s["description"],
                            "tool_name": s.get("tool_name"),
                            "tool_params": s.get("tool_params", {}),
                            "expected_output": s.get("expected_output", ""),
                        }
                        for s in steps
                    ],
                })

            return {
                "plan": plan,
                "plan_revision_count": revision_count + 1,
                "decision": "continue",
                "execution_log": [
                    f"[思考层] 生成计划：{plan['task_summary']} ({plan['total_steps']}步)"
                ],
            }
        else:
            print("  ⚠️ 模型返回空计划，降级为直接回答")
            fallback_plan = Plan(
                task_summary=user_input,
                steps=[PlanStep(
                    step_id=1, description="直接回答",
                    tool_name=None, tool_params={},
                    expected_output="自然语言回答", status="pending",
                )],
                current_step=0, total_steps=1, revision_count=0,
                created_at=str(time.time()),
            )
            return {
                "plan": fallback_plan,
                "decision": "continue",
                "execution_log": ["[思考层] 模型返回空计划，使用默认计划"],
            }

    except Exception as e:
        print(f"  ❌ 规划失败：{e}")
        traceback.print_exc()
        logger = _log()
        if logger:
            logger.error(f"规划异常：{e}", layer="think")
        return {
            "decision": "fallback",
            "degradation_reason": f"规划异常：{str(e)}",
            "error_message": str(e),
            "execution_log": [f"[思考层] 规划异常：{e}"],
        }


# ==================== 节点 3：执行（ToolRegistry 直接调用 + 推理） ====================

# 注意：DeepSeek thinking mode 不支持 tool_choice/bind_tools，
# 因此工具调用通过 planner 指定 + ToolRegistry 直接执行（确定性最强）
# 纯推理步骤使用直接 LLM 调用生成回答


def executor_node(state: AgentState) -> dict:
    """
    执行节点（执行层）— ToolRegistry 直接调用，避免 prompt 约束

    执行当前计划步骤：
    - plan 指定工具 → 通过 ToolRegistry 直接调用（确定性强，非 LLM 决策）
    - plan 未指定 → 纯推理，直接 LLM 生成回答
    - 记录执行日志
    """
    t0 = time.time()
    print("\n" + "=" * 50)
    plan = state.get("plan")
    if not plan:
        print("⚡ [执行层] 无计划，跳过")
        return {"decision": "end", "execution_log": ["[执行层] 无计划可执行"]}

    step_idx = plan["current_step"]
    steps = plan["steps"]

    if step_idx >= len(steps):
        print("⚡ [执行层] 所有步骤已完成")
        return {"decision": "end", "execution_log": ["[执行层] 所有步骤执行完毕"]}

    current_step = steps[step_idx]
    print(f"⚡ [执行层] 执行步骤 {step_idx + 1}/{len(steps)}：{current_step['description'][:80]}")

    logger = _log()
    if logger:
        logger.execute(
            f"步骤{step_idx+1}/{len(steps)} 开始 │ "
            f"{'🔧 '+current_step.get('tool_name','?') if current_step.get('tool_name') else '💬 纯推理'} "
            f"{current_step['description'][:60]}"
        )
        # 记录原始输入
        logger.trace_input("exec", {
            "step_index": step_idx,
            "total_steps": len(steps),
            "description": current_step.get("description", ""),
            "tool_name": current_step.get("tool_name"),
            "tool_params": current_step.get("tool_params", {}),
            "is_pure_reasoning": not bool(current_step.get("tool_name")),
        })

    try:
        tool_name = current_step.get("tool_name")
        tool_params = current_step.get("tool_params", {})

        if tool_name:
            # === 路径 A：plan 已明确指定工具 → 直接通过 ToolRegistry 调用 ===
            # 修复 planner 生成的参数（可能用占位符或错误字段名）
            if tool_name == "analyze_image":
                # 从 state 获取真实图片 URL
                img_url = state.get("short_term_context", {}).get("image_source", "")
                if img_url and ("image_source" not in tool_params or not tool_params["image_source"]):
                    tool_params["image_source"] = img_url
                # 兼容 planner 可能用的字段名
                if "image" in tool_params and "image_source" not in tool_params:
                    tool_params["image_source"] = tool_params.pop("image")
                if "task_type" in tool_params:
                    tool_params["task"] = tool_params.pop("task_type")
                # 去掉占位符
                for k, v in tool_params.items():
                    if isinstance(v, str) and "{{" in v:
                        tool_params[k] = img_url if "image" in k else "general"

            print(f"  🔧 调用工具（plan 指定）：{tool_name}")
            print(f"  📥 参数：{tool_params}")

            tool_registry = get_all_tools()
            result = tool_registry.execute(tool_name, tool_params)

            tool_record = ToolCallRecord(
                tool_name=tool_name,
                params=tool_params,
                result=result.get("result") if result["success"] else None,
                error=result.get("error") if not result["success"] else None,
                duration_ms=result.get("duration_ms", 0),
                timestamp=str(time.time()),
            )

            current_step["status"] = "completed" if result["success"] else "failed"

            print(f"  {'✅' if result['success'] else '❌'} 耗时：{result['duration_ms']}ms")
            if result["success"]:
                print(f"  📤 结果预览：{str(result['result'])[:200]}...")
            else:
                print(f"  ⚠️ 错误：{result['error']}")

            if logger:
                dur = (time.time() - t0) * 1000
                logger.tool_call(
                    tool_name, result.get("duration_ms", dur),
                    result["success"],
                    str(result.get("result", ""))[:100]
                )
                # 记录原始输出：完整工具返回
                logger.trace_output("exec", {
                    "step_index": step_idx,
                    "tool_name": tool_name,
                    "tool_params": tool_params,
                    "success": result["success"],
                    "result": str(result.get("result", ""))[:2000],
                    "error": result.get("error"),
                    "duration_ms": result.get("duration_ms", 0),
                })

            return {
                "plan": plan,
                "tool_results": state.get("tool_results", []) + [tool_record],
                "observation": (
                    result.get("result")
                    if result["success"]
                    else f"工具执行失败：{result.get('error')}"
                ),
                "retry_count": 0 if result["success"] else state.get("retry_count", 0) + 1,
                "execution_log": [
                    f"[执行层] 步骤{step_idx+1}: {tool_name} → "
                    f"{'成功' if result['success'] else '失败'} ({result['duration_ms']}ms)"
                ],
            }
        else:
            # === 路径 B：纯推理步骤 → 直接 LLM 推理（plan 已确认无需工具） ===
            print(f"  💬 纯推理步骤，直接 LLM 生成回答")

            llm = ChatOpenAI(
                model=LLM_CONFIG["model"],
                temperature=0.5,
                api_key=LLM_CONFIG["api_key"],
                base_url=LLM_CONFIG["base_url"],
                max_tokens=1024,
            )

            tool_results = state.get("tool_results", [])
            results_text = "\n".join(
                f"[{r['tool_name']}]: {r.get('result', r.get('error', ''))[:500]}"
                for r in tool_results[-3:]
            )

            # 对话历史（帮助理解追问和指代）
            history_text = _build_history_from_messages(state.get("messages", []))

            msg = HumanMessage(content=(
                f"用户问题：{state.get('user_input', '')}\n\n"
                f"{history_text}"
                f"当前步骤：{current_step['description']}\n"
                f"已获取的信息：\n{results_text}\n\n"
                f"请根据已有信息完成此步骤。如果信息不足，请说明需要什么。"
            ))

            response = llm.invoke([msg])

            current_step["status"] = "completed"

            if logger:
                dur = (time.time() - t0) * 1000
                logger.execute(
                    f"步骤{step_idx+1}: 推理完成 ({dur:.0f}ms) │ "
                    f"输出={len(response.content)}字符"
                )
                logger.trace_output("exec", {
                    "step_index": step_idx,
                    "type": "pure_reasoning",
                    "output": response.content[:2000],
                    "output_length": len(response.content),
                    "duration_ms": round(dur, 1),
                })

            return {
                "plan": plan,
                "observation": response.content,
                "execution_log": [f"[执行层] 步骤{step_idx+1}: 推理完成（无工具调用）"],
            }

    except Exception as e:
        print(f"  ❌ 执行异常：{e}")
        traceback.print_exc()
        logger = _log()
        if logger:
            logger.error(f"步骤{step_idx+1} 执行异常：{e}", layer="exec")
        current_step["status"] = "failed"
        return {
            "plan": plan,
            "observation": f"执行异常：{str(e)}",
            "retry_count": state.get("retry_count", 0) + 1,
            "error_message": str(e),
            "execution_log": [f"[执行层] 步骤{step_idx+1}: 异常 - {e}"],
        }


# ==================== 节点 4：观察 ====================

def observer_node(state: AgentState) -> dict:
    """
    观察节点（观察层）

    评估执行结果，决定下一步：
    - continue:    当前步骤成功，继续下一步
    - retry:       当前步骤失败，重试
    - revise_plan: 需要修订计划
    - fallback:    触发降级/兜底
    - end:         所有步骤完成，生成回复
    """
    t0 = time.time()
    print("\n" + "=" * 50)
    print("👁️ [观察层] 评估执行结果...")

    plan = state.get("plan")
    if not plan:
        return {"decision": "end"}

    step_idx = plan["current_step"]
    steps = plan["steps"]
    current_step = steps[step_idx] if step_idx < len(steps) else None

    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", EXECUTION_CONFIG["max_retries"])
    observation = state.get("observation", "")

    logger = _log()

    # 记录原始输入
    if logger:
        logger.trace_input("observe", {
            "step_index": step_idx,
            "total_steps": plan["total_steps"],
            "current_step_status": current_step.get("status") if current_step else None,
            "current_step_tool": current_step.get("tool_name") if current_step else None,
            "retry_count": retry_count,
            "max_retries": max_retries,
            "observation_preview": observation[:300] if observation else "",
        })

    # 情况1：重试次数超限
    if retry_count >= max_retries:
        print(f"  ⚠️ 步骤 {step_idx + 1} 重试 {retry_count} 次已达上限 → 跳过")
        if logger:
            logger.observe(f"步骤{step_idx+1} 重试超限 ({retry_count}/{max_retries}) → 跳过")
        if current_step:
            current_step["status"] = "skipped"
        plan["current_step"] = step_idx + 1

        # 所有步骤都完成/跳过了？
        if plan["current_step"] >= plan["total_steps"]:
            return {
                "plan": plan,
                "decision": "end",
                "retry_count": 0,
                "observation_history": state.get("observation_history", []) + [
                    f"步骤{step_idx+1}超过最大重试次数，已跳过"
                ],
                "execution_log": [f"[观察层] 步骤{step_idx+1} 重试超限，跳过"],
            }
        return {
            "plan": plan,
            "decision": "continue",
            "retry_count": 0,
            "execution_log": [f"[观察层] 步骤{step_idx+1} 重试超限，继续下一步"],
        }

    # 情况2：工具输出校验 — 结果为空/过短/明显异常 → 标记失败
    if current_step and current_step.get("status") == "completed" and current_step.get("tool_name"):
        obs = observation or ""
        # Harness 校验规则
        is_empty = len(obs.strip()) < 5
        is_error = any(kw in obs for kw in ["❌", "失败", "error", "Error", "异常"])
        if is_empty or is_error:
            print(f"  ⚠️ 工具输出校验不通过: empty={is_empty} error={is_error} → 标记失败")
            current_step["status"] = "failed"

    # 情况3：当前步骤失败 → 重试
    # （注意：情况2 在上面 — 工具输出校验不通过时已标记为 failed）
    if current_step and current_step.get("status") == "failed":
        print(f"  🔄 步骤 {step_idx + 1} 执行失败，准备重试 ({retry_count + 1}/{max_retries})")
        if logger:
            logger.observe(f"步骤{step_idx+1} 失败 → 重试 ({retry_count+1}/{max_retries})")
            logger.trace_output("observe", {
                "decision": "retry",
                "step_index": step_idx,
                "retry_count": retry_count + 1,
                "max_retries": max_retries,
                "reason": f"步骤{step_idx+1}执行失败",
            })

        checkpoint = create_checkpoint(state)

        return {
            "decision": "retry",
            "checkpoint_data": checkpoint,
            "execution_log": [f"[观察层] 步骤{step_idx+1} 失败，重试 {retry_count + 1}/{max_retries}"],
        }

    # 情况4：当前步骤成功 → 继续下一步
    if current_step and current_step.get("status") == "completed":
        plan["current_step"] = step_idx + 1
        print(f"  ✅ 步骤 {step_idx + 1} 完成 → 步骤 {plan['current_step'] + 1 if plan['current_step'] < plan['total_steps'] else '完成'}")

        if plan["current_step"] >= plan["total_steps"]:
            print(f"  🎯 所有步骤完成！")
            if logger:
                dur = (time.time() - t0) * 1000
                logger.observe(f"全部{plan['total_steps']}步骤完成 ({dur:.0f}ms) → 进入输出层")
                logger.trace_output("observe", {
                    "decision": "end",
                    "reason": f"全部{plan['total_steps']}步骤完成",
                    "duration_ms": round(dur, 1),
                })
            return {
                "plan": plan,
                "decision": "end",
                "observation_history": state.get("observation_history", []) + [
                    f"步骤{step_idx+1}完成"
                ],
                "execution_log": [f"[观察层] 全部 {plan['total_steps']} 个步骤完成"],
            }
        if logger:
            logger.observe(f"步骤{step_idx+1} 完成 → 继续步骤{plan['current_step']+1}")
            logger.trace_output("observe", {
                "decision": "continue",
                "next_step": plan["current_step"] + 1,
                "total_steps": plan["total_steps"],
            })
        return {
            "plan": plan,
            "decision": "continue",
            "retry_count": 0,
            "execution_log": [f"[观察层] 步骤{step_idx+1} 完成，继续"],
        }

    # 情况5：需要修订计划
    if state.get("decision") == "revise_plan":
        print(f"  📝 需要修订计划")
        if logger:
            logger.observe("触发计划修订")
        return {
            "decision": "revise_plan",
            "plan_revision_count": state.get("plan_revision_count", 0) + 1,
        }

    # 默认：继续
    return {"decision": "continue"}


# ==================== 节点 5：记忆更新 ====================

def memory_update_node(state: AgentState) -> dict:
    """
    记忆更新节点

    对话回合结束后更新所有记忆层：
    - 更新短期记忆上下文
    - 必要时压缩为长期记忆
    - 更新用户画像
    """
    t0 = time.time()
    print("\n" + "=" * 50)
    print("💾 [记忆层] 更新记忆...")

    try:
        # 记录原始输入
        logger = _log()
        if logger:
            logger.trace_input("memory", {
                "turn_count_before": state.get("short_term_context", {}).get("turn_count", 0),
                "needs_update": state.get("memory_needs_update", False),
                "plan_completed": bool(state.get("plan")),
            })

        manager = MemoryManager.get_instance()
        updated_state = manager.update_after_turn(state)

        turn_count = updated_state.get('short_term_context', {}).get('turn_count', 0)
        print(f"  ├─ 短期记忆轮次：{turn_count}")
        print(f"  └─ 记忆更新完成")

        if logger:
            dur = (time.time() - t0) * 1000
            logger.memory(f"记忆更新完成 ({dur:.0f}ms) │ 轮次={turn_count}")
            logger.trace_output("memory", {
                "turn_count_after": turn_count,
                "updated_context_keys": list(updated_state.get("short_term_context", {}).keys()),
            })

        return {
            "short_term_context": updated_state.get("short_term_context", {}),
            "memory_needs_update": False,
            "execution_log": ["[记忆层] 记忆更新完成"],
        }
    except Exception as e:
        print(f"  ⚠️ 记忆更新失败：{e}")
        if logger:
            logger.warning(f"记忆更新失败：{e}", layer="memory")
        return {
            "memory_needs_update": False,
            "execution_log": [f"[记忆层] 记忆更新失败：{e}"],
        }


# ==================== 节点 6：生成回复 ====================
# 注意：节点内 LLM 启用 streaming=True，配合 graph.astream(stream_mode="messages")
# LangGraph 会捕获 LLM 的每个 token chunk 并通过事件流推送，实现真正的 token 级流式输出。

RESPONSE_SYSTEM_PROMPT = """你是一个贴心、专业的个人全能助手。根据执行结果生成自然、有帮助的回复。

## 回复原则
1. 直接给出答案，不要啰嗦的 preamble（"根据搜索结果..."等）
2. 格式清晰，善用 emoji 和分段
3. 如果有不确定性，诚实说明
4. 提供可操作的下一步建议
5. 结合用户画像进行个性化回复
6. 用中文回复
"""


def response_node(state: AgentState) -> dict:
    """
    回复生成节点（输出层）

    整合所有执行结果，生成最终的自然语言回复。
    LLM 使用 streaming=True，配合 graph.astream(stream_mode="messages")
    实现真正的 token 级流式输出。
    """
    print("\n" + "=" * 50)
    print("💬 [输出层] 生成最终回复...")

    user_input = state.get("user_input", "")
    plan = state.get("plan", {})
    tool_results = state.get("tool_results", [])
    profile = state.get("user_profile")

    # 记录原始输入
    logger = _log()
    if logger:
        logger.trace_input("output", {
            "user_input": user_input[:300],
            "task_summary": plan.get("task_summary", ""),
            "total_steps_completed": plan.get("total_steps", 0),
            "tool_results_count": len(tool_results or []),
            "tools_called": [r.get("tool_name") for r in (tool_results or [])],
            "has_profile": bool(profile),
        })

    try:
        llm = ChatOpenAI(
            model=LLM_CONFIG["model"],
            temperature=0.7,
            api_key=LLM_CONFIG["api_key"],
            base_url=LLM_CONFIG["base_url"],
            max_tokens=2048,
            streaming=True,  # 启用流式，使 astream(stream_mode="messages") 可捕获 token
        )

        # 构建工具结果上下文
        results_text = "\n".join(
            f"[{r.get('tool_name', '?')}]: {str(r.get('result') or r.get('error') or '')[:600]}"
            for r in (tool_results or [])[-5:]
        ) if tool_results else "（无工具调用）"

        # 对话历史上下文（state["messages"] 在 graph 调用前已包含历史消息）
        history_text = _build_history_from_messages(state.get("messages", []))

        # 用户画像上下文
        profile_hint = ""
        if profile:
            prefs = profile.get("preferences", {})
            if prefs:
                parts = []
                for cat, items in prefs.items():
                    item_strs = []
                    for k, v in items.items():
                        if isinstance(v, dict) and v.get("confidence", 0) >= 0.6:
                            item_strs.append(f"{k}: {v.get('value', v)}")
                    if item_strs:
                        parts.append(f"  {cat}: {', '.join(item_strs)}")
                if parts:
                    profile_hint = "\n用户偏好：\n" + "\n".join(parts) + "\n"

        response = llm.invoke([
            SystemMessage(content=RESPONSE_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"用户问题：{user_input}\n"
                f"任务：{plan.get('task_summary', '回答用户问题')}\n\n"
                f"{history_text}"
                f"执行结果：\n{results_text}"
                f"{profile_hint}\n"
                f"请生成回复。"
            )),
        ])

        final_response = response.content.strip()
        print(f"  ✅ 回复生成完成（{len(final_response)} 字符）")

        if logger:
            tools_count = len(state.get("tool_results", []))
            logger.output(f"回复生成完成 │ 长度={len(final_response)}字符 工具调用={tools_count}次")
            logger.trace_output("output", {
                "final_response": final_response,
                "response_length": len(final_response),
                "tools_called": tools_count,
            })

        return {
            "final_response": final_response,
            "messages": [response],
            "execution_log": [f"[输出层] 回复生成完成 ({len(final_response)} 字符)"],
        }

    except Exception as e:
        print(f"  ❌ 回复生成失败：{e}")
        if logger:
            logger.error(f"回复生成失败：{e}", layer="output")
        fallback_response = _build_fallback_response(state)
        return {
            "final_response": fallback_response,
            "messages": [AIMessage(content=fallback_response)],
            "error_message": str(e),
            "execution_log": [f"[输出层] 回复生成降级：{e}"],
        }


# ==================== 节点 7：兜底处理 ====================

def fallback_node(state: AgentState) -> dict:
    """
    兜底/降级处理节点

    多层降级策略：
    Level 1: 重试当前步骤
    Level 2: 降级工具集（仅核心工具）
    Level 3: 使用缓存/历史回答
    Level 4: 人工介入
    """
    t0 = time.time()
    print("\n" + "=" * 50)
    level = state.get("fallback_level", 1)
    reason = state.get("degradation_reason", "未知原因")

    print(f"🆘 [兜底] 降级级别 {level}：{reason}")

    logger = _log()
    if logger:
        logger.fallback(f"Level {level} | 原因：{reason}")
        logger.trace_input("fallback", {
            "level": level,
            "reason": reason,
            "retry_count": state.get("retry_count", 0),
            "has_checkpoint": bool(state.get("checkpoint_data")),
            "plan": {
                "task_summary": state.get("plan", {}).get("task_summary"),
                "current_step": state.get("plan", {}).get("current_step"),
                "total_steps": state.get("plan", {}).get("total_steps"),
            } if state.get("plan") else None,
        })

    if level <= 1:
        # Level 1: 从检查点恢复并重试
        checkpoint = state.get("checkpoint_data")
        if checkpoint:
            print("  ├─ 从检查点恢复状态")
            restore_from_checkpoint(state, checkpoint)
        print("  └─ 策略：使用简化计划重试")
        if logger:
            logger.trace_output("fallback", {
                "level": 1,
                "decision": "retry",
                "strategy": "从检查点恢复重试",
            })
        return {
            "decision": "retry",
            "fallback_level": 1,
            "execution_log": [f"[兜底] Level 1 - 从检查点恢复重试"],
        }

    elif level == 2:
        # Level 2: 缩减到核心工具
        print("  └─ 策略：降级为核心工具集")
        if logger:
            logger.fallback("Level 2 | 降级为核心工具集")
            logger.trace_output("fallback", {
                "level": 2,
                "decision": "retry",
                "strategy": "降级为核心工具集",
            })
        return {
            "fallback_level": 2,
            "decision": "retry",
            "execution_log": [f"[兜底] Level 2 - 降级为核心工具"],
        }

    elif level >= 3:
        # Level 3+: 人工介入
        print("  └─ 策略：请求人工介入")
        if logger:
            logger.fallback(f"Level {level} | 人工介入：{reason}")
            logger.trace_output("fallback", {
                "level": level,
                "decision": "end",
                "strategy": "人工介入",
                "reason": reason,
            })
        return {
            "requires_human": True,
            "decision": "end",
            "final_response": (
                f"😔 抱歉，当前任务处理遇到困难：{reason}\n\n"
                f"已尝试 {level} 级降级策略，仍无法完成。\n"
                f"建议：\n"
                f"1. 换一种方式描述你的需求\n"
                f"2. 将复杂任务拆分为多个简单任务\n"
                f"3. 联系人工客服获取帮助"
            ),
            "execution_log": [f"[兜底] Level {level} - 人工介入"],
        }

    return {"decision": "end"}


# ==================== 工具函数 ====================

def _build_history_from_messages(messages: list) -> str:
    """从 state["messages"] 中提取最近对话历史（排除当前用户消息）"""
    if not messages or len(messages) < 2:
        return ""
    # 排除最后一条（当前用户消息），取最近 3 轮（6 条）
    history_msgs = messages[-7:-1] if len(messages) > 1 else []
    if not history_msgs:
        return ""
    parts = []
    for msg in history_msgs[-6:]:
        role = "用户" if isinstance(msg, HumanMessage) else "助手"
        content = getattr(msg, "content", str(msg))[:300]
        parts.append(f"{role}: {content}")
    return "对话历史：\n" + "\n".join(parts) + "\n\n"


def _build_fallback_response(state: AgentState) -> str:
    """构建降级回复（直接拼接工具结果）"""
    parts = []
    tool_results = state.get("tool_results", [])

    if tool_results:
        parts.append("以下是根据已获取信息的整理：\n")
        for r in tool_results:
            tool_name = r.get("tool_name", "未知工具")
            result = r.get("result", r.get("error", "无结果"))
            parts.append(f"🔹 {tool_name}：\n{result[:500]}\n")
    else:
        user_input = state.get("user_input", "")
        parts.append(f"收到你的问题：「{user_input}」\n")
        parts.append("当前无法完成完整的处理流程。请稍后重试或换一种方式提问。")

    return "\n".join(parts)
