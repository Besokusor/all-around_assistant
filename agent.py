import os
import json
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from tavily import TavilyClient
# 导入工具
from agent_tool import get_real_weather, calculator

load_dotenv()

# 模型初始化
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)
# 初始化Tavily客户端
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

tools = {
    "get_real_weather": get_real_weather,
    "calculator": calculator
}

# 工具指令
TOOL_PROMPT = """
你是智能助手，可调用工具：
1. get_real_weather(city: str) → 查询城市天气
2. calculator(expression: str) → 数学计算

需要调用工具时，**仅输出纯JSON**，无多余内容：
{"name":"工具名","parameters":{"参数名":"值"}}
已有工具返回结果时，直接整理成自然语言回答，不要再调用工具。
"""

# 状态定义
class AgentState(TypedDict):
    messages: list
    need_tool_call: dict | None
    tool_result: str | None

# AI 思考节点
def agent_node(state: AgentState):
    print("\n=====================================")
    print("📝 步骤1：AI 思考中...")
    print(f"步骤1：{state}")
    # 拼接完整上下文（用户问题 + 历史消息 + 工具结果）
    full_ctx = ""
    for msg in state["messages"]:
        full_ctx += f"{msg.content}\n"
    if state["tool_result"]:
        full_ctx += f"【工具返回结果】：{state['tool_result']}\n"

    full_ctx += TOOL_PROMPT
    ai_response = llm.invoke(full_ctx)
    ai_content = ai_response.content.strip()
    print(f"🤖 AI 原始输出：\n{ai_content}")

    # 解析工具调用
    need_tool_call = None
    try:
        if "{" in ai_content and "}" in ai_content:
            json_start = ai_content.find("{")
            json_end = ai_content.rfind("}") + 1
            tool_json = json.loads(ai_content[json_start:json_end])
            if "name" in tool_json and "parameters" in tool_json:
                need_tool_call = tool_json
    except Exception:
        print("⚠️  无需调用工具，准备输出最终答案")

    if need_tool_call:
        print(f"✅ 判定调用工具：{need_tool_call['name']}")
        print(f"📥 工具参数：{need_tool_call['parameters']}")
    else:
        print("✅ 无需调用工具，流程结束")

    # 追加当前AI回复到消息列表
    new_messages = state["messages"] + [AIMessage(content=ai_content)]
    return {
        "messages": new_messages,
        "need_tool_call": need_tool_call,
        "tool_result": None  # 清空工具结果，避免重复使用
    }

# 工具执行节点
def tool_exec_node(state: AgentState):
    print("\n=====================================")
    print("🔧 步骤2：执行工具调用...")
    print(f"步骤2：{state}")
    call = state["need_tool_call"]
    tool_name = call["name"]
    params = call["parameters"]

    tool_func = tools[tool_name]
    result = tool_func.invoke(params)
    result_str = str(result)

    print(f"✅ 工具【{tool_name}】执行完成")
    print(f"📤 工具返回结果：\n{result_str}")
    return {"tool_result": result_str}

# 路由判断
def router(state: AgentState):
    return "tool_exec" if state["need_tool_call"] else END

# 构建工作流
wf = StateGraph(AgentState)
wf.add_node("agent", agent_node)
wf.add_node("tool_exec", tool_exec_node)
wf.set_entry_point("agent")
wf.add_conditional_edges(
    "agent",       # 起点：AI思考节点
    router,        # 用这个函数判断
    {
        "tool_exec": "tool_exec",  # 返回 tool_exec → 去工具节点
        END: END                   # 返回 END → 结束流程
    }
)
wf.add_edge("tool_exec", "agent")
app = wf.compile()

# 主运行逻辑
if __name__ == "__main__":
    user_input = "武汉今天的天气怎么样"
    output = app.invoke({
        "messages": [HumanMessage(content=user_input)],
        "need_tool_call": None,
        "tool_result": None
    })

    print("\n=====================================")
    print("🎯 最终回答：", output["messages"][-1].content)