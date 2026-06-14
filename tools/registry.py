"""
工具注册中心 — 个人全能助手所有工具

包含工具：
  1. web_search      — 联网搜索（Tavily）
  2. calculator      — 数学计算
  3. get_weather     — 天气查询
  4. plan_travel     — 出行规划
  5. recommend_recipe— 食材推荐菜谱
  6. execute_cli     — CLI 命令执行
  7. analyze_image   — 图像分析（见 image_tools.py）
  8. generate_image  — 图像生成（见 image_tools.py）

支持：
  - 工具注册/发现
  - 超时控制
  - 错误捕获和降级
  - 执行统计
"""
import time
import subprocess
import json
import re
from typing import Optional
from langchain_core.tools import tool

from config import TAVILY_CONFIG, CITY_COORDINATES, EXECUTION_CONFIG, LLM_CONFIG


# ==================== 1. 联网搜索 ====================

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    联网搜索，获取最新信息。
    适用场景：新闻、实时数据、百科知识、任何需要联网查询的问题。

    Args:
        query: 搜索关键词
        max_results: 返回结果数量（默认5）
    """
    try:
        from tavily import TavilyClient
        import os
        from dotenv import load_dotenv
        load_dotenv()

        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(
            query=query,
            search_depth=TAVILY_CONFIG.get("search_depth", "advanced"),
            max_results=max_results,
            include_answer=TAVILY_CONFIG.get("include_answer", True),
        )

        # 格式化结果
        parts = []
        if response.get("answer"):
            parts.append(f"📝 摘要：{response['answer']}")

        for i, result in enumerate(response.get("results", [])[:max_results], 1):
            title = result.get("title", "无标题")
            content = result.get("content", "")[:300]
            url = result.get("url", "")
            parts.append(f"{i}. {title}\n   {content}\n   🔗 {url}")

        return "\n\n".join(parts) if parts else "未找到相关结果"

    except ImportError:
        return "❌ Tavily 客户端未安装。请安装: pip install tavily-python"
    except Exception as e:
        return f"❌ 搜索失败：{str(e)}"


# ==================== 2. 数学计算 ====================

@tool
def calculator(expression: str) -> str:
    """
    安全数学计算器。支持 +、-、*、/、**（幂）、%（取余）、() 括号。
    示例：3 + 5 * 10, (100 - 20) / 4, 2 ** 10

    Args:
        expression: 数学表达式字符串
    """
    # 安全沙箱：只允许数学运算
    allowed = set("0123456789+-*/.()% e")
    sanitized = "".join(c for c in expression if c in allowed)

    if sanitized != expression.strip():
        return f"⚠️ 表达式包含不允许的字符，已清理为：{sanitized}"

    try:
        # 使用受限的 eval（仅允许数学运算）
        safe_dict = {
            "__builtins__": None,
            "abs": abs, "round": round, "min": min, "max": max,
            "pow": pow, "int": int, "float": float,
        }
        result = eval(sanitized, safe_dict, {})
        # 格式化输出
        if isinstance(result, float):
            result = round(result, 10)  # 避免浮点精度问题
            if result == int(result):
                result = int(result)
        return f"📊 计算结果：{expression.strip()} = {result}"
    except ZeroDivisionError:
        return "❌ 错误：不能除以零"
    except Exception as e:
        return f"❌ 计算失败：{str(e)}"


# ==================== 3. 天气查询 ====================

@tool
def get_weather(city: str) -> str:
    """
    查询指定城市的实时天气（温度、湿度、风速、天气状况）。
    支持中国主要城市。

    Args:
        city: 城市名称，如"北京"、"上海"、"武汉"
    """
    import requests

    if city not in CITY_COORDINATES:
        supported = "、".join(list(CITY_COORDINATES.keys()))
        return f"🌍 暂不支持「{city}」的天气查询。当前支持的城市：{supported}"

    lat, lon = CITY_COORDINATES[city]
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
    )

    try:
        res = requests.get(url, timeout=8).json()
        curr = res["current"]
        temp = curr["temperature_2m"]
        hum = curr["relative_humidity_2m"]
        wind = curr["wind_speed_10m"]
        code = curr["weather_code"]

        weather_map = {
            0: "☀️ 晴朗", 1: "🌤 晴", 2: "⛅ 多云", 3: "☁️ 阴天",
            45: "🌫 雾", 48: "🌫 雾凇",
            51: "🌦 小雨", 53: "🌧 中雨", 55: "⛈ 大雨",
            61: "🌦 小雨", 63: "🌧 中雨", 65: "⛈ 大雨",
            71: "🌨 小雪", 73: "🌨 中雪", 75: "❄️ 大雪",
            80: "🌦 阵雨", 95: "⛈ 雷暴",
        }
        condition = weather_map.get(code, f"未知天气(code:{code})")

        return (
            f"🏙️ {city} 实时天气：\n"
            f"  天气：{condition}\n"
            f"  温度：{temp}℃\n"
            f"  湿度：{hum}%\n"
            f"  风速：{wind} m/s"
        )
    except requests.exceptions.Timeout:
        return "⏰ 天气查询超时，请稍后再试"
    except Exception as e:
        return f"❌ 天气查询失败：{str(e)}"


# ==================== 辅助函数：联网搜索 ====================

def _web_search_fallback(query: str, max_results: int = 5) -> str:
    """
    内部联网搜索兜底函数。
    供 plan_travel / recommend_recipe 在本地数据未命中时动态查询。
    """
    try:
        from tavily import TavilyClient
        import os as _os
        from dotenv import load_dotenv
        load_dotenv()

        client = TavilyClient(api_key=_os.getenv("TAVILY_API_KEY"))
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            include_answer=True,
        )

        parts = []
        if response.get("answer"):
            parts.append(response["answer"])

        for result in response.get("results", [])[:max_results]:
            parts.append(
                f"  • {result.get('title', '')}\n"
                f"    {result.get('content', '')[:400]}\n"
                f"    🔗 {result.get('url', '')}"
            )

        return "\n\n".join(parts) if parts else "未找到相关信息"

    except ImportError:
        return "⚠️ Tavily 未安装，无法联网搜索"
    except Exception as e:
        return f"⚠️ 联网搜索失败：{str(e)}"


# ==================== 4. 出行规划 ====================

@tool
def plan_travel(origin: str, destination: str, date: str = "", preference: str = "") -> str:
    """
    出行规划工具。根据出发地、目的地、日期和偏好，提供出行建议。
    包括交通方式推荐、预计时间、注意事项等。
    优先使用本地缓存数据，未命中时自动联网搜索。

    Args:
        origin: 出发城市
        destination: 目的城市
        date: 出行日期（可选，格式 YYYY-MM-DD）
        preference: 出行偏好，如"最快"、"最省钱"、"高铁"、"自驾"等
    """
    # 本地缓存：常见城市对（秒级响应，离线可用）
    travel_cache = {
        ("北京", "上海"): {"高铁": "4.5h", "飞机": "2h", "自驾": "12h", "距离": "1200km"},
        ("北京", "武汉"): {"高铁": "4h", "飞机": "2h", "自驾": "11h", "距离": "1100km"},
        ("上海", "武汉"): {"高铁": "3.5h", "飞机": "1.5h", "自驾": "8h", "距离": "800km"},
        ("武汉", "广州"): {"高铁": "3.5h", "飞机": "1.5h", "自驾": "9h", "距离": "900km"},
        ("武汉", "深圳"): {"高铁": "4h", "飞机": "1.5h", "自驾": "10h", "距离": "1000km"},
        ("武汉", "成都"): {"高铁": "8h", "飞机": "2h", "自驾": "14h", "距离": "1200km"},
        ("武汉", "杭州"): {"高铁": "4.5h", "飞机": "1.5h", "自驾": "7h", "距离": "700km"},
    }

    key = None
    for (a, b), data in travel_cache.items():
        if (origin in (a, b)) and (destination in (a, b)):
            key = (a, b)
            break

    parts = [f"🗺️ 出行规划：{origin} → {destination}"]
    if date:
        parts.append(f"📅 日期：{date}")

    if key:
        # ── 命中本地缓存 ──
        data = travel_cache[key]
        parts.append(f"⚡ 数据来源：本地缓存")
        parts.append(f"\n📏 距离：约 {data['距离']}")
        parts.append("\n🚀 交通方式对比：")

        for mode, duration in data.items():
            if mode == "距离":
                continue
            icon = {"高铁": "🚄", "飞机": "✈️", "自驾": "🚗"}.get(mode, "🔹")
            recommendation = ""
            if preference and preference in mode:
                recommendation = " ⭐ 推荐"
            elif not preference and mode == "高铁":
                recommendation = " ⭐ 推荐（性价比最高）"
            parts.append(f"  {icon} {mode}：约 {duration}{recommendation}")
    else:
        # ── 本地未命中 → 联网搜索 ──
        search_query = f"{origin}到{destination} 交通方式 高铁 飞机 距离 时间"
        if preference:
            search_query += f" {preference}"

        parts.append(f"🌐 数据来源：联网实时搜索")

        web_result = _web_search_fallback(search_query, max_results=4)

        if web_result and not web_result.startswith("⚠️"):
            parts.append(f"\n{web_result}")

            # 额外搜索高铁时刻
            train_result = _web_search_fallback(
                f"{origin} {destination} 高铁 时刻 票价", max_results=2
            )
            if train_result and not train_result.startswith("⚠️"):
                parts.append(f"\n🚄 高铁参考：\n{train_result}")
        else:
            # 联网也失败了，给通用建议
            parts.append(f"\n📏 预估距离：请以实际地图为准")
            parts.append(f"⚠️ 联网搜索暂不可用，以下为通用建议：")
            parts.append("\n🚀 建议交通方式：")
            parts.append("  🚄 高铁：中短途出行首选，准时舒适")
            parts.append("  ✈️ 飞机：超过1000km推荐，节省时间")
            parts.append("  🚗 自驾：灵活自由，适合沿途游览")

    parts.append(f"\n📋 出行小贴士：")
    parts.append("  • 提前预订车票/机票，假期提前至少2周")
    parts.append("  • 关注目的地天气，备好相应衣物")
    parts.append("  • 预留充足中转时间，避免赶车赶飞机")

    return "\n".join(parts)


# ==================== 5. 食材推荐菜谱 ====================

RECIPE_DB = {
    "鸡蛋": [
        {"name": "番茄炒蛋", "difficulty": "简单", "time": "15分钟",
         "ingredients": ["鸡蛋3个", "番茄2个", "葱", "盐", "糖"],
         "steps": ["鸡蛋打散加盐", "番茄切块", "先炒鸡蛋盛出", "炒番茄出汁", "倒入鸡蛋翻炒均匀"]},
        {"name": "蛋炒饭", "difficulty": "简单", "time": "10分钟",
         "ingredients": ["鸡蛋2个", "米饭1碗", "葱花", "盐", "酱油"],
         "steps": ["鸡蛋打散", "热油炒鸡蛋", "加入米饭翻炒", "加盐酱油调味", "撒葱花出锅"]},
        {"name": "蒸水蛋", "difficulty": "简单", "time": "20分钟",
         "ingredients": ["鸡蛋3个", "温水", "盐", "酱油", "香油"],
         "steps": ["鸡蛋打散加温水1:1.5", "过筛去泡", "盖保鲜膜", "水开后蒸10分钟", "淋酱油香油"]},
    ],
    "土豆": [
        {"name": "酸辣土豆丝", "difficulty": "简单", "time": "15分钟",
         "ingredients": ["土豆2个", "干辣椒", "醋", "盐", "蒜"],
         "steps": ["土豆切细丝泡水去淀粉", "热油爆香辣椒蒜", "大火快炒土豆丝", "加醋和盐调味"]},
        {"name": "土豆炖牛肉", "difficulty": "中等", "time": "1小时",
         "ingredients": ["土豆3个", "牛肉500g", "胡萝卜", "洋葱", "酱油", "八角"],
         "steps": ["牛肉焯水切块", "热油炒香洋葱", "加牛肉翻炒", "加水炖40分钟", "加土豆胡萝卜炖20分钟"]},
    ],
    "鸡胸肉": [
        {"name": "宫保鸡丁", "difficulty": "中等", "time": "25分钟",
         "ingredients": ["鸡胸肉300g", "花生米", "黄瓜", "干辣椒", "酱油", "醋", "糖", "淀粉"],
         "steps": ["鸡肉切丁腌制", "调酱汁(酱油醋糖淀粉)", "炒花生米盛出", "爆香辣椒炒鸡丁", "加黄瓜和酱汁翻炒"]},
        {"name": "香煎鸡胸", "difficulty": "简单", "time": "20分钟",
         "ingredients": ["鸡胸肉1块", "黑胡椒", "盐", "橄榄油", "柠檬"],
         "steps": ["鸡胸肉横切薄片", "盐和黑胡椒腌制15分钟", "中火煎至两面金黄", "挤柠檬汁"]},
    ],
    "西红柿": [
        {"name": "番茄牛腩", "difficulty": "中等", "time": "1.5小时",
         "ingredients": ["牛腩500g", "西红柿3个", "洋葱", "姜", "番茄酱", "盐"],
         "steps": ["牛腩焯水切块", "西红柿烫皮切块", "炒香洋葱姜", "加牛腩翻炒", "加西红柿炖1小时"]},
        {"name": "凉拌西红柿", "difficulty": "简单", "time": "5分钟",
         "ingredients": ["西红柿2个", "白糖"],
         "steps": ["西红柿切片", "撒白糖", "静置5分钟即可"]},
    ],
    "豆腐": [
        {"name": "麻婆豆腐", "difficulty": "中等", "time": "20分钟",
         "ingredients": ["嫩豆腐1块", "肉末100g", "豆瓣酱", "花椒粉", "葱", "淀粉"],
         "steps": ["豆腐切块焯水", "炒肉末至变色", "加豆瓣酱炒出红油", "加水和豆腐煮5分钟", "勾芡撒花椒粉葱花"]},
        {"name": "家常豆腐", "difficulty": "简单", "time": "15分钟",
         "ingredients": ["老豆腐1块", "青椒", "木耳", "酱油", "盐"],
         "steps": ["豆腐煎至两面金黄", "青椒木耳翻炒", "加酱油盐调味", "加豆腐翻炒均匀"]},
    ],
    "青菜": [
        {"name": "蒜蓉炒青菜", "difficulty": "简单", "time": "5分钟",
         "ingredients": ["青菜300g", "蒜3瓣", "盐", "油"],
         "steps": ["青菜洗净", "蒜切末", "热油爆香蒜", "大火快炒青菜", "加盐即可"]},
    ],
    "面条": [
        {"name": "葱油拌面", "difficulty": "简单", "time": "15分钟",
         "ingredients": ["面条200g", "葱100g", "酱油", "糖", "油"],
         "steps": ["面条煮熟过凉水", "葱切段炸至金黄", "酱油糖调汁", "浇葱油拌匀"]},
        {"name": "番茄鸡蛋面", "difficulty": "简单", "time": "15分钟",
         "ingredients": ["面条200g", "鸡蛋2个", "番茄2个", "盐", "葱"],
         "steps": ["鸡蛋炒散盛出", "番茄炒出汁", "加水煮开下面条", "加鸡蛋和盐调味"]},
    ],
}


@tool
def recommend_recipe(ingredients: str) -> str:
    """
    根据食材推荐菜谱。输入手头有的食材（逗号分隔），返回可做的菜。
    支持拍照识别食材后调用（配合 analyze_image 使用）。

    Args:
        ingredients: 食材列表，如"鸡蛋,西红柿,葱" 或 "土豆,牛肉"
    """
    items = [i.strip() for i in ingredients.replace("，", ",").split(",") if i.strip()]
    if not items:
        return "🍳 请输入至少一种食材，例如：recommend_recipe('鸡蛋,西红柿')"

    matched_recipes = []
    seen = set()

    for item in items:
        # 模糊匹配本地 RECIPE_DB
        for key, recipes in RECIPE_DB.items():
            if key in item or item in key:
                for recipe in recipes:
                    if recipe["name"] not in seen:
                        # 计算匹配度
                        matched_ingredients = sum(
                            1 for ing in recipe["ingredients"]
                            if any(item.lower() in ing.lower() for item in items)
                        )
                        match_score = matched_ingredients / len(recipe["ingredients"])
                        seen.add(recipe["name"])
                        matched_recipes.append({
                            **recipe,
                            "match_score": match_score,
                            "matched_by": item,
                            "source": "local",
                        })

    # 去重并按匹配度排序
    matched_recipes.sort(key=lambda r: (-r["match_score"], r["difficulty"]))

    if not matched_recipes:
        # ── 本地未命中 → 联网搜索 ──
        search_query = f"{' '.join(items)} 菜谱 做法 食材"
        web_result = _web_search_fallback(search_query, max_results=5)

        if web_result and not web_result.startswith("⚠️"):
            return (
                f"🍳 根据「{'、'.join(items)}」联网搜索到以下菜谱建议：\n\n"
                f"🌐 数据来源：联网实时搜索\n\n"
                f"{web_result}\n\n"
                f"💡 提示：搜索结果来自网络，请根据实际食材和口味调整。"
            )
        else:
            # 联网也失败
            return (
                f"🍳 本地和联网均未找到「{'、'.join(items)}」的匹配菜谱。\n\n"
                f"💡 建议：\n"
                f"  • 尝试搜索单个食材\n"
                f"  • 拍照识别食材后再查询（使用 analyze_image）\n"
                f"  • 试试这些常见食材：鸡蛋、土豆、鸡胸肉、西红柿、豆腐、青菜、面条"
            )

    parts = [f"🍳 根据「{'、'.join(items)}」找到 {len(matched_recipes)} 个菜谱："]
    parts.append(f"⚡ 数据来源：本地缓存\n")

    for i, recipe in enumerate(matched_recipes[:5], 1):
        match_pct = int(recipe["match_score"] * 100)
        parts.append(
            f"{i}. {recipe['name']}  ⏱{recipe['time']} | 难度：{recipe['difficulty']} | 匹配度：{match_pct}%"
        )
        parts.append(f"   食材：{'、'.join(recipe['ingredients'])}")
        steps_short = " → ".join(recipe["steps"][:3]) + ("..." if len(recipe["steps"]) > 3 else "")
        parts.append(f"   步骤：{steps_short}")
        parts.append("")

    return "\n".join(parts)


# ==================== 6. CLI 命令执行 ====================

# 危险命令黑名单
DANGEROUS_COMMANDS = [
    "rm -rf /", "format", "mkfs", "dd if=", ":(){ :|:& };:",
    "shutdown", "reboot", "init 0", "init 6",
    "> /dev/sda", "chmod 777 /", "chown -R",
]


@tool
def execute_cli(command: str, working_dir: str = ".") -> str:
    """
    在本地终端执行 CLI 命令并返回结果。
    ⚠️ 仅执行安全命令，危险操作会被拦截。

    支持的命令类型：
    - 文件操作：ls, dir, cat, head, tail, cp, mv, mkdir
    - 系统信息：whoami, pwd, date, ps, df, du
    - 网络工具：ping, curl, nslookup
    - Python 脚本：python script.py
    - Git 操作：git status, git log, git diff
    - 包管理：pip list, npm list

    Args:
        command: 要执行的命令
        working_dir: 工作目录（默认当前目录）
    """
    # 安全检查
    cmd_lower = command.lower().strip()
    for dangerous in DANGEROUS_COMMANDS:
        if dangerous.lower() in cmd_lower:
            return f"⛔ 安全拦截：检测到危险命令模式「{dangerous}」，已阻止执行。"

    # 额外安全检查
    dangerous_patterns = [
        r"rm\s+(-rf?|--recursive).*/",   # 递归删除根目录
        r"chmod\s+777\s+/",               # 修改系统目录权限
        r">\s*/dev/sd",                   # 覆盖磁盘
        r"mkfs\.",                        # 格式化
        r"eval\s+",                       # eval 注入
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, command):
            return f"⛔ 安全拦截：命令「{command}」包含危险模式，已阻止执行。"

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=EXECUTION_CONFIG.get("tool_timeout", 30),
            encoding="utf-8",
            errors="replace",
        )

        output = result.stdout.strip()
        error = result.stderr.strip()

        parts = []
        if output:
            output_short = output[:2000]  # 截断长输出
            parts.append(f"📤 stdout:\n{output_short}")
            if len(output) > 2000:
                parts.append(f"... (截断，完整输出 {len(output)} 字符)")

        if error:
            error_short = error[:500]
            parts.append(f"⚠️ stderr:\n{error_short}")

        parts.append(f"\n🔢 返回码：{result.returncode}")

        if not output and not error:
            parts.append("✅ 命令执行完成（无输出）")

        return "\n".join(parts)

    except subprocess.TimeoutExpired:
        return f"⏰ 命令执行超时（>{EXECUTION_CONFIG.get('tool_timeout', 30)}秒）"
    except FileNotFoundError:
        return f"❌ 命令不存在：{command.split()[0]}"
    except Exception as e:
        return f"❌ 执行失败：{str(e)}"


# ==================== 7. 文档上传与管理 ====================

@tool
def document_upload(file_path: str) -> str:
    """
    上传文档到知识库。支持 PDF、Word、Markdown、TXT、CSV、JSON、代码文件。
    上传后自动进行智能切片（RecursiveCharacterTextSplitter）和向量化，
    后续可通过 knowledge_qa 进行知识问答。

    Args:
        file_path: 文档文件路径，如 "./report.pdf" 或 "D:/docs/manual.docx"
    """
    import os as _os
    if not _os.path.exists(file_path):
        return f"❌ 文件不存在：{file_path}"

    try:
        from memory.manager import MemoryManager
        manager = MemoryManager.get_instance()
        result = manager.upload_document(file_path)
        if result["success"]:
            return (
                f"✅ 文档上传成功！\n"
                f"📄 文件：{result['source']}\n"
                f"📦 切片数：{result['chunks']} 个\n"
                f"🔍 已启用多路召回（原始/改写/HyDE/BM25）+ RRF 融合 + Qwen3-8B Reranker 精排\n"
                f"💡 现在可以对该文档进行知识问答了"
            )
        else:
            return f"❌ 上传失败：{result.get('error', '未知错误')}"
    except Exception as e:
        return f"❌ 文档上传失败：{str(e)}"


@tool
def list_documents() -> str:
    """
    列出知识库中所有已上传的文档。
    """
    try:
        from memory.manager import MemoryManager
        manager = MemoryManager.get_instance()
        docs = manager.list_documents()
        if not docs:
            return "📚 知识库为空，尚未上传任何文档。使用 document_upload 上传文档。"
        parts = ["📚 已上传文档：\n"]
        for i, doc in enumerate(docs, 1):
            parts.append(
                f"  {i}. {doc['source']} ({doc['file_type']}) — {doc['chunk_count']} 个切片"
            )
        return "\n".join(parts)
    except Exception as e:
        return f"❌ 获取文档列表失败：{str(e)}"


@tool
def delete_document(source: str) -> str:
    """
    从知识库中删除指定文档。
    Args:
        source: 文档文件名，如 "report.pdf"
    """
    try:
        from memory.manager import MemoryManager
        manager = MemoryManager.get_instance()
        result = manager.delete_document(source)
        if result["success"]:
            return f"🗑️ 已删除文档「{source}」，共移除 {result['deleted_chunks']} 个切片"
        else:
            return f"⚠️ 未找到文档「{source}」"
    except Exception as e:
        return f"❌ 删除失败：{str(e)}"


# ==================== 8. 知识问答（基于文档） ====================

@tool
def knowledge_qa(query: str) -> str:
    """
    基于已上传文档的知识问答。使用多路召回（原始查询/查询改写/HyDE/BM25）
    + RRF 融合 + Qwen3-8B Reranker 精排，从文档知识库中检索最相关内容。

    需要先使用 document_upload 上传文档。

    注意：本工具仅返回检索到的文档上下文，最终回答由输出层统一生成。

    Args:
        query: 关于文档内容的问题，如"报告中提到的Q3营收是多少？"
    """
    try:
        from memory.manager import MemoryManager

        manager = MemoryManager.get_instance()

        if manager.long_term.get_document_count() == 0:
            return "📚 知识库为空。请先使用 document_upload 工具上传文档。"

        # 检索相关知识（仅返回 context，不在此生成答案）
        qa_result = manager.knowledge_qa(query)

        if not qa_result["answerable"]:
            return (
                f"❓ 未在已上传的文档中找到与「{query}」相关的内容。\n\n"
                f"💡 建议：\n"
                f"  • 换一种表述方式重新提问\n"
                f"  • 确认相关文档已上传（使用 list_documents 查看）\n"
                f"  • 上传更多相关文档"
            )

        # 返回检索到的上下文和来源（由输出层 LLM 统一生成最终答案）
        sources = qa_result["sources"]
        source_names = list(set(s.get("source", "未知") for s in sources))
        context = qa_result["context"]

        return (
            f"📚 文档检索结果（共 {len(sources)} 个相关片段）：\n\n"
            f"【检索到的文档内容】\n{context}\n\n"
            f"📄 参考文档：{'、'.join(source_names)}\n\n"
            f"---\n请基于以上文档内容回答用户问题。严格基于文档内容，不要编造信息。"
        )

    except Exception as e:
        return f"❌ 知识问答失败：{str(e)}"


# ==================== 工具注册表 ====================

class ToolRegistry:
    """
    工具注册中心

    功能：
    - 注册/注销工具
    - 按名称查找工具
    - 获取工具描述（供 LLM 选择）
    - 安全执行（超时、错误处理）
    - 降级工具集（核心功能保障）
    """

    # 降级时保留的核心工具
    CORE_TOOLS = {"web_search", "calculator"}

    def __init__(self):
        self._tools: dict = {}
        self._register_defaults()

    def _register_defaults(self):
        """注册默认工具"""
        from tools.image_tools import analyze_image, generate_image
        for t in [web_search, calculator, get_weather, plan_travel,
                   recommend_recipe, execute_cli,
                   document_upload, list_documents, delete_document,
                   knowledge_qa, analyze_image, generate_image]:
            self.register(t)

    def register(self, tool_obj):
        """注册工具"""
        self._tools[tool_obj.name] = tool_obj

    def unregister(self, name: str):
        """注销工具"""
        self._tools.pop(name, None)

    def get(self, name: str):
        """获取工具"""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """列出所有工具名"""
        return list(self._tools.keys())

    def get_tool_descriptions(self) -> str:
        """获取所有工具的描述（供 LLM prompt）"""
        lines = []
        for name, t in self._tools.items():
            doc = (t.description or "无描述")[:150]
            lines.append(f"  • {name}: {doc}")
        return "\n".join(lines)

    def get_langchain_tools(self) -> list:
        """
        获取 LangChain 原生工具列表（用于 llm.bind_tools()）

        返回的是 @tool 装饰器包装的 LangChain BaseTool 实例，
        可在 API 层面强制工具调用约束（而非 prompt 约束）
        """
        return list(self._tools.values())

    def get_degraded_tools(self) -> list:
        """获取降级工具集（仅核心工具，LangChain 格式）"""
        return [t for name, t in self._tools.items() if name in self.CORE_TOOLS]

    def execute(self, name: str, params: dict) -> dict:
        """
        安全执行工具

        Returns:
            {"success": bool, "result": str, "duration_ms": float, "error": str|None}
        """
        tool_obj = self._tools.get(name)
        if not tool_obj:
            return {
                "success": False,
                "result": None,
                "duration_ms": 0,
                "error": f"工具「{name}」未注册。可用工具：{self.list_tools()}",
            }

        start = time.time()
        try:
            result = tool_obj.invoke(params)
            elapsed = (time.time() - start) * 1000
            return {
                "success": True,
                "result": str(result),
                "duration_ms": round(elapsed, 1),
                "error": None,
            }
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return {
                "success": False,
                "result": None,
                "duration_ms": round(elapsed, 1),
                "error": f"{type(e).__name__}: {str(e)}",
            }


# 全局单例
_tool_registry: Optional[ToolRegistry] = None


def get_all_tools() -> ToolRegistry:
    """获取工具注册表单例"""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry
