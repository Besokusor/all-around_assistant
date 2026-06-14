"""
图像工具 — 图像分析和生成

支持：
  - analyze_image: 分析图像内容（食材识别、物体检测、场景理解）
  - generate_image: 根据描述生成图像
"""
import base64
import os
from typing import Optional
from langchain_core.tools import tool


# ==================== 图像分析 ====================

@tool
def analyze_image(image_source: str, task: str = "general") -> str:
    """
    分析图像内容。支持本地文件路径或 base64 编码的图像。

    分析任务类型：
    - general:     通用分析（物体、场景、文字）
    - food:        食材识别（拍照冰箱/食材 → 列出食材清单）
    - document:    文档/截图识别
    - scene:       场景理解
    - travel:      地标/景点识别

    Args:
        image_source: 图像路径（如 "./photo.jpg"）或 base64 字符串
        task: 分析任务类型
    """
    # 加载图像
    image_data = _load_image(image_source)
    if not image_data:
        return "❌ 无法加载图像。请提供有效的文件路径或 base64 编码。"

    # 根据任务类型构建提示词
    prompts = {
        "food": (
            "请仔细识别这张图片中的所有食材和食物。"
            "列出你能看到的每一种食材名称（用逗号分隔）。"
            "如果有包装食品，也请识别食品类型。"
            "最后，给出一个综合的食材清单，格式如下：\n"
            "【识别食材】食材1, 食材2, 食材3\n"
            "【建议】可以尝试用这些食材做什么菜？"
        ),
        "general": (
            "请详细描述这张图片的内容。包括："
            "1. 主要物体/人物\n2. 场景/环境\n3. 颜色和氛围\n4. 任何文字内容"
        ),
        "document": (
            "请识别并提取这张图片中的所有文字内容。"
            "如果有表格或结构化数据，请保持格式。"
        ),
        "scene": (
            "请分析这张图片的场景："
            "1. 地点类型（室内/室外/自然/城市等）\n"
            "2. 光照条件\n3. 主要元素\n4. 氛围和风格"
        ),
        "travel": (
            "请识别这张图片中的地点/地标/景点。"
            "如果认识，请提供以下信息：\n"
            "1. 地点名称\n2. 所在城市/国家\n3. 简要介绍\n4. 游览建议"
        ),
    }

    prompt = prompts.get(task, prompts["general"])

    image_url = image_source

    # Qwen3-VL-32B 多模态识别（讯飞 MaaS /v2）
    try:
        from config import QWEN_MAAS_CONFIG
        import openai as _openai

        qwen_client = _openai.OpenAI(
            api_key=QWEN_MAAS_CONFIG["api_key"],
            base_url=QWEN_MAAS_CONFIG["base_url"],
        )

        # HTTP URL → base64
        img_for_vision = image_data
        if image_data.startswith(("http://", "https://")):
            import requests as _req
            resp = _req.get(image_data, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 100:
                ct = resp.headers.get("Content-Type", "image/png")
                img_for_vision = f"data:{ct};base64,{base64.b64encode(resp.content).decode()}"

        response = qwen_client.chat.completions.create(
            model="xop3qwen32bvl",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": img_for_vision}},
            ]}],
            max_tokens=1024,
        )
        result = response.choices[0].message.content
        if result:
            return f"[图片分析] {result}\n(图片来源: {image_url})"
    except Exception as e:
        return f"[图片分析失败] {str(e)[:200]}\n(图片来源: {image_url})"

    return f"[图片] 用户上传了图片: {image_url}"


# ==================== 图像生成 ====================

@tool
def generate_image(description: str, style: str = "realistic") -> str:
    """
    根据文字描述生成图像。

    Args:
        description: 图像描述（越详细越好）
        style: 风格 — realistic/photographic/anime/illustration/pixel_art/sketch

    Returns:
        生成图像的路径或 URL
    """
    # 注意：需要接入实际的图像生成 API（如 DALL-E, Stable Diffusion 等）
    # 这里是框架实现，实际部署时替换为真实 API 调用

    style_prompts = {
        "realistic": "photorealistic, highly detailed, 8k resolution",
        "photographic": "professional photography, natural lighting, sharp focus",
        "anime": "anime style, studio ghibli, vibrant colors, clean lines",
        "illustration": "digital illustration, vector art, flat design, colorful",
        "pixel_art": "pixel art style, 16-bit, retro game aesthetic",
        "sketch": "pencil sketch, hand-drawn, monochrome, artistic",
    }

    enhanced_prompt = f"{description}, {style_prompts.get(style, style_prompts['realistic'])}"

    # 尝试使用 DeepSeek 或其他 API 生成
    # 实际实现需要替换为真实的图像生成 API
    try:
        # 方案1：使用 OpenAI DALL-E 兼容接口
        # from openai import OpenAI
        # client = OpenAI(api_key=..., base_url=...)
        # response = client.images.generate(prompt=enhanced_prompt, n=1, size="1024x1024")
        # return f"🎨 图像已生成：{response.data[0].url}"

        # 方案2：本地 Stable Diffusion
        # import requests
        # response = requests.post("http://localhost:7860/sdapi/v1/txt2img", json={...})

        # 当前返回提示信息
        return (
            f"🎨 图像生成请求已记录\n"
            f"📝 描述：{description}\n"
            f"🎭 风格：{style}\n"
            f"🔧 增强提示词：{enhanced_prompt}\n\n"
            f"⚠️ 当前为框架模式，请接入实际图像生成 API（DALL-E / Stable Diffusion / Midjourney API）"
        )

    except Exception as e:
        return f"❌ 图像生成失败：{str(e)}"


# ==================== 工具函数 ====================

def _load_image(source: str) -> Optional[str]:
    """
    加载图像为 base64 data URL

    支持：
    - 本地文件路径："./photo.jpg"
    - base64 字符串（自动识别）
    - HTTP URL（直接返回）
    """
    if not source:
        return None

    # 已经是 data URL
    if source.startswith("data:image/"):
        return source

    # HTTP URL
    if source.startswith(("http://", "https://")):
        return source

    # base64 字符串（无前缀）→ 自动添加
    if len(source) > 100 and not os.path.exists(source):
        # 尝试判断是否为纯 base64
        try:
            base64.b64decode(source[:100], validate=True)
            return f"data:image/jpeg;base64,{source}"
        except Exception:
            pass

    # 本地文件路径
    if os.path.isfile(source):
        try:
            ext = os.path.splitext(source)[1].lower()
            mime_map = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".gif": "image/gif",
                ".webp": "image/webp", ".bmp": "image/bmp",
            }
            mime = mime_map.get(ext, "image/jpeg")
            with open(source, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime};base64,{encoded}"
        except Exception:
            return None

    return None
