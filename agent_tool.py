from langchain_core.tools import tool
import requests


# 计算tool
@tool
def calculator(expression: str) -> str:
    """
    用于计算数学表达式，例如：3+5*10
    arg:
     expression: str ,例如 3+5*10

    """
    try:
        res = eval(expression, {"__builtins__": None}, {})
        return f"计算结果：{res}"
    except:
        return "计算失败"




# ====================== 工具：Open‑Meteo 稳定天气 ======================
city_latlng = {
    "北京": (39.9042, 116.4074),
    "上海": (31.2304, 121.4737),
    "广州": (23.1200, 113.3271),
    "深圳": (22.6272, 114.0737),
    "武汉": (30.5928, 114.3055),
    "杭州": (30.2741, 120.1551),
    "成都": (30.5728, 104.0668),
    "南京": (32.0603, 118.7969)
}
@tool
def get_real_weather(city: str) -> str:
    """
    查询指定城市实时天气（温度、湿度、风速、天气状况）
    arg:
        city:str
    """
    if city not in city_latlng:
        return f"暂不支持 {city}，请用：{list(city_latlng.keys())}"

    lat, lon = city_latlng[city]
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
    )
    try:
        res = requests.get(url, timeout=8).json()  # 缩短超时
        curr = res["current"]
        temp = curr["temperature_2m"]
        hum = curr["relative_humidity_2m"]
        wind = curr["wind_speed_10m"]
        code = curr["weather_code"]

        # 简单天气文字映射
        wmap = {
            0: "晴朗", 1: "晴", 2: "多云", 3: "阴天",
            45: "雾", 51: "小雨", 53: "中雨", 55: "大雨",
            61: "小雨", 63: "中雨", 65: "大雨", 80: "阵雨"
        }
        cond = wmap.get(code, f"未知天气({code})")

        return (
            f"{city} 实时天气：{cond}，温度 {temp}℃，"
            f"湿度 {hum}%，风速 {wind} m/s"
        )
    except requests.exceptions.Timeout:
        return "天气查询超时，请稍后再试"
    except Exception as e:
        return f"天气查询失败：{str(e)}"
if __name__ == "__main__":
    # from tavily import TavilyClient
    # import os
    # from dotenv import load_dotenv
    # load_dotenv()
    # tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    # response = tavily_client.search(
    #             query="武汉今天的天气怎么样", search_depth="basic", max_results=5, include_answer=True
    #         )
    response = {'query': '武汉今天的天气怎么样', 'follow_up_questions': None, 'answer': 'Today in Wuhan, it is mostly cloudy with a temperature range of 17°C to 26°C. The humidity is high, and there is an 80% chance of rain. The air quality is moderate.', 'images': [], 'results': [{'url': 'https://weathernew.pae.baidu.com/weathernew/pc?query=%E6%B9%96%E5%8C%97%E6%AD%A6%E6%B1%89%E5%A4%A9%E6%B0%94&srcid=4982', 'title': '武汉 - 百度', 'content': '武汉 今天：小雨 17°~21°C 东北风3级. 06月08日 周一 农历四月廿三. 19°. 32 优. 雾 东北风 4级. 湿度97% 日出05:20 日落19:24. 24小时预报.', 'score': 0.829285, 'raw_content': None}, {'url': 'https://www.aqi.in/weather/cn/china/hubei/wuhan', 'title': 'Wuhan天气状况：气温| 30天预报 - AQI.in', 'content': "# Wuhan Weather Conditions Current Temperature Level. 最后更新：2026-06-04 08:00 (本地时间). 星期六 (六月 13) : 气温 26°C，湿度 69%，Moderate or heavy rain shower 天气状况。. ## **Wuhan**'s Locations Weather Conditions. ## FAQs About Wuhan Weather Conditions. Wuhan 今天 04 六月 2026 的当前气温是多少？. Wuhan 的当前气温为 26°C，体感温度为 29°C。今日预报显示最高气温 34°C，最低气温 24°C，截至 08:00 AM 04 六月 2026，全天温差为 10 度。. 以下是截至 08:00 AM 04 六月 2026 Wuhan 当前天气状况的完整概览：. Wuhan 从 08:00 AM 04 六月 2026 起的10天天气预报是什么？. 从 08:00 AM 04 六月 2026 起，Wuhan 的10天预报显示以下趋势：. 星期五 (六月 5) : 气温 30°C，湿度 53%，Sunny 天气状况。. 星期三 (六月 10) : 气温 30°C，湿度 38%，Partly Cloudy 天气状况。. 截至 08:00 AM 04 六月 2026，Wuhan 今晚约 2026-06-04 08:00 有 5% 的降雨概率。当前降水量为 0 mm。建议居民出门前携带雨伞。. 六月 2026 Wuhan 预计将有 5 个晴天、20 个雨天、5 个阴天和 0 个雪天。本月气温从最低约 20°C 到最高 35°C。. Prana Air Weather Station with air quality monitoring device. ### Construction Air Pollution: The Invisible Hazard Reshaping India's Urban Air. Kitchen Air Pollution: How Cooking Can Spike Your Indoor AQI. Construction Air Pollution: The Invisible Hazard Reshaping India's Urban Air.", 'score': 0.751375, 'raw_content': None}, {'url': 'https://weather.yahoo.com/zh-hant-hk/cn/%E6%B9%96%E5%8C%97/%E6%AD%A6%E6%BC%A2%E5%B8%82', 'title': 'CN武漢市的天氣預報、情況和地圖 - Yahoo Weather', 'content': '今天大致多雲，最高溫度為26°C，最低溫度為17°C。 降雨機率為80%。', 'score': 0.74563974, 'raw_content': None}, {'url': 'https://www.msn.com/zh-tw/weather/forecast/in-Wuhan,%20%20%20%20%20Hubei', 'title': '武漢市, 湖北省, 中國天氣預測 - MSN', 'content': 'Ninnescah, 堪薩斯州, 美國. 最近存取的位置. 多雲. 武漢市, 湖北省, 中國. 目前天氣. 08:00. 多雲 20°C. 體感17°. 多雲. 空氣品質. 83 · 風速. 16 公里/小時.', 'score': 0.70823324, 'raw_content': None}, {'url': 'https://www.ventusky.com/zh-tw/wuhan', 'title': '天氣- 武汉市- 14天預報：氣溫、風和雷達 - Ventusky', 'content': '| 多雲 25\xa0°C 0\xa0mm 0 %  東北  15\xa0km/h | 多雲 24\xa0°C 0\xa0mm 0 %  東北  15\xa0km/h | 大部分多雲 24\xa0°C 0\xa0mm 0 %  東北  17\xa0km/h | 大部分多雲 24\xa0°C 0\xa0mm 10 %  北  16\xa0km/h | 大部分多雲 25\xa0°C 0\xa0mm 10 %  北  18\xa0km/h  陣風:  43\xa0km/h | 灰濛蒙 25\xa0°C 0\xa0mm 10 %  北  20\xa0km/h  陣風:  43\xa0km/h | 灰濛蒙 26\xa0°C 0\xa0mm 10 %  北  18\xa0km/h  陣風:  43\xa0km/h | 多雲 26\xa0°C 0\xa0mm 10 %  北  18\xa0km/h  陣風:  43\xa0km/h | 灰濛蒙 25\xa0°C 0\xa0mm 10 %  北  19\xa0km/h  陣風:  43\xa0km/h | 灰濛蒙 23\xa0°C 0\xa0mm 0 %  北  19\xa0km/h  陣風:  43\xa0km/h | 灰濛蒙 22\xa0°C 0\xa0mm 0 %  北  17\xa0km/h  陣風:  43\xa0km/h | 灰濛蒙 21\xa0°C 0\xa0mm 10 %  北  17\xa0km/h | 灰濛蒙 21\xa0°C 0\xa0mm 10 %  北  15\xa0km/h | 灰濛蒙 20\xa0°C 0\xa0mm 10 %  北  15\xa0km/h | 灰濛蒙 20\xa0°C 0\xa0mm 10 %  北  15\xa0km/h | 灰濛蒙 20\xa0°C 0\xa0mm 20 %  北  15\xa0km/h | 灰濛蒙 20\xa0°C 0\xa0mm 30 %  北  17\xa0km/h | 夾雜陣雨 20\xa0°C 0.2\xa0mm 70 %  北  17\xa0km/h | 陰有小雨 20\xa0°C 0.2\xa0mm 90 %  北  17\xa0km/h | 陰有雨 19\xa0°C 0.9\xa0mm 90 %  北  15\xa0km/h | 陰有雨 19\xa0°C 0.4\xa0mm 90 %  北  15\xa0km/h | 陰有雨 18\xa0°C 0.3\xa0mm 90 %  北  13\xa0km/h | 陰有雨 18\xa0°C 0.6\xa0mm 90 %  北  13\xa0km/h | 陰有小雨 18\xa0°C 0.2\xa0mm 90 %  北  13\xa0km/h | 灰濛蒙 18\xa0°C 0\xa0mm 50 %  北  13\xa0km/h |.', 'score': 0.7009158, 'raw_content': None}], 'response_time': 1.29, 'request_id': '84a725c2-6a45-4182-8b06-3350ed6f9e88'}

    print(response['answer'])