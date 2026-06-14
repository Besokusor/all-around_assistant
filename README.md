# 🤖 All-Around Assistant — 个人全能助手

基于 LangGraph 五层架构的智能 Agent，支持智能问答、联网搜索、出行规划、计算、拍食材推荐菜谱、CLI 执行、图像分析、知识库问答，附带 FastAPI Web 前端。

---

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────────┐
│                     🧠 记忆层                            │
│  短期记忆(State) + 长期记忆(Chroma+Qwen3-8B) + 用户画像(MySQL) │
├─────────────────────────────────────────────────────────┤
│                     💡 思考层                            │
│  PydanticOutputParser + json_mode 严格结构化规划          │
├─────────────────────────────────────────────────────────┤
│                     ⚡ 执行层                            │
│  ToolRegistry 统一调度 10 个工具，多智能体协同             │
├─────────────────────────────────────────────────────────┤
│                     👁️ 观察层                            │
│  评估执行结果 → continue / retry / revise / fallback / end │
├─────────────────────────────────────────────────────────┤
│                     🛡️ 兜底层                            │
│  Level1 检查点回退 → Level2 降级核心工具 → Level3 人工介入  │
└─────────────────────────────────────────────────────────┘
```

### LangGraph 工作流（7 节点）

```
START → 🧠 memory_retrieve → 💡 planner → ⚡ executor ↔ 👁️ observer
                                                    ↕
                                              🆘 fallback
                                                    ↓
                              💾 memory_update → 💬 response → END
```

---

## 🛠️ 技能 & 工具

| 工具 | 功能 | 实现方式 |
|------|------|---------|
| `web_search` | 联网搜索 | Tavily API |
| `calculator` | 数学计算 | 安全沙箱 eval |
| `get_weather` | 天气查询 | Open-Meteo API |
| `plan_travel` | 出行规划 | 高铁/飞机/自驾对比 |
| `recommend_recipe` | 食材推荐菜谱 | 内置菜谱库 + 图像识别联动 |
| `execute_cli` | CLI 命令执行 | subprocess 安全沙箱 |
| `document_upload` | 文档上传 | PDF/Word/Markdown → 智能切片 → 向量化 |
| `list_documents` | 列出文档 | 查看知识库中所有已上传文档 |
| `delete_document` | 删除文档 | 从知识库移除指定文档 |
| `knowledge_qa` | 知识问答 | 多路召回 + RRF + Reranker 精排 |
| `analyze_image` | 图像分析 | Vision 模型多模态识别 |
| `generate_image` | 图像生成 | API 图像生成 |

---

## 🔍 文档检索（多路召回 + RRF + Reranker）

```
User Query
  ├─ ① 原始查询向量召回 (Qwen3-8B Embedding)
  ├─ ② LLM 查询改写 → 向量召回
  ├─ ③ HyDE 假设文档 → 向量召回
  └─ ④ BM25 关键词检索（中文 2-gram 分词）
        ↓
  RRF 融合 (k=60)
        ↓
  Qwen3-8B Reranker 精排
        ↓
  Final Top-K
```

**文档切片**：RecursiveCharacterTextSplitter（按 `\n\n → \n → 。→ ；` 优先级递归切分），避免固定长度切分破坏语义。

---

## 🎯 关键特性

- **严格结构化输出**：规划器使用 `json_mode`（API 层 `response_format` 约束 JSON）+ Pydantic 双重校验，非 prompt 约束
- **流式返回**：CLI 和 Web 均支持逐 token 流式输出；Web 端通过 SSE 推送
- **记忆智能压缩**：短期记忆超阈值自动摘要压缩存入 Chroma；用户偏好累计 ≥3 次确认写入永久画像
- **Checkpoint 回退**：LangGraph MemorySaver 保存状态，失败时从检查点恢复
- **多级降级**：Error → 重试 → 回退检查点 → 核心工具降级 → 人工介入
- **图像输入**：支持拍照/拖拽/粘贴，识别食材后联动菜谱推荐
- **异步队列**：Web 端每 session 独立 `asyncio.Queue`，保证请求串行不冲突

---

## 🖥️ 使用方式

### CLI 模式

```bash
python agent.py                        # 交互模式
python agent.py "今天天气怎么样"          # 单次查询
python agent.py --image photo.jpg "有什么食材"  # 图像分析
python agent.py --stream "搜索AI新闻"    # 流式输出
```

### Web 模式

```bash
pip install -r requirements.txt
python run_web.py                      # 启动服务器
# 打开 http://localhost:8000
```

---

## 📦 技术栈

| 层级 | 技术 |
|------|------|
| **LLM** | DeepSeek-v4 (Flash/Pro) |
| **Agent 框架** | LangGraph + LangChain |
| **Embedding** | Qwen3-8B (xop3qwen8bembedding, 讯飞 MaaS, 768维) |
| **Reranker** | Qwen3-8B (xop3qwen8breranker, 讯飞 MaaS) |
| **向量数据库** | ChromaDB |
| **结构化输出** | Pydantic + json_mode |
| **搜索** | Tavily API |
| **关系数据库** | MySQL (用户画像永久存储) |
| **文档解析** | pdfplumber / pypdf / python-docx / chardet |
| **文档切片** | RecursiveCharacterTextSplitter |
| **关键词检索** | rank-bm25 |
| **Web 框架** | FastAPI + Uvicorn |
| **流式推送** | SSE (Server-Sent Events) |
| **前端** | Vanilla HTML/CSS/JS（暗色主题 + 响应式） |

---

## 📂 项目结构

```
all-around_assistant/
├── agent.py              # 主入口：AllAroundAssistant 类 + CLI
├── state.py              # AgentState 状态定义（26 字段）
├── config.py             # 全局配置
├── run_web.py            # Web 服务器启动脚本
│
├── graph/                # LangGraph 工作流
│   ├── builder.py        #   条件路由 + checkpoint
│   └── nodes.py          #   7 节点实现
│
├── tools/                # 工具层
│   ├── registry.py       #   12 个工具 + 注册中心
│   └── image_tools.py    #   图像分析/生成
│
├── memory/               # 记忆层
│   ├── manager.py        #   记忆协调器
│   ├── long_term.py      #   Chroma + Qwen Embedding
│   ├── user_profile.py   #   MySQL 用户画像
│   ├── document_handler.py # 文档解析 + 智能切片
│   └── rerank.py         #   多路召回 + RRF + Reranker
│
├── server/               # Web 层
│   ├── main.py           #   FastAPI 应用
│   ├── session_manager.py #  会话管理 + 异步队列
│   ├── routes/           #   API 路由
│   └── static/           #   前端页面
│
└── scripts/
    └── init_mysql.sql    # 数据库初始化
```
