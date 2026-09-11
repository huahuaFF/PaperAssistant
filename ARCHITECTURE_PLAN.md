# 科研文献助手：架构计划（v0.1）

## 1. 目标与范围

构建一个单用户、本地优先、人工可控的科研文献助手。用户可以提出论文相关问题；系统优先检索本地知识库。当本地证据不足时，系统请求用户授权搜索 arXiv；用户从候选论文中选择要导入的项目后，系统下载 PDF、解析、向量化并写入本地知识库，最后基于新旧文献给出可追溯的知识汇报。

首版只覆盖以下闭环：

```text
用户提问
→ 本地文献检索
→ 覆盖度判断
→ 人工批准 arXiv 搜索
→ 人工选择导入论文
→ PDF 解析、向量化、入库
→ 重新检索
→ 带出处的论文知识汇报
```

### v0.1 不做的事情

- 不做多人协作、复杂权限与云端部署。
- 不做自动联网搜索或自动导入：外部搜索与入库均需用户明确确认。
- 不做复杂长期记忆、自动摘要或知识图谱。
- 不先做 Web 前端；先通过 FastAPI OpenAPI 文档和命令行 demo 验证完整后端闭环。
- 不引入 PostgreSQL、Redis、Celery、对象存储等基础设施。

## 2. 架构原则

1. **Agentic Workflow，而不是自由 Agent。** LangGraph 主图掌控状态、权限、审批与恢复；只在需要语义理解和推理的节点使用 Agent/LLM。
2. **人工控制外部副作用。** 模型不能直接下载 PDF、搜索外部文献或写入知识库。
3. **本地优先、知识增量。** 外部检索结果不自动成为长期知识；只有获批准导入的论文才会向量化并写入本地库。
4. **每个结论可追溯。** 生成回答只能引用已检索到的 chunk，服务端将 chunk 映射为论文、版本、页码与原文片段。
5. **轻量、可迁移。** 先以 Chroma + SQLite 实现；通过 Repository/Provider 接口隔离实现细节，之后可替换为 pgvector 或其他论文来源。

## 3. 技术选型

| 范围 | v0.1 选择 | 职责 |
|---|---|---|
| Python 环境 | Python 3.12 + `uv` | 依赖、虚拟环境、锁定版本 |
| Web API | FastAPI | 任务、审批、上传、报告与 SSE 事件 |
| 编排 | LangGraph `StateGraph` | 主工作流、分支、暂停与恢复 |
| LLM/Agent | MiniMax | 意图理解、覆盖度评估、检索规划、论文汇报 |
| Embedding | DashScope `text-embedding-v4`（1024 维） | 文献与问题向量化 |
| 向量库 | Chroma `PersistentClient` | 本地语义检索 |
| 业务持久化 | SQLite + SQLAlchemy + aiosqlite | 会话、消息、论文、任务、审批与报告 |
| 工作流持久化 | SQLite-backed LangGraph checkpointer | `interrupt()` 后的状态恢复 |
| PDF 文件 | 本地磁盘 | 原始 PDF 与解析产物 |
| 文献发现 | arXiv API | 论文元数据检索与 PDF 地址 |
| PDF 解析 | PyMuPDF（首版） | 文本和页码提取；后续可替换为结构化科研 PDF 解析器 |
| 测试/质量 | pytest、ruff、mypy | 单元、集成与静态检查 |

### 模型配置

```env
MINIMAX_API_KEY=
MINIMAX_BASE_URL=https://api.minimaxi.com/v1
MINIMAX_MODEL=MiniMax-M2.7-highspeed

DASHSCOPE_API_KEY=
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSIONS=1024
```

MiniMax 用于所有需要理解、分类或生成的任务。DashScope 只用于 embedding。二者必须分别封装，不能将供应商调用散落在业务节点中。

> 更换 embedding 模型或维度后，必须新建 Chroma collection 并全量重新索引；不得将不同向量空间混在同一个 collection。

## 4. 系统组成

```text
┌────────────────────────────────────────────────┐
│  客户端（首版：FastAPI /docs + 命令行 demo）     │
└──────────────────────┬─────────────────────────┘
                       │ REST / SSE
┌──────────────────────▼─────────────────────────┐
│ FastAPI                                         │
│ 会话、消息、任务、审批、上传、报告、事件推送     │
└──────────────────────┬─────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────┐
│ LangGraph 主工作流                              │
│ 路由、检索、审批、导入、重新检索、报告生成       │
└───────┬──────────────────────────┬─────────────┘
        │                          │
┌───────▼────────┐        ┌────────▼────────────┐
│ Local RAG      │        │ 导入流水线            │
│ Chroma 查询    │        │ arXiv/PDF/解析/嵌入   │
└───────┬────────┘        └────────┬────────────┘
        │                          │
┌───────▼──────────────────────────▼────────────┐
│ SQLite | Chroma 持久化目录 | 本地 PDF/解析文件   │
└────────────────────────────────────────────────┘
```

## 5. LangGraph 主工作流

```text
START
  ↓
classify_query                         [Agent]
  ↓
retrieve_local                         [确定性工具]
  ↓
assess_coverage                        [Agent + 阈值规则]
  ├─ 证据充分 ───────────────────────→ generate_report → END
  │                                      [Agent]
  └─ 证据不足
          ↓
     request_search_approval           [interrupt]
          ├─ 拒绝 ────────────────────→ explain_knowledge_gap → END
          │                              [Agent]
          └─ 批准
                 ↓
              build_arxiv_query        [Agent]
                 ↓
              search_arxiv             [确定性工具]
                 ↓
              request_import_approval  [interrupt]
                 ├─ 跳过 ─────────────→ report_candidates → END
                 └─ 选择论文
                        ↓
                    ingest_papers      [确定性服务]
                        ↓
                    retrieve_augmented_library
                        ↓
                    generate_report    [Agent]
                        ↓
                       END
```

### Agent 节点

- `classify_query`：识别是否为论文/科研问题、主题、时间范围与预期输出。
- `assess_coverage`：解释本地检索证据是否足够，并列出缺失方面。
- `build_arxiv_query`：将用户问题转换为受控的 arXiv 检索式。
- `generate_report`：仅基于已传入证据生成论文知识汇报。
- `explain_knowledge_gap`：在用户拒绝外部搜索时，明确说明本地库缺口。

### 确定性工具或服务节点

- Chroma 检索。
- arXiv API 请求。
- PDF 下载、解析、文本切块、embedding 和 Chroma 写入。
- SQLite 的任务、审批、论文与报告记录。

模型只能产生结构化建议，不能自行执行联网、下载或写库副作用。

## 6. 人工审批

必须存在两个审批点：

1. **外部搜索审批**：本地库证据不足时，询问用户是否允许搜索 arXiv。
2. **论文导入审批**：系统展示候选论文；用户勾选后才允许下载和入库。

审批节点使用 LangGraph `interrupt()`。系统为每次问题创建独立 `run_id`，并将它作为图的 `thread_id`。在等待审批时保存 checkpoint；用户提交决策后以同一个 `thread_id` 和 `Command(resume=...)` 继续。

所有副作用均放在 `interrupt()` 之后；如某节点可能重试，下载和写库操作必须使用幂等键。

## 7. 会话、任务与持久化

需要区分三个层次：

```text
Conversation：一段长期聊天
Run：用户的一次提问触发的一次工作流
Checkpoint：该 Run 的 LangGraph 暂停/恢复快照
```

```text
Conversation: “流匹配研究”
  ├─ Message: “流匹配有哪些代表性论文？”
  ├─ Run A（等待审批、导入、最终报告）
  └─ Message: “比较刚才导入的两篇方法”
       └─ Run B
```

目录与数据库分离：

```text
data/
  app.db                # 业务数据
  checkpoints.db        # LangGraph checkpoints
  chroma/               # Chroma 持久化目录
  papers/               # 原始 PDF
  parsed/               # PDF 解析产物
```

SQLite 的最小表：

```text
conversations
  id, title, created_at, updated_at

messages
  id, conversation_id, run_id, role, content, status, created_at

research_runs
  id, conversation_id, graph_thread_id, status,
  user_query, report_id, created_at, updated_at

approval_decisions
  id, run_id, approval_type, request_payload,
  decision_payload, created_at

papers
paper_versions
documents
ingestion_jobs
reports
report_citations
```

运行策略：

- 用户消息先写入 `messages`，再创建 `research_run`。
- `graph_thread_id = research_run.id`，以避免同一会话的并行问题混用 checkpoint。
- LangGraph 输出在前端流式展示；完整结果在完成后保存为 assistant message。
- 审批提示也保存为 assistant message，用户决策写入 `approval_decisions`。
- v0.1 每个会话只允许一个运行中或等待审批的 Run。

SQLite 使用 WAL、外键和合理的 busy timeout；当前单用户阶段不需要 PostgreSQL。

## 8. 文献与向量数据

Chroma 只保存可检索 chunk 与扁平 metadata。SQLite 作为论文、版本、任务和审批的事实来源。

每个 chunk 至少包含：

```text
chunk_id
workspace_id
paper_id
paper_version_id
arxiv_id
title
page_number
section
text
embedding_model_version
```

Chroma collection 初始命名：

```text
paper_chunks_embedding_v1
```

向量检索流程：

```text
用户问题
→ DashScope embedding
→ Chroma top-k 检索（按 workspace_id 过滤）
→ 分数、论文数和主题覆盖的规则判断
→ MiniMax 解释覆盖度或基于证据回答
```

覆盖不足不只意味着“零结果”。下列任一情况均可触发外部搜索审批：

- 没有命中 chunk。
- 高相关 chunk 过少。
- 结果仅提及概念，没有方法、实验或结论证据。
- 用户的多个子问题只有部分得到支持。
- 用户明确需要近年进展，而本地文献过旧。

## 9. 导入流水线

```text
用户选择候选论文
→ arXiv ID / DOI / 文件哈希去重
→ 下载 PDF
→ 保存原文件
→ 提取正文和页码
→ 按摘要、章节、段落切块
→ DashScope 批量 embedding
→ Chroma 写入
→ SQLite 更新导入状态
→ 标记为 ready
```

导入状态：

```text
pending → downloading → parsing → embedding → indexed → ready
```

失败状态和错误原因必须保存。重复执行时，服务根据 `arxiv_id + version` 和文件哈希跳过已完成步骤。

## 10. API 契约（后端优先）

```text
POST /api/conversations
  创建会话

POST /api/runs
  提交用户问题并创建 Run

GET /api/runs/{run_id}
  查询任务状态和待审批内容

GET /api/runs/{run_id}/events
  SSE：模型输出、节点进度和审批中断

POST /api/runs/{run_id}/resume
  提交搜索批准/拒绝或候选论文选择

POST /api/papers/upload
  上传本地 PDF，进入同一导入流水线

GET /api/papers
  查询本地文献库

GET /api/reports/{report_id}
  读取带证据引用的报告
```

在实现前端之前，可用 FastAPI `/docs` 和 `scripts/demo.py` 走通审批与恢复。

当前 Agent 核心可先通过命令行运行：

```powershell
uv run python -m scripts.run_agent
```

## 11. 项目结构

```text
paper-assistant/
  app/
    api/
    config.py
    graph/
      research_graph.py
      state.py
      nodes/
    llm/
      minimax.py
      dashscope_embeddings.py
    services/
      retrieval.py
      arxiv_provider.py
      ingestion.py
      report_service.py
    repositories/
      chroma_store.py
      sqlite_repository.py
    models/
    workers/
  scripts/
    demo.py
  tests/
  data/
  .env.example
  pyproject.toml
  uv.lock
```

业务逻辑只能通过 `ChromaStore` 访问 Chroma，通过 `LiteratureProvider` 访问 arXiv。后续替换 pgvector 或新增 PubMed、Crossref 等来源时，不改主图。

## 12. Python 与依赖管理

使用 `uv` 管理环境和依赖：

```text
Python 3.12
pyproject.toml：唯一依赖定义
uv.lock：提交到版本控制
.venv：本地生成，不提交
```

常用命令：

```powershell
uv sync
uv run uvicorn app.main:app --reload
uv run pytest
uv run ruff check .
```

初始运行时依赖：

```text
fastapi
uvicorn
langchain
langchain-openai
langgraph
langgraph-checkpoint-sqlite
chromadb
sqlalchemy
aiosqlite
pydantic-settings
httpx
pymupdf
dashscope
```

开发依赖：

```text
pytest
pytest-asyncio
ruff
mypy
```

`.env`、`.venv/` 与 `data/` 必须加入 `.gitignore`；`.env.example`、`pyproject.toml` 与 `uv.lock` 必须提交。

## 13. 实施顺序与验收

### 阶段 1：Agent 主图与节点（当前优先）

- 先固定完整 LangGraph 主图、状态契约、审批分支和路由不变量。
- 逐个实现 MiniMax 驱动的意图理解、证据覆盖度评估和 arXiv 检索规划 Agent 节点。
- Agent 使用 LangChain 的 `ChatPromptTemplate`、`MessagesPlaceholder`、`create_agent`、`ToolStrategy` 和 `with_retry`；Agent 不直接执行工具或修改数据。
- 提供命令行入口和 Fake Model 单测，先验证图拓扑、路由和节点协作。

### 阶段 2：本地文献工具与工作流

- 接入 DashScope、Chroma 和本地 PDF 导入。
- 实现 Agent 主图中的本地检索、覆盖度路由和引用证据传递。
- 建立 SQLite-backed LangGraph checkpointer。

### 阶段 3：受控 arXiv 扩展

- 实现 arXiv Provider。
- 实现“是否搜索”与“选择导入”两个 interrupt 审批点。
- 实现可恢复、幂等的 PDF 导入流水线。
- 导入后重新检索并自动完成原问题的报告。

### 阶段 4：后端 API、质量与体验

- 为 Agent 和工作流增加会话、Run、审批和 SSE API。
- 验证报告的引用只来自检索证据，建立端到端 demo 和测试用例。
- 后续再接入 Web 前端、混合检索、reranker、更多论文来源或多人协作。

### v0.1 验收标准

- 本地证据充分时，不调用 arXiv。
- 本地证据不足时，未经批准不调用 arXiv。
- 未经选择不下载、解析或导入论文。
- 导入后自动基于扩展后的知识库回答原问题。
- 每个关键结论可追溯到论文、版本、页码和 chunk。
- 任务在等待审批时重启服务，仍能恢复。
