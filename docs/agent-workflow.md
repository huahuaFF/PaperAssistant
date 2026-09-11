# Agent 主图（节点实现前）

本文件固定主图的节点和路由边界。它不是实现清单：每个节点在单独的后续步骤中完成。

```mermaid
flowchart TD
    start([START]) --> classify_query
    classify_query -->|科研请求| retrieve_local
    classify_query -->|指定论文导入| request_import_approval
    classify_query -->|需要澄清| request_clarification
    classify_query -->|不在范围内| explain_out_of_scope
    retrieve_local --> assess_coverage

    assess_coverage -->|本地证据充分| generate_report
    assess_coverage -->|本地证据不足| request_search_approval

    request_search_approval -->|用户同意搜索| build_arxiv_query
    request_search_approval -->|用户拒绝搜索| explain_knowledge_gap
    build_arxiv_query --> search_arxiv
    search_arxiv --> request_import_approval

    request_import_approval -->|选择论文导入| ingest_papers
    request_import_approval -->|跳过导入| report_candidates
    ingest_papers --> retrieve_augmented_library
    retrieve_augmented_library --> generate_report

    generate_report --> done([END])
    explain_knowledge_gap --> done
    report_candidates --> done
    request_clarification --> done
    explain_out_of_scope --> done
```

## 节点完成顺序

1. `classify_query`
2. `retrieve_local`
3. `assess_coverage`
4. `request_search_approval`
5. `build_arxiv_query`
6. `search_arxiv`
7. `request_import_approval`
8. `ingest_papers`
9. `retrieve_augmented_library`
10. `generate_report`
11. `request_clarification`、`explain_out_of_scope`、`explain_knowledge_gap` 与 `report_candidates`

## 已完成：`classify_query`

该节点使用 LangChain Runnable 组合，不手写 JSON 提取或 Pydantic 解析：

```text
ChatPromptTemplate
+ MessagesPlaceholder(chat_history, 最近 4 条)
+ create_agent(..., response_format=ToolStrategy(QueryIntent))
+ Runnable.with_retry(stop_after_attempt=2)
```

分类结果中的 `intent.route` 由确定性路由函数消费：

```text
research             → retrieve_local
direct_import        → request_import_approval
needs_clarification  → request_clarification
out_of_scope         → explain_out_of_scope
```

分类上下文固定包含当前查询、限长会话摘要、最近消息与最多五篇活跃论文的简写 metadata；分类结果是内部状态，不写入面向用户的聊天历史。

## 已完成：`retrieve_local`

这是确定性 RAG 节点，不是 Agent，不使用 MiniMax。它将原始问题、`intent.topic`、去重后的 `key_concepts` 与已引用论文 ID 组合为透明的 embedding query，然后通过 LangChain Chroma 的带相关性分数检索接口获取候选 chunk。

- 取最多 12 个候选，相关性阈值为 `0.35`；每篇论文最多保留 2 个 chunk。
- 每个入选结果必须有 `paper_id`、`chunk_id`、`title`，可选 `page_number`；缺少引用溯源数据会显式失败。
- 空库或没有达到阈值的结果返回空 `local_evidence`，后续 `assess_coverage` 负责决定是否请求 arXiv 搜索权限。

## 进行中：`ingest_papers`

已完成 PDF 解析、版面块排序、清洗、章节识别和页内 token 分块，结果先写入 `data/parsed/` 供人工检查。DashScope 批量 embedding、Chroma 幂等 upsert、SQLite 导入状态机与该图节点接线仍是下一阶段，未经人工检查的 parsed artifact 不会进入向量库。

## 已完成：`assess_coverage`

该节点只判断本地证据是否足够回答，不生成回答。没有 `local_evidence` 时由确定性规则直接要求 arXiv 搜索审批；有证据时使用结构化 `CoverageAssessment` Agent 输出 `sufficient`、置信度、缺失方面和下一步动作。其结构化结果由确定性边路由到 `generate_report` 或 `request_search_approval`。

## 已完成：`request_search_approval`

当本地证据不足时，节点使用 LangGraph `interrupt()` 暂停并返回 JSON 审批载荷。调用方必须以同一个 `thread_id` 使用 `Command(resume={"decision": "approve" | "reject"})` 恢复；SQLite checkpointer 会持久化暂停状态。批准后才路由到 `build_arxiv_query`，拒绝则进入 `explain_knowledge_gap`。

## 已完成：`build_arxiv_query`

该 Agent 节点必须先验证 `search_approval == "approved"`，再根据 `QueryIntent` 与 `CoverageAssessment` 输出结构化 `ArxivSearchPlan`。它只生成可复现的 arXiv API 查询表达式、分类、理由与结果上限；不执行网络请求。

## 路由不变量

- `search_arxiv` 只可能从 `request_search_approval` 的 `approved` 分支到达。
- `ingest_papers` 只可能从 `request_import_approval` 的 `selected` 分支到达，且至少选择一篇论文。
- 任何路由所需状态缺失时，图抛出 `WorkflowRoutingError`，不猜测用户意图。
