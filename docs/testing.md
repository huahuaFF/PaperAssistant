# 测试工程

默认测试必须离线、可重复，不能消耗 MiniMax、DashScope 或 arXiv 配额。

```powershell
uv run pytest
uv run pytest -m unit
uv run pytest -m integration
```

`live` 标记仅用于人工确认真实供应商联通性，默认被 pytest 排除：

```powershell
uv run pytest -m live
```

## 分层

- **unit**：schema、prompt 上下文、检索筛选、解析、节点输入输出等单一组件。
- **integration**：使用 `tests/support` 的节点替身运行编译后的 LangGraph，验证完整路由、审批边界和终止状态；不得访问网络或真实模型。
- **live**：手工 smoke test，例如 `scripts/manual_invoke_workflow.py`。它可以使用真实凭据，但不属于 CI 的通过条件。

## 工作流验收契约

每次改变路由、状态字段或审批策略，都需要至少覆盖以下契约：

1. 本地证据充分：`research → retrieve_local → assess_coverage → generate_report`。
2. 用户拒绝搜索：不得调用 arXiv，必须终止于 `explain_knowledge_gap`。
3. 批准搜索但跳过导入：不得下载或向量化，必须终止于 `report_candidates`。
4. 指定论文直接导入：审批通过后只能按 `ingest_papers → retrieve_augmented_library → generate_report` 执行。

新发现的真实行为问题应先写成 integration 或 unit 场景测试，再修改实现。对于尚未实现的产品能力，先记录为验收场景，不用临时放宽已有断言。
