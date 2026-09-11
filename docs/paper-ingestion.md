# 论文导入流水线

`ingest_papers` 在 LangGraph 主图中是一个节点，但内部是一条确定性、可恢复的流水线。本阶段只完成其第一段：PDF 解析与分块；不会调用 Embedding，也不会写入 Chroma。

```text
PaperSource
  -> layout-aware PDF parsing
  -> page-preserving text cleanup
  -> section detection
  -> token chunking
  -> ParsedPaper JSON (human review)
  -> [next] DashScope embedding and Chroma upsert
```

## 当前数据契约

`PaperSource` 必须由上游协调器指定稳定的 `paper_id`，不能从文件名推导。生成的每条 `PaperChunk` 包含：

- `chunk_id`: `<paper_id>:section_token_v2:p<page>:<content_type>:c<index>`，用于幂等 upsert；
- `page_number`、`section`、`title`：保证未来回答可追溯；
- `text`: 原始清洗后的证据文本；
- `content_type`: `body` 或独立的 `figure_caption`；
- `embedding_text`: 仅供向量化使用的标题、章节、页码和正文组合。

## 解析策略

- 使用 PyMuPDF 按页面文本块读取双栏 PDF，再转为 LangChain `Document` 和 token splitter 的输入。
- 页眉页脚、arXiv 边栏、版权与会议出版信息会在解析阶段过滤。
- 第一页的标题作者区不进入正文 chunk；标题单独提取。
- 同时识别固定章节名和通用编号标题，例如 `3 Flow-Based Recommender - FlowCF`。
- References 之后的内容默认不入库；无文本或没有合格 chunk 的 PDF 显式报错，等待 OCR 流程。

## 当前分块参数

| 参数 | 值 |
| --- | ---: |
| chunk 大小 | 800 tokens |
| overlap | 120 tokens |
| 最小 chunk | 120 tokens |
| 页面边界 | 不跨页 |

## 手动运行

```powershell
uv run python -m scripts.parse_paper data/papers/2502.07303v2.pdf --paper-id local-2502-07303-v2 --arxiv-id 2502.07303
```

产物写入 `data/parsed/`。向量化前必须检查：标题正确、chunk ID 唯一、页码合理、章节没有错误继承，以及正文没有作者/版权等版面噪声。

## 已审核产物的向量化

审核通过后，`PaperIndexer` 将使用 LangChain Chroma 的 `add_documents` 接口以每批最多 10 个 chunk 写入；稳定 ID 使重复运行变为 upsert，而不是重复插入。为了让检索结果仍能展示原文，Chroma 中的 `page_content` 使用带标题和章节的 `embedding_text`，`source_text` metadata 保存原始正文用于引用摘录。
