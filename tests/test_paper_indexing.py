from langchain_core.documents import Document

from app.services.paper_indexing import PaperIndexer
from app.services.paper_parsing import PaperChunk, ParsedPaper


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls: list[tuple[list[Document], list[str]]] = []

    def add_documents(self, documents: list[Document], *, ids: list[str]) -> list[str]:
        self.calls.append((documents, ids))
        return ids


def chunk(index: int) -> PaperChunk:
    return PaperChunk(
        chunk_id=f"paper-1:section_token_v2:p0001:body:c{index:04d}",
        paper_id="paper-1",
        title="A Paper",
        page_number=1,
        section="Method",
        chunk_index=index,
        source_type="local_pdf",
        text=f"Source text {index}.",
    )


def test_indexer_batches_documents_and_preserves_raw_source_text() -> None:
    store = FakeVectorStore()
    parsed = ParsedPaper(
        paper_id="paper-1",
        source_path="paper.pdf",
        file_sha256="a" * 64,
        title="A Paper",
        page_count=1,
        chunks=[chunk(index) for index in range(1, 12)],
    )

    result = PaperIndexer(store).index(parsed)

    assert [len(documents) for documents, _ in store.calls] == [10, 1]
    first_document = store.calls[0][0][0]
    assert first_document.page_content.startswith("Title: A Paper\nSection: Method")
    assert first_document.metadata["source_text"] == "Source text 1."
    assert result.upserted_chunk_ids == [item.chunk_id for item in parsed.chunks]
