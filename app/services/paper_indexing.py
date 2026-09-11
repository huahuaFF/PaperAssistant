"""Batch, idempotent Chroma indexing for parsed research-paper chunks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from langchain_core.documents import Document

from app.services.paper_parsing import PaperChunk, ParsedPaper


class DocumentVectorStore(Protocol):
    """The LangChain vector-store write operation used by the ingestion pipeline."""

    def add_documents(self, documents: list[Document], *, ids: list[str]) -> list[str]: ...


@dataclass(frozen=True)
class IndexingResult:
    paper_id: str
    requested_chunk_count: int
    upserted_chunk_ids: list[str]


class PaperIndexer:
    """Write parsed chunks to Chroma in DashScope-compatible batches of at most ten."""

    def __init__(self, vector_store: DocumentVectorStore, *, batch_size: int = 10) -> None:
        if not 1 <= batch_size <= 10:
            raise ValueError("batch_size must be between 1 and 10 for text-embedding-v4.")
        self._vector_store = vector_store
        self._batch_size = batch_size

    def index(self, parsed: ParsedPaper) -> IndexingResult:
        upserted_ids: list[str] = []
        for batch in _batches(parsed.chunks, self._batch_size):
            documents = [_chunk_to_document(chunk) for chunk in batch]
            ids = [chunk.chunk_id for chunk in batch]
            returned_ids = self._vector_store.add_documents(documents, ids=ids)
            if returned_ids != ids:
                raise RuntimeError("Chroma returned IDs different from the requested deterministic chunk IDs.")
            upserted_ids.extend(returned_ids)
        return IndexingResult(
            paper_id=parsed.paper_id,
            requested_chunk_count=len(parsed.chunks),
            upserted_chunk_ids=upserted_ids,
        )


def _chunk_to_document(chunk: PaperChunk) -> Document:
    metadata: dict[str, str | int] = {
        "paper_id": chunk.paper_id,
        "chunk_id": chunk.chunk_id,
        "title": chunk.title,
        "page_number": chunk.page_number,
        "chunk_index": chunk.chunk_index,
        "parser_version": chunk.parser_version,
        "chunker_version": chunk.chunker_version,
        "source_type": chunk.source_type,
        "content_type": chunk.content_type,
        "source_text": chunk.text,
    }
    if chunk.section:
        metadata["section"] = chunk.section
    if chunk.arxiv_id:
        metadata["arxiv_id"] = chunk.arxiv_id
    return Document(page_content=chunk.embedding_text, metadata=metadata, id=chunk.chunk_id)


def _batches(items: Sequence[PaperChunk], batch_size: int) -> list[Sequence[PaperChunk]]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]
