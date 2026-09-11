"""Deterministic local-RAG query construction and evidence selection."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from langchain_core.documents import Document

from app.models.schemas import EvidenceItem, QueryIntent


class SimilaritySearchStore(Protocol):
    """The scored LangChain vector-store operation needed by local retrieval."""

    def similarity_search_with_relevance_scores(
        self, query: str, *, k: int, score_threshold: float
    ) -> list[tuple[Document, float]]: ...


class RetrievalDataError(ValueError):
    """Raised when an indexed chunk lacks provenance required for a citation."""


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    evidence: list[EvidenceItem]
    candidate_count: int
    unique_paper_count: int


def build_retrieval_query(*, user_query: str, intent: QueryIntent) -> str:
    """Build a transparent embedding query without a second LLM call."""
    parts = [user_query.strip(), f"主题: {intent.topic.strip()}"]
    concepts = [concept.strip() for concept in intent.key_concepts if concept.strip()]
    if concepts:
        parts.append(f"关键词: {', '.join(dict.fromkeys(concepts))}")
    if intent.referenced_paper_ids:
        parts.append(f"已引用论文: {', '.join(intent.referenced_paper_ids)}")
    return "\n".join(part for part in parts if part)


class LocalEvidenceRetriever:
    """Retrieve scored chunks, retaining diversity and citation provenance."""

    def __init__(
        self,
        vector_store: SimilaritySearchStore,
        *,
        candidate_k: int = 12,
        score_threshold: float = 0.35,
        max_chunks_per_paper: int = 2,
        excerpt_max_characters: int = 1_200,
    ) -> None:
        self._vector_store = vector_store
        self._candidate_k = candidate_k
        self._score_threshold = score_threshold
        self._max_chunks_per_paper = max_chunks_per_paper
        self._excerpt_max_characters = excerpt_max_characters

    async def retrieve(self, query: str) -> RetrievalResult:
        candidates = await asyncio.to_thread(
            self._vector_store.similarity_search_with_relevance_scores,
            query,
            k=self._candidate_k,
            score_threshold=self._score_threshold,
        )
        evidence = self._select_evidence(candidates)
        return RetrievalResult(
            query=query,
            evidence=evidence,
            candidate_count=len(candidates),
            unique_paper_count=len({item.paper_id for item in evidence}),
        )

    def _select_evidence(self, candidates: Sequence[tuple[Document, float]]) -> list[EvidenceItem]:
        per_paper_count: defaultdict[str, int] = defaultdict(int)
        selected_chunk_ids: set[str] = set()
        evidence: list[EvidenceItem] = []
        for document, score in candidates:
            if score < self._score_threshold:
                continue
            item = self._to_evidence(document, score)
            if item.chunk_id in selected_chunk_ids:
                continue
            if per_paper_count[item.paper_id] >= self._max_chunks_per_paper:
                continue
            selected_chunk_ids.add(item.chunk_id)
            per_paper_count[item.paper_id] += 1
            evidence.append(item)
        return evidence

    def _to_evidence(self, document: Document, score: float) -> EvidenceItem:
        metadata = document.metadata
        paper_id = metadata.get("paper_id")
        chunk_id = metadata.get("chunk_id")
        title = metadata.get("title")
        if (
            not isinstance(paper_id, str)
            or not paper_id
            or not isinstance(chunk_id, str)
            or not chunk_id
            or not isinstance(title, str)
            or not title
        ):
            raise RetrievalDataError("Indexed chunks require string paper_id, chunk_id, and title metadata.")

        page_number = metadata.get("page_number")
        if page_number is not None and not isinstance(page_number, int):
            raise RetrievalDataError("Indexed chunk page_number must be an integer when present.")
        section = metadata.get("section")
        if section is not None and not isinstance(section, str):
            raise RetrievalDataError("Indexed chunk section must be a string when present.")
        return EvidenceItem(
            chunk_id=chunk_id,
            paper_id=paper_id,
            title=title,
            page_number=page_number,
            section=section,
            excerpt=_source_excerpt(document)[: self._excerpt_max_characters],
            score=score,
        )


def _source_excerpt(document: Document) -> str:
    """Prefer the raw source preserved during contextual embedding over its prefixed text."""
    source_text = document.metadata.get("source_text")
    return source_text if isinstance(source_text, str) else document.page_content
