"""Deterministic local-RAG query construction and evidence selection."""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from langchain_core.documents import Document

from app.models.schemas import EvidenceItem, QueryIntent


class SimilaritySearchStore(Protocol):
    """The scored LangChain vector-store operation needed by local retrieval."""

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        *,
        k: int,
        score_threshold: float | None,
        filter: dict[str, object] | None = None,
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
    """Return the stable first-pass query used for local evidence retrieval.

    ``QueryIntent`` is deliberately not used to expand the embedding query:
    structured LLM classification can produce equivalent, but non-identical,
    concepts on separate runs. Routing may be probabilistic; the initial RAG
    lookup must not be.  The parameter remains part of the public contract so
    callers still validate that classification completed before retrieval.
    """
    del intent
    return user_query.strip()


_EXACT_RETRIEVAL_ANCHOR = re.compile(
    r"(?<![A-Za-z0-9])(?=[A-Za-z0-9_-]*[A-Z0-9])[A-Za-z][A-Za-z0-9_-]{2,}(?![A-Za-z0-9])"
)


def extract_retrieval_anchors(user_query: str) -> list[str]:
    """Extract explicit model/paper identifiers for a guarded recall fallback."""
    return list(dict.fromkeys(match.group(0) for match in _EXACT_RETRIEVAL_ANCHOR.finditer(user_query)))


class LocalEvidenceRetriever:
    """Retrieve scored chunks, retaining diversity and citation provenance."""

    def __init__(
        self,
        vector_store: SimilaritySearchStore,
        *,
        candidate_k: int = 12,
        max_chunks_per_paper: int = 2,
        excerpt_max_characters: int = 1_200,
    ) -> None:
        self._vector_store = vector_store
        self._candidate_k = candidate_k
        self._max_chunks_per_paper = max_chunks_per_paper
        self._excerpt_max_characters = excerpt_max_characters

    async def retrieve(
        self,
        query: str,
        *,
        metadata_filter: dict[str, object] | None = None,
        exact_anchors: Sequence[str] = (),
    ) -> RetrievalResult:
        if metadata_filter is not None:
            candidates = await asyncio.to_thread(
                self._vector_store.similarity_search_with_relevance_scores,
                query,
                k=self._candidate_k,
                score_threshold=None,
                filter=metadata_filter,
            )
        else:
            candidates = await asyncio.to_thread(
                self._vector_store.similarity_search_with_relevance_scores,
                query,
                k=self._candidate_k,
                score_threshold=None,
            )
        evidence = self._select_evidence(candidates, exact_anchors=exact_anchors)
        return RetrievalResult(
            query=query,
            evidence=evidence,
            candidate_count=len(candidates),
            unique_paper_count=len({item.paper_id for item in evidence}),
        )

    def _select_evidence(
        self, candidates: Sequence[tuple[Document, float]], *, exact_anchors: Sequence[str] = ()
    ) -> list[EvidenceItem]:
        # Vector scores are model- and collection-dependent.  They determine
        # rank only; a fixed value must not silently decide whether the user
        # sees a local answer or an arXiv-approval branch.
        ranked_candidates = sorted(
            candidates,
            key=lambda candidate: (
                self._document_matches_exact_anchor(candidate[0], exact_anchors),
                candidate[1],
            ),
            reverse=True,
        )
        per_paper_count: defaultdict[str, int] = defaultdict(int)
        selected_chunk_ids: set[str] = set()
        evidence: list[EvidenceItem] = []
        for document, score in ranked_candidates:
            item = self._to_evidence(document, score)
            if item.chunk_id in selected_chunk_ids:
                continue
            if per_paper_count[item.paper_id] >= self._max_chunks_per_paper:
                continue
            selected_chunk_ids.add(item.chunk_id)
            per_paper_count[item.paper_id] += 1
            evidence.append(item)
        return evidence

    def _document_matches_exact_anchor(
        self, document: Document, exact_anchors: Sequence[str]
    ) -> bool:
        """Prioritize documents that explicitly name a queried model or paper."""
        if not exact_anchors:
            return False
        title = document.metadata.get("title")
        searchable_text = "\n".join(
            value for value in (title, _source_excerpt(document)) if isinstance(value, str)
        ).casefold()
        return any(anchor.casefold() in searchable_text for anchor in exact_anchors)

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
