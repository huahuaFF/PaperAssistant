"""Retrieve final-report evidence after approved papers have entered Chroma."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence

from app.graph.nodes.contracts import NodeUpdate
from app.graph.state import ResearchState
from app.models.schemas import EvidenceItem, QueryIntent
from app.services.local_retrieval import (
    LocalEvidenceRetriever,
    build_retrieval_query,
    extract_retrieval_anchors,
)

MAX_FINAL_EVIDENCE = 10


def create_retrieve_augmented_library_node(
    retriever: LocalEvidenceRetriever,
) -> Callable[[ResearchState], Awaitable[NodeUpdate]]:
    """Create a no-LLM final-evidence node that prioritizes newly imported papers."""

    async def retrieve_augmented_library(state: ResearchState) -> NodeUpdate:
        imported_paper_ids = _validated_imported_paper_ids(state)
        intent_data = state.get("intent")
        if not isinstance(intent_data, dict):
            raise TypeError("retrieve_augmented_library requires intent from classify_query.")
        intent = QueryIntent.model_validate(intent_data)
        query = build_retrieval_query(user_query=state["user_query"], intent=intent)
        exact_anchors = extract_retrieval_anchors(state["user_query"])

        imported_result = await retriever.retrieve(
            query,
            metadata_filter={"paper_id": {"$in": imported_paper_ids}},
            exact_anchors=exact_anchors,
        )
        library_result = await retriever.retrieve(query, exact_anchors=exact_anchors)
        evidence = _merge_evidence(imported_result.evidence, library_result.evidence)
        serialized_evidence = [item.model_dump() for item in evidence]
        imported_evidence_ids = [
            item.chunk_id for item in evidence if item.paper_id in set(imported_paper_ids)
        ]
        return {
            "final_evidence": serialized_evidence,
            "final_evidence_ids": [item["chunk_id"] for item in serialized_evidence],
            "final_retrieval_query": query,
            "new_paper_evidence_ids": imported_evidence_ids,
            "augmented_retrieval_stats": {
                "imported_paper_count": len(imported_paper_ids),
                "imported_candidate_count": imported_result.candidate_count,
                "imported_selected_count": len(imported_result.evidence),
                "library_candidate_count": library_result.candidate_count,
                "library_selected_count": len(library_result.evidence),
                "final_selected_count": len(evidence),
                "new_paper_evidence_count": len(imported_evidence_ids),
            },
            "status": "augmented_library_retrieved",
        }

    return retrieve_augmented_library


def _validated_imported_paper_ids(state: ResearchState) -> list[str]:
    if state.get("import_approval") != "selected":
        raise PermissionError("retrieve_augmented_library requires selected import approval.")
    imported_paper_ids = state.get("imported_paper_ids")
    if not isinstance(imported_paper_ids, list) or not imported_paper_ids:
        raise TypeError("retrieve_augmented_library requires imported_paper_ids from ingest_papers.")
    if not all(isinstance(paper_id, str) and paper_id for paper_id in imported_paper_ids):
        raise TypeError("imported_paper_ids must contain non-empty strings.")
    return list(dict.fromkeys(imported_paper_ids))


def _merge_evidence(
    imported_evidence: Sequence[EvidenceItem], library_evidence: Sequence[EvidenceItem]
) -> list[EvidenceItem]:
    """Keep imported evidence first, then add global evidence without duplicate chunks."""
    merged: list[EvidenceItem] = []
    seen_chunk_ids: set[str] = set()
    for item in [*imported_evidence, *library_evidence]:
        if item.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(item.chunk_id)
        merged.append(item)
        if len(merged) == MAX_FINAL_EVIDENCE:
            break
    return merged
