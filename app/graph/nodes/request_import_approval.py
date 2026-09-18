"""Human approval node for selecting papers before download and ingestion."""

from __future__ import annotations

from langgraph.types import interrupt

from app.graph.nodes.contracts import NodeUpdate
from app.graph.state import ResearchState
from app.models.schemas import (
    ArxivCandidate,
    ImportApprovalCandidate,
    ImportApprovalRequest,
    ImportApprovalResume,
    QueryIntent,
)


def request_import_approval(state: ResearchState) -> NodeUpdate:
    """Pause before import and accept only IDs included in the approval payload.

    This node intentionally has no side effects before ``interrupt``: LangGraph
    restarts a node from its beginning after a durable resume.
    """
    request = _build_approval_request(state)
    response = ImportApprovalResume.model_validate(interrupt(request.model_dump(mode="json")))
    selected_ids = _validated_selection(response, request)
    offered_candidates = [candidate.model_dump(mode="json") for candidate in request.candidates]
    if response.decision == "skip":
        return {
            "import_approval": "skipped",
            "import_candidates": offered_candidates,
            "selected_paper_ids": [],
            "status": "import_skipped",
        }
    return {
        "import_approval": "selected",
        "import_candidates": offered_candidates,
        "selected_paper_ids": selected_ids,
        "status": "import_selected",
    }


def _build_approval_request(state: ResearchState) -> ImportApprovalRequest:
    candidates_data = state.get("arxiv_candidates")
    if candidates_data is not None:
        if not isinstance(candidates_data, list):
            raise TypeError("arxiv_candidates must be a list.")
        candidates = [
            _from_arxiv_candidate(ArxivCandidate.model_validate(candidate))
            for candidate in candidates_data
        ]
        return ImportApprovalRequest(
            source="arxiv_search", query=state["user_query"], candidates=candidates
        )

    intent_data = state.get("intent")
    if not isinstance(intent_data, dict):
        raise TypeError("request_import_approval requires arxiv_candidates or an import intent.")
    intent = QueryIntent.model_validate(intent_data)
    if intent.route != "direct_import":
        raise ValueError("request_import_approval requires arxiv search results or direct_import intent.")
    return ImportApprovalRequest(
        source="direct_import",
        query=state["user_query"],
        candidates=_direct_import_candidates(intent),
    )


def _validated_selection(
    response: ImportApprovalResume, request: ImportApprovalRequest
) -> list[str]:
    if response.decision == "skip":
        return []
    available_ids = {candidate.paper_id for candidate in request.candidates}
    selected_ids = list(dict.fromkeys(response.selected_paper_ids))
    unknown_ids = set(selected_ids) - available_ids
    if unknown_ids:
        unknown_display = ", ".join(sorted(unknown_ids))
        raise ValueError(f"Selected paper IDs were not offered for approval: {unknown_display}")
    return selected_ids


def _from_arxiv_candidate(candidate: ArxivCandidate) -> ImportApprovalCandidate:
    return ImportApprovalCandidate(
        paper_id=f"arxiv:{candidate.arxiv_id}",
        source_type="arxiv",
        source_url=candidate.pdf_url,
        title=candidate.title,
        authors=candidate.authors,
        abstract=candidate.abstract,
        categories=candidate.categories,
    )


def _direct_import_candidates(intent: QueryIntent) -> list[ImportApprovalCandidate]:
    candidates: list[ImportApprovalCandidate] = []
    for arxiv_id in dict.fromkeys(identifier.strip() for identifier in intent.target_arxiv_ids if identifier.strip()):
        candidates.append(
            ImportApprovalCandidate(
                paper_id=f"arxiv:{arxiv_id}",
                source_type="arxiv",
                source_url=f"https://arxiv.org/abs/{arxiv_id}",
            )
        )
    for doi in dict.fromkeys(identifier.strip() for identifier in intent.target_dois if identifier.strip()):
        candidates.append(
            ImportApprovalCandidate(
                paper_id=f"doi:{doi}", source_type="doi", source_url=f"https://doi.org/{doi}"
            )
        )
    for url in dict.fromkeys(value.strip() for value in intent.target_urls if value.strip()):
        candidates.append(ImportApprovalCandidate(paper_id=f"url:{url}", source_type="url", source_url=url))
    return candidates
