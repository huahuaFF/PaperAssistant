"""Human-in-the-loop approval node for external arXiv search."""

from __future__ import annotations

from langgraph.types import interrupt

from app.graph.nodes.contracts import NodeUpdate
from app.graph.state import ResearchState
from app.models.schemas import (
    CoverageAssessment,
    QueryIntent,
    SearchApprovalRequest,
    SearchApprovalResume,
)


def request_search_approval(state: ResearchState) -> NodeUpdate:
    """Pause the graph until a human approves or rejects an arXiv search.

    This node intentionally has no side effects before ``interrupt`` because
    LangGraph restarts the node from its beginning during resume.
    """
    payload = _build_approval_request(state)
    response = SearchApprovalResume.model_validate(interrupt(payload.model_dump()))
    return {
        "search_approval": "approved" if response.decision == "approve" else "rejected",
        "status": "search_approved" if response.decision == "approve" else "search_rejected",
    }


def _build_approval_request(state: ResearchState) -> SearchApprovalRequest:
    intent_data = state.get("intent")
    coverage_data = state.get("coverage")
    if not isinstance(intent_data, dict):
        raise TypeError("request_search_approval requires intent from classify_query.")
    if not isinstance(coverage_data, dict):
        raise TypeError("request_search_approval requires coverage from assess_coverage.")
    intent = QueryIntent.model_validate(intent_data)
    coverage = CoverageAssessment.model_validate(coverage_data)
    if coverage.sufficient:
        raise ValueError("Search approval cannot be requested when local coverage is sufficient.")

    evidence = state.get("local_evidence", [])
    if not isinstance(evidence, list):
        raise TypeError("local_evidence must be a list.")
    return SearchApprovalRequest(
        query=state["user_query"],
        topic=intent.topic,
        coverage_reason=coverage.reason,
        missing_aspects=coverage.missing_aspects,
        local_evidence_count=len(evidence),
    )
