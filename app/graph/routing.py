"""Deterministic branch rules for the research workflow.

These rules are code, not LLM decisions. Agent nodes may populate structured
state, but they cannot bypass human approval or run ingestion directly.
"""

from __future__ import annotations

from typing import Literal, cast

from app.graph.state import ResearchState

RouteAfterCoverage = Literal["generate_report", "request_search_approval"]
RouteAfterSearchApproval = Literal["build_arxiv_query", "explain_knowledge_gap"]
RouteAfterImportApproval = Literal["ingest_papers", "report_candidates"]
RouteAfterClassification = Literal[
    "retrieve_local", "request_import_approval", "request_clarification", "explain_out_of_scope"
]


class WorkflowRoutingError(ValueError):
    """Raised when a node fails to provide state required by the next edge."""


def route_after_classification(state: ResearchState) -> RouteAfterClassification:
    intent = state.get("intent")
    if not isinstance(intent, dict):
        raise WorkflowRoutingError("classify_query must set intent.")

    route = intent.get("route")
    if route == "research":
        return "retrieve_local"
    if route == "direct_import":
        return "request_import_approval"
    if route == "needs_clarification":
        return "request_clarification"
    if route == "out_of_scope":
        return "explain_out_of_scope"
    raise WorkflowRoutingError("classify_query returned an unsupported intent.route.")


def route_after_coverage(state: ResearchState) -> RouteAfterCoverage:
    coverage = state.get("coverage")
    if not isinstance(coverage, dict) or not isinstance(coverage.get("sufficient"), bool):
        raise WorkflowRoutingError("assess_coverage must set coverage.sufficient.")
    return "generate_report" if coverage["sufficient"] else "request_search_approval"


def route_after_search_approval(state: ResearchState) -> RouteAfterSearchApproval:
    decision = state.get("search_approval")
    if decision == "approved":
        return "build_arxiv_query"
    if decision == "rejected":
        return "explain_knowledge_gap"
    raise WorkflowRoutingError("request_search_approval must set search_approval.")


def route_after_import_approval(state: ResearchState) -> RouteAfterImportApproval:
    decision = state.get("import_approval")
    if decision == "selected":
        selected_paper_ids = cast(list[str], state.get("selected_paper_ids", []))
        if not selected_paper_ids:
            raise WorkflowRoutingError("Selected import requires at least one paper ID.")
        return "ingest_papers"
    if decision == "skipped":
        return "report_candidates"
    raise WorkflowRoutingError("request_import_approval must set import_approval.")
