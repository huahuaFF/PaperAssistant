"""LangGraph node that ingests the papers explicitly selected by a human."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from app.graph.nodes.contracts import NodeUpdate
from app.graph.nodes.request_import_approval import _build_approval_request
from app.graph.state import ResearchState
from app.services.paper_ingestion import PaperIngestionCoordinator


def create_ingest_papers_node(
    coordinator: PaperIngestionCoordinator,
) -> Callable[[ResearchState], Awaitable[NodeUpdate]]:
    """Create the post-approval ingestion node without a model or an Agent."""

    async def ingest_papers(state: ResearchState) -> NodeUpdate:
        if state.get("import_approval") != "selected":
            raise PermissionError("ingest_papers requires selected import approval.")
        selected_ids = state.get("selected_paper_ids")
        if not isinstance(selected_ids, list) or not all(isinstance(item, str) for item in selected_ids):
            raise TypeError("ingest_papers requires selected_paper_ids from request_import_approval.")
        offered = {candidate.paper_id: candidate for candidate in _build_approval_request(state).candidates}
        unknown_ids = set(selected_ids) - set(offered)
        if unknown_ids:
            raise ValueError("Selected paper IDs are not present in the approved candidates.")

        outcomes = []
        for paper_id in dict.fromkeys(selected_ids):
            outcomes.append(await coordinator.ingest(run_id=state["run_id"], candidate=offered[paper_id]))
        return {
            "imported_paper_ids": [outcome.paper_id for outcome in outcomes],
            "ingestion_results": [outcome.to_state() for outcome in outcomes],
            "status": "papers_ingested",
        }

    return ingest_papers
