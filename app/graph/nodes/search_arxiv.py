"""LangGraph node that executes an already-approved arXiv search tool."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.runnables import Runnable

from app.graph.nodes.contracts import NodeUpdate
from app.graph.state import ResearchState
from app.models.schemas import ArxivCandidate, ArxivSearchPlan


def create_search_arxiv_node(
    search_tool: Runnable[dict[str, object], object],
) -> Callable[[ResearchState], Awaitable[NodeUpdate]]:
    """Create a deterministic tool node guarded by the prior human approval."""

    async def search_arxiv(state: ResearchState) -> NodeUpdate:
        if state.get("search_approval") != "approved":
            raise PermissionError("search_arxiv requires an approved search_approval.")
        plan_data = state.get("arxiv_search_plan")
        if not isinstance(plan_data, dict):
            raise TypeError("search_arxiv requires arxiv_search_plan from build_arxiv_query.")
        plan = ArxivSearchPlan.model_validate(plan_data)
        tool_result = await search_tool.ainvoke(
            {"keywords": plan.keywords, "categories": plan.categories, "max_results": plan.max_results}
        )
        if not isinstance(tool_result, list):
            raise TypeError("search_arxiv tool must return a list of candidate dictionaries.")
        candidates = [ArxivCandidate.model_validate(item) for item in tool_result]
        serialized = [candidate.model_dump(mode="json") for candidate in candidates]
        return {
            "arxiv_candidates": serialized,
            "arxiv_search_stats": {"returned_count": len(serialized), "requested_count": plan.max_results},
            "status": "arxiv_searched",
        }

    return search_arxiv
