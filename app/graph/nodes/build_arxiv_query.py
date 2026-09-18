"""LangGraph node that plans an approved arXiv search without executing it."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.runnables import Runnable

from app.agent.arxiv_query_context import ArxivQueryContextBuilder
from app.graph.nodes.contracts import NodeUpdate
from app.graph.state import ResearchState
from app.models.schemas import ArxivSearchPlan
from app.prompts.build_arxiv_query import ARXIV_QUERY_PROMPT, ARXIV_QUERY_PROMPT_VERSION


def create_build_arxiv_query_node(
    agent: Runnable[dict[str, object], dict[str, object]],
    context_builder: ArxivQueryContextBuilder | None = None,
) -> Callable[[ResearchState], Awaitable[NodeUpdate]]:
    """Create an Agent-backed search-plan node guarded by explicit approval."""
    builder = context_builder or ArxivQueryContextBuilder()

    async def build_arxiv_query(state: ResearchState) -> NodeUpdate:
        if state.get("search_approval") != "approved":
            raise PermissionError("build_arxiv_query requires an approved search_approval.")
        context = builder.build(state)
        prompt_value = ARXIV_QUERY_PROMPT.invoke(context.prompt_values())
        agent_result = await agent.ainvoke({"messages": prompt_value.to_messages()})
        plan = agent_result.get("structured_response")
        if not isinstance(plan, ArxivSearchPlan):
            raise TypeError("arxiv_query_planner did not return an ArxivSearchPlan structured response.")
        return {
            "arxiv_query": ", ".join(plan.keywords),
            "arxiv_search_plan": plan.model_dump(),
            "status": "arxiv_query_planned",
            "prompt_version": ARXIV_QUERY_PROMPT_VERSION,
        }

    return build_arxiv_query
