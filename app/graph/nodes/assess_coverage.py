"""LangGraph node that assesses whether local RAG evidence is sufficient."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.runnables import Runnable

from app.agent.coverage_context import CoverageContextBuilder
from app.graph.nodes.contracts import NodeUpdate
from app.graph.state import ResearchState
from app.models.schemas import CoverageAssessment
from app.prompts.assess_coverage import COVERAGE_PROMPT, COVERAGE_PROMPT_VERSION


def create_assess_coverage_node(
    agent: Runnable[dict[str, object], dict[str, object]],
    context_builder: CoverageContextBuilder | None = None,
) -> Callable[[ResearchState], Awaitable[NodeUpdate]]:
    """Create an Agent-backed coverage node with a deterministic empty-evidence guard."""
    builder = context_builder or CoverageContextBuilder()

    async def assess_coverage(state: ResearchState) -> NodeUpdate:
        raw_evidence = state.get("local_evidence", [])
        if raw_evidence == []:
            return _empty_evidence_update()
        context = builder.build(state)
        prompt_value = COVERAGE_PROMPT.invoke(context.prompt_values())
        agent_result = await agent.ainvoke({"messages": prompt_value.to_messages()})
        assessment = agent_result.get("structured_response")
        if not isinstance(assessment, CoverageAssessment):
            raise TypeError("coverage_assessor did not return a CoverageAssessment structured response.")
        return {
            "coverage": assessment.model_dump(),
            "status": "coverage_assessed",
            "prompt_version": COVERAGE_PROMPT_VERSION,
        }

    return assess_coverage


def _empty_evidence_update() -> NodeUpdate:
    """Avoid an LLM call when retrieval produced no usable local evidence."""
    assessment = CoverageAssessment(
        sufficient=False,
        confidence=1.0,
        reason="本地文献库没有返回可引用的证据。",
        missing_aspects=["缺少本地检索证据"],
        recommended_action="ask_arxiv_permission",
    )
    return {
        "coverage": assessment.model_dump(),
        "status": "coverage_assessed",
        "prompt_version": COVERAGE_PROMPT_VERSION,
    }
