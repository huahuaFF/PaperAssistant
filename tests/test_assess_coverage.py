from typing import cast

import pytest
from langchain_core.runnables import Runnable, RunnableLambda

from app.agent.coverage_context import CoverageContextBuilder
from app.graph.nodes.assess_coverage import create_assess_coverage_node
from app.models.schemas import CoverageAssessment, QueryIntent


def research_intent() -> QueryIntent:
    return QueryIntent(
        route="research",
        task_type="literature_question",
        topic="FlowCF",
        key_concepts=["behavior-guided prior"],
        requires_literature_evidence=True,
    )


@pytest.mark.asyncio
async def test_empty_evidence_is_insufficient_without_calling_the_agent() -> None:
    async def should_not_run(_: dict[str, object]) -> dict[str, object]:
        raise AssertionError("Agent must not run when local evidence is empty.")

    agent = cast(Runnable[dict[str, object], dict[str, object]], RunnableLambda(should_not_run))
    node = create_assess_coverage_node(agent)
    update = await node(
        {
            "run_id": "run-1",
            "conversation_id": "conversation-1",
            "workspace_id": "default",
            "user_query": "FlowCF 是什么？",
            "status": "local_retrieved",
            "intent": research_intent().model_dump(),
            "local_evidence": [],
        }
    )

    assert update["coverage"]["sufficient"] is False
    assert update["coverage"]["recommended_action"] == "ask_arxiv_permission"


@pytest.mark.asyncio
async def test_coverage_node_returns_typed_agent_assessment() -> None:
    assessment = CoverageAssessment(
        sufficient=True,
        confidence=0.82,
        reason="两段证据直接说明了该方法的设计与效果。",
        recommended_action="answer",
    )
    agent: Runnable[dict[str, object], dict[str, object]] = RunnableLambda(
        lambda _: {"structured_response": assessment}
    )
    node = create_assess_coverage_node(agent)
    update = await node(
        {
            "run_id": "run-1",
            "conversation_id": "conversation-1",
            "workspace_id": "default",
            "user_query": "behavior-guided prior 有什么作用？",
            "status": "local_retrieved",
            "intent": research_intent().model_dump(),
            "local_evidence": [
                {
                    "chunk_id": "paper-1:chunk-1",
                    "paper_id": "paper-1",
                    "title": "FlowCF",
                    "page_number": 4,
                    "section": "Behavior-Guided Prior",
                    "excerpt": "The prior incorporates user behavior information.",
                    "score": 0.91,
                }
            ],
        }
    )

    assert update["coverage"] == assessment.model_dump()
    assert update["status"] == "coverage_assessed"


def test_coverage_context_bounds_evidence_excerpts() -> None:
    context = CoverageContextBuilder().build(
        {
            "run_id": "run-1",
            "conversation_id": "conversation-1",
            "workspace_id": "default",
            "user_query": "FlowCF 是什么？",
            "status": "local_retrieved",
            "intent": research_intent().model_dump(),
            "local_evidence": [
                {
                    "chunk_id": "paper-1:chunk-1",
                    "paper_id": "paper-1",
                    "title": "FlowCF",
                    "excerpt": "x" * 2_000,
                }
            ],
        }
    )

    assert len(context.evidence_json) < 1_500
