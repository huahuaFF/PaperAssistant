from typing import Any, cast

import pytest
from langchain_core.runnables import Runnable, RunnableLambda

from app.agent.arxiv_query_context import ArxivQueryContextBuilder
from app.graph.nodes.build_arxiv_query import create_build_arxiv_query_node
from app.models.schemas import ArxivSearchPlan


def approved_search_state() -> dict[str, object]:
    return {
        "run_id": "run-1",
        "conversation_id": "conversation-1",
        "workspace_id": "default",
        "user_query": "请列出流匹配的代表性论文。",
        "status": "search_approved",
        "search_approval": "approved",
        "intent": {
            "schema_version": "query_intent_v1",
            "route": "research",
            "task_type": "literature_discovery",
            "topic": "Flow Matching",
            "key_concepts": ["flow matching", "generative modeling"],
            "requires_literature_evidence": True,
            "target_arxiv_ids": [],
            "target_dois": [],
            "target_urls": [],
            "referenced_paper_ids": [],
            "output_language": "zh",
            "clarification_question": None,
        },
        "coverage": {
            "sufficient": False,
            "confidence": 0.95,
            "reason": "本地证据无法覆盖代表性论文。",
            "missing_aspects": ["缺少多篇代表论文"],
            "recommended_action": "ask_arxiv_permission",
        },
    }


@pytest.mark.asyncio
async def test_build_arxiv_query_returns_typed_plan_after_approval() -> None:
    plan = ArxivSearchPlan(
        query='all:"flow matching" AND (cat:cs.LG OR cat:stat.ML)',
        rationale="检索生成建模中的流匹配代表作。",
        categories=["cs.LG", "stat.ML"],
        max_results=12,
    )
    agent: Runnable[dict[str, object], dict[str, object]] = RunnableLambda(
        lambda _: {"structured_response": plan}
    )
    node = create_build_arxiv_query_node(agent)

    update = await node(cast(Any, approved_search_state()))

    assert update["arxiv_query"] == plan.query
    assert update["arxiv_search_plan"] == plan.model_dump()
    assert update["status"] == "arxiv_query_planned"


@pytest.mark.asyncio
async def test_build_arxiv_query_rejects_unapproved_search() -> None:
    agent = cast(Runnable[dict[str, object], dict[str, object]], RunnableLambda(lambda _: {}))
    node = create_build_arxiv_query_node(agent)
    state = approved_search_state()
    state["search_approval"] = "rejected"

    with pytest.raises(PermissionError, match="approved"):
        await node(cast(Any, state))


def test_arxiv_query_context_rejects_sufficient_local_coverage() -> None:
    state = approved_search_state()
    state["coverage"] = {
        "sufficient": True,
        "confidence": 0.9,
        "reason": "本地证据已经充分。",
        "missing_aspects": [],
        "recommended_action": "answer",
    }

    with pytest.raises(ValueError, match="only valid"):
        ArxivQueryContextBuilder().build(cast(Any, state))
