from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterator, cast

import arxiv
import pytest
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool

from app.graph.nodes.search_arxiv import create_search_arxiv_node
from app.models.schemas import ArxivCandidate, ArxivSearchPlan
from app.services.arxiv_search import ArxivSearchClient, build_arxiv_library_query, create_search_arxiv_tool


class FakeArxivLibraryClient:
    def __init__(self, results: list[arxiv.Result]) -> None:
        self._results = results
        self.searches: list[arxiv.Search] = []

    def results(self, search: arxiv.Search) -> Iterator[arxiv.Result]:
        self.searches.append(search)
        return iter(self._results)


def library_result() -> arxiv.Result:
    now = datetime(2022, 10, 6, 17, 58, tzinfo=timezone.utc)
    return arxiv.Result(
        entry_id="https://arxiv.org/abs/2210.02747v2",
        title=" Flow Matching for Generative Modeling ",
        authors=[arxiv.Result.Author("Ada Lovelace")],
        summary=" A normalized abstract. ",
        categories=["cs.LG", "stat.ML"],
        published=now,
        updated=now,
        links=[arxiv.Result.Link("https://arxiv.org/pdf/2210.02747v2", title="pdf")],
    )


def planned_state() -> dict[str, object]:
    plan = ArxivSearchPlan(
        keywords=["flow matching", "generative modeling"],
        rationale="测试。",
        categories=["cs.LG", "stat.ML"],
        max_results=5,
    )
    return {
        "run_id": "run-1",
        "conversation_id": "conversation-1",
        "workspace_id": "default",
        "user_query": "流匹配论文",
        "status": "arxiv_query_planned",
        "search_approval": "approved",
        "arxiv_search_plan": plan.model_dump(),
    }


def test_library_query_is_built_deterministically_from_semantic_keywords() -> None:
    plan = ArxivSearchPlan(
        keywords=["flow matching", "generative modeling", "flow matching"],
        rationale="测试。",
        categories=["cs.LG", "stat.ML"],
    )

    assert build_arxiv_library_query(plan) == (
        '(all:"flow matching" OR all:"generative modeling") AND (cat:cs.LG OR cat:stat.ML)'
    )


@pytest.mark.asyncio
async def test_search_arxiv_tool_uses_arxiv_library_and_normalizes_results() -> None:
    library_client = FakeArxivLibraryClient([library_result()])
    tool: BaseTool = create_search_arxiv_tool(ArxivSearchClient(client=library_client))

    result = await tool.ainvoke(
        {"keywords": ["flow matching"], "categories": ["cs.LG"], "max_results": 5}
    )

    assert isinstance(result, list)
    assert result[0]["arxiv_id"] == "2210.02747"
    assert result[0]["authors"] == ["Ada Lovelace"]
    assert result[0]["abstract"] == "A normalized abstract."
    assert result[0]["pdf_url"] == "https://arxiv.org/pdf/2210.02747v2"
    assert library_client.searches[0].query == '(all:"flow matching") AND (cat:cs.LG)'


def test_library_query_rejects_raw_api_syntax_in_agent_keywords() -> None:
    plan = ArxivSearchPlan(keywords=['all:"flow matching"'], rationale="测试。")

    with pytest.raises(ValueError, match="quotes"):
        build_arxiv_library_query(plan)


@pytest.mark.asyncio
async def test_search_arxiv_node_writes_serialized_candidates() -> None:
    candidate = ArxivCandidate.model_validate(
        {
            "arxiv_id": "2210.02747",
            "title": "Flow Matching for Generative Modeling",
            "authors": ["Ada Lovelace"],
            "abstract": "A normalized abstract.",
            "categories": ["cs.LG"],
            "published_at": "2022-10-06T17:58:00Z",
            "updated_at": "2022-10-06T17:58:00Z",
            "abs_url": "https://arxiv.org/abs/2210.02747v2",
            "pdf_url": "https://arxiv.org/pdf/2210.02747v2",
        }
    )
    tool: Any = RunnableLambda(lambda _: [candidate.model_dump(mode="json")])
    node = create_search_arxiv_node(cast(Any, tool))

    update = await node(cast(Any, planned_state()))

    assert update["status"] == "arxiv_searched"
    assert update["arxiv_search_stats"] == {"returned_count": 1, "requested_count": 5}
    assert update["arxiv_candidates"][0]["arxiv_id"] == "2210.02747"


@pytest.mark.asyncio
async def test_search_arxiv_node_rejects_unapproved_search() -> None:
    node = create_search_arxiv_node(cast(Any, RunnableLambda(lambda _: [])))
    state = planned_state()
    state["search_approval"] = "rejected"

    with pytest.raises(PermissionError, match="approved"):
        await node(cast(Any, state))
