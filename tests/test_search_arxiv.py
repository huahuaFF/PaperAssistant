from __future__ import annotations

from typing import Any, cast

import httpx
import pytest
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import BaseTool

from app.graph.nodes.search_arxiv import create_search_arxiv_node
from app.models.schemas import ArxivCandidate, ArxivSearchPlan
from app.services.arxiv_search import ArxivSearchClient, create_search_arxiv_tool

ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>1</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2210.02747v2</id>
    <updated>2022-10-06T17:58:00Z</updated>
    <published>2022-10-06T17:58:00Z</published>
    <title>Flow Matching for Generative Modeling</title>
    <summary>  A  normalized abstract. </summary>
    <author><name>Ada Lovelace</name></author>
    <category term="cs.LG" />
    <category term="stat.ML" />
    <link title="pdf" href="https://arxiv.org/pdf/2210.02747v2" />
  </entry>
</feed>"""


def planned_state() -> dict[str, object]:
    plan = ArxivSearchPlan(
        query='all:"flow matching"',
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


def mock_client() -> ArxivSearchClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["search_query"] == '(all:"flow matching") AND (cat:cs.LG OR cat:stat.ML)'
        assert request.url.params["max_results"] == "5"
        return httpx.Response(200, text=ARXIV_FEED)

    return ArxivSearchClient(transport=httpx.MockTransport(handler), retry_attempts=0)


@pytest.mark.asyncio
async def test_search_arxiv_tool_calls_official_api_and_normalizes_atom_feed() -> None:
    tool: BaseTool = create_search_arxiv_tool(mock_client())

    result = await tool.ainvoke(
        {"query": 'all:"flow matching"', "categories": ["cs.LG", "stat.ML"], "max_results": 5}
    )

    assert isinstance(result, list)
    assert result[0]["arxiv_id"] == "2210.02747"
    assert result[0]["authors"] == ["Ada Lovelace"]
    assert result[0]["abstract"] == "A normalized abstract."
    assert result[0]["pdf_url"] == "https://arxiv.org/pdf/2210.02747v2"


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
