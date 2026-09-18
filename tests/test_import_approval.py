from __future__ import annotations

from typing import Any, cast

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.graph.nodes.request_import_approval import _build_approval_request, request_import_approval
from app.graph.state import ResearchState


def searched_papers_state() -> ResearchState:
    return {
        "run_id": "run-1",
        "conversation_id": "conversation-1",
        "workspace_id": "default",
        "user_query": "请找流匹配论文。",
        "status": "arxiv_searched",
        "search_approval": "approved",
        "arxiv_candidates": [
            {
                "arxiv_id": "2210.02747",
                "title": "Flow Matching for Generative Modeling",
                "authors": ["Ada Lovelace"],
                "abstract": "A flow matching paper.",
                "categories": ["cs.LG"],
                "published_at": "2022-10-06T17:58:00Z",
                "updated_at": "2022-10-06T17:58:00Z",
                "abs_url": "https://arxiv.org/abs/2210.02747v2",
                "pdf_url": "https://arxiv.org/pdf/2210.02747v2",
            }
        ],
    }


@pytest.mark.asyncio
async def test_import_approval_interrupts_and_resumes_selected_papers(tmp_path: Any) -> None:
    checkpoint_path = tmp_path / "checkpoints.db"
    builder = StateGraph(ResearchState)
    builder.add_node("request_import_approval", request_import_approval)
    builder.add_edge(START, "request_import_approval")
    builder.add_edge("request_import_approval", END)
    config = {"configurable": {"thread_id": "import-approval-thread-1"}}

    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        await checkpointer.setup()
        graph = builder.compile(checkpointer=checkpointer)
        interrupted = await graph.ainvoke(searched_papers_state(), config=cast(Any, config))
        payload = interrupted["__interrupt__"][0].value
        assert payload["approval_type"] == "import_papers"
        assert payload["source"] == "arxiv_search"
        assert payload["candidates"][0]["paper_id"] == "arxiv:2210.02747"
        assert payload["candidates"][0]["title"] == "Flow Matching for Generative Modeling"

    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as reopened_checkpointer:
        await reopened_checkpointer.setup()
        reopened_graph = builder.compile(checkpointer=reopened_checkpointer)
        resumed = await reopened_graph.ainvoke(
            Command(resume={"decision": "select", "selected_paper_ids": ["arxiv:2210.02747"]}),
            config=cast(Any, config),
        )

    assert resumed["import_approval"] == "selected"
    assert resumed["selected_paper_ids"] == ["arxiv:2210.02747"]
    assert resumed["status"] == "import_selected"


def test_import_approval_builds_direct_import_candidates() -> None:
    state = searched_papers_state()
    state.pop("arxiv_candidates")
    state["intent"] = {
        "schema_version": "query_intent_v1",
        "route": "direct_import",
        "task_type": "paper_import",
        "topic": "指定论文",
        "key_concepts": [],
        "requires_literature_evidence": True,
        "target_arxiv_ids": ["2210.02747"],
        "target_dois": ["10.1000/example"],
        "target_urls": ["https://example.org/paper.pdf"],
        "referenced_paper_ids": [],
        "output_language": "zh",
        "clarification_question": None,
    }

    request = _build_approval_request(state)

    assert request.source == "direct_import"
    assert [candidate.paper_id for candidate in request.candidates] == [
        "arxiv:2210.02747",
        "doi:10.1000/example",
        "url:https://example.org/paper.pdf",
    ]
