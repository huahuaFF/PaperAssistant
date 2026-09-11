from typing import Any, cast

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.graph.nodes.request_search_approval import request_search_approval
from app.graph.state import ResearchState


def pending_search_state() -> ResearchState:
    return {
        "run_id": "run-1",
        "conversation_id": "conversation-1",
        "workspace_id": "default",
        "user_query": "请列出流匹配的代表性论文。",
        "status": "coverage_assessed",
        "intent": {
            "schema_version": "query_intent_v1",
            "route": "research",
            "task_type": "literature_discovery",
            "topic": "Flow Matching",
            "key_concepts": ["flow matching"],
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
            "reason": "本地证据不足以提供代表性论文列表。",
            "missing_aspects": ["缺少多篇代表论文"],
            "recommended_action": "ask_arxiv_permission",
        },
        "local_evidence": [],
    }


@pytest.mark.asyncio
async def test_search_approval_interrupts_and_resumes_with_same_sqlite_thread(tmp_path: Any) -> None:
    checkpoint_path = tmp_path / "checkpoints.db"
    builder = StateGraph(ResearchState)
    builder.add_node("request_search_approval", request_search_approval)
    builder.add_edge(START, "request_search_approval")
    builder.add_edge("request_search_approval", END)

    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        await checkpointer.setup()
        graph = builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "approval-thread-1"}}
        interrupted = await graph.ainvoke(pending_search_state(), config=cast(Any, config))

        interrupt_payload = interrupted["__interrupt__"][0].value
        assert interrupt_payload == {
            "approval_type": "search_arxiv",
            "query": "请列出流匹配的代表性论文。",
            "topic": "Flow Matching",
            "coverage_reason": "本地证据不足以提供代表性论文列表。",
            "missing_aspects": ["缺少多篇代表论文"],
            "local_evidence_count": 0,
        }

    # Reopen the durable database to prove this is not an in-memory resume.
    async with AsyncSqliteSaver.from_conn_string(str(checkpoint_path)) as reopened_checkpointer:
        await reopened_checkpointer.setup()
        reopened_graph = builder.compile(checkpointer=reopened_checkpointer)
        resumed = await reopened_graph.ainvoke(
            Command(resume={"decision": "approve"}), config=cast(Any, config)
        )

    assert resumed["search_approval"] == "approved"
    assert resumed["status"] == "search_approved"
