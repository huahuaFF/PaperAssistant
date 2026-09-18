from datetime import datetime, timezone

import pytest

from app.graph.nodes.terminal_responses import (
    create_explain_knowledge_gap_node,
    create_explain_out_of_scope_node,
    create_report_candidates_node,
    create_request_clarification_node,
)


def base_state() -> dict[str, object]:
    return {
        "run_id": "run-1",
        "conversation_id": "conversation-1",
        "workspace_id": "default",
        "user_query": "请帮我处理这个请求",
        "status": "intent_classified",
    }


@pytest.mark.asyncio
async def test_clarification_terminal_node_uses_the_classified_question() -> None:
    state = base_state()
    state["intent"] = {
        "route": "needs_clarification",
        "task_type": "other",
        "topic": "不明确请求",
        "requires_literature_evidence": False,
        "clarification_question": "你想检索论文，还是解读已有论文？",
    }

    update = await create_request_clarification_node()(state)  # type: ignore[arg-type]

    assert update == {
        "report_content": "你想检索论文，还是解读已有论文？",
        "status": "clarification_requested",
    }


@pytest.mark.asyncio
async def test_out_of_scope_terminal_node_does_not_claim_research() -> None:
    state = base_state()
    state["intent"] = {
        "route": "out_of_scope",
        "task_type": "other",
        "topic": "天气",
        "requires_literature_evidence": False,
    }

    update = await create_explain_out_of_scope_node()(state)  # type: ignore[arg-type]

    assert update["status"] == "out_of_scope_explained"
    assert "科研文献助手" in update["report_content"]


@pytest.mark.asyncio
async def test_knowledge_gap_terminal_node_requires_rejected_search_permission() -> None:
    state = base_state()
    state.update(
        {
            "search_approval": "rejected",
            "coverage": {
                "sufficient": False,
                "confidence": 0.2,
                "reason": "本地证据不能回答该问题。",
                "missing_aspects": ["缺少方法细节"],
                "recommended_action": "ask_arxiv_permission",
            },
        }
    )

    update = await create_explain_knowledge_gap_node()(state)  # type: ignore[arg-type]

    assert update["status"] == "knowledge_gap_explained"
    assert "缺少方法细节" in update["report_content"]


@pytest.mark.asyncio
async def test_candidate_terminal_node_reports_search_results_without_importing() -> None:
    now = datetime.now(timezone.utc)
    state = base_state()
    state.update(
        {
            "import_approval": "skipped",
            "arxiv_candidates": [
                {
                    "arxiv_id": "2210.02747",
                    "title": "Flow Matching for Generative Modeling",
                    "authors": ["Yaron Lipman"],
                    "abstract": "A flow matching paper.",
                    "categories": ["cs.LG"],
                    "published_at": now,
                    "updated_at": now,
                    "abs_url": "https://arxiv.org/abs/2210.02747",
                    "pdf_url": "https://arxiv.org/pdf/2210.02747",
                }
            ],
        }
    )

    update = await create_report_candidates_node()(state)  # type: ignore[arg-type]

    assert update["status"] == "candidates_reported"
    assert "Flow Matching for Generative Modeling" in update["report_content"]


@pytest.mark.asyncio
async def test_candidate_terminal_node_reports_direct_import_candidates() -> None:
    state = base_state()
    state.update(
        {
            "import_approval": "skipped",
            "import_candidates": [
                {
                    "paper_id": "doi:10.1000/example",
                    "source_type": "doi",
                    "source_url": "https://doi.org/10.1000/example",
                    "title": None,
                    "authors": [],
                    "abstract": None,
                    "categories": [],
                }
            ],
        }
    )

    update = await create_report_candidates_node()(state)  # type: ignore[arg-type]

    assert "doi:10.1000/example" in update["report_content"]
