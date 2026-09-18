from typing import Any, cast

import pytest
from langchain_core.runnables import Runnable, RunnableLambda

from app.graph.nodes.generate_report import create_generate_report_node
from app.models.schemas import GroundedClaim, GroundedReportDraft, ReportSection


def report_state() -> dict[str, object]:
    return {
        "run_id": "run-1",
        "conversation_id": "conversation-1",
        "workspace_id": "default",
        "user_query": "流匹配如何构造生成路径？",
        "status": "augmented_library_retrieved",
        "intent": {
            "schema_version": "query_intent_v1",
            "route": "research",
            "task_type": "paper_analysis",
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
        "final_evidence": [
            {
                "chunk_id": "chunk-1",
                "paper_id": "arxiv-2210-02747",
                "title": "Flow Matching for Generative Modeling",
                "excerpt": "The method defines a conditional probability path.",
                "page_number": 2,
                "section": "Method",
                "score": 0.91,
            }
        ],
    }


@pytest.mark.asyncio
async def test_generate_report_renders_node_owned_citations() -> None:
    draft = GroundedReportDraft(
        sections=[
            ReportSection(
                heading="核心机制",
                claims=[
                    GroundedClaim(
                        statement="该方法定义条件概率路径。",
                        evidence_id="chunk-1",
                        supporting_quote="defines a conditional probability path",
                    )
                ],
            )
        ],
        limitations=["当前证据只覆盖一个方法片段。"],
    )
    agent: Runnable[dict[str, object], dict[str, object]] = RunnableLambda(
        lambda _: {"structured_response": draft}
    )
    node = create_generate_report_node(agent)

    update = await node(cast(Any, report_state()))

    assert "# 流匹配如何构造生成路径？" in update["report_content"]
    assert "【Flow Matching for Generative Modeling，Method，p. 2】" in update["report_content"]
    assert update["report_citations"] == [
        {
            "chunk_id": "chunk-1",
            "citation_label": "【Flow Matching for Generative Modeling，Method，p. 2】",
            "page_number": 2,
        }
    ]
    assert update["status"] == "report_generated"


@pytest.mark.asyncio
async def test_generate_report_rejects_hallucinated_evidence_id() -> None:
    draft = GroundedReportDraft(
        sections=[
            ReportSection(
                heading="错误引用",
                claims=[
                    GroundedClaim(
                        statement="内容。",
                        evidence_id="unknown",
                        supporting_quote="copied text",
                    )
                ],
            )
        ],
    )
    agent = cast(
        Runnable[dict[str, object], dict[str, object]],
        RunnableLambda(lambda _: {"structured_response": draft}),
    )
    node = create_generate_report_node(agent)

    with pytest.raises(ValueError, match="not supplied"):
        await node(cast(Any, report_state()))


@pytest.mark.asyncio
async def test_generate_report_uses_local_evidence_when_no_augmented_retrieval_exists() -> None:
    draft = GroundedReportDraft(
        sections=[
            ReportSection(
                heading="结论",
                claims=[
                    GroundedClaim(
                        statement="局部结论。",
                        evidence_id="chunk-1",
                        supporting_quote="conditional probability path",
                    )
                ],
            )
        ],
    )
    state = report_state()
    state["local_evidence"] = state.pop("final_evidence")
    node = create_generate_report_node(
        cast(
            Runnable[dict[str, object], dict[str, object]],
            RunnableLambda(lambda _: {"structured_response": draft}),
        )
    )

    update = await node(cast(Any, state))

    assert update["final_evidence_ids"] == ["chunk-1"]


@pytest.mark.asyncio
async def test_generate_report_rejects_a_quote_not_present_in_cited_evidence() -> None:
    draft = GroundedReportDraft(
        sections=[
            ReportSection(
                heading="错误摘录",
                claims=[
                    GroundedClaim(
                        statement="不受证据支持。",
                        evidence_id="chunk-1",
                        supporting_quote="invented supporting quotation",
                    )
                ],
            )
        ]
    )
    node = create_generate_report_node(
        cast(
            Runnable[dict[str, object], dict[str, object]],
            RunnableLambda(lambda _: {"structured_response": draft}),
        )
    )

    with pytest.raises(ValueError, match="supporting_quote"):
        await node(cast(Any, report_state()))
