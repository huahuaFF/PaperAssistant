"""Offline end-to-end contracts for every currently implemented graph branch."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest

from app.graph.nodes.contracts import create_placeholder_nodes
from app.graph.research_graph import build_research_graph
from tests.support.workflow import traced_node, workflow_state


def _nodes_for(
    trace: list[str], **overrides: Any
) -> Any:
    """Build a complete fake node container; every invoked node is explicit."""
    defaults = {
        "classify_query": traced_node("classify_query", {}, trace),
        "retrieve_local": traced_node("retrieve_local", {}, trace),
        "assess_coverage": traced_node("assess_coverage", {}, trace),
        "request_search_approval": traced_node("request_search_approval", {}, trace),
        "build_arxiv_query": traced_node("build_arxiv_query", {}, trace),
        "search_arxiv": traced_node("search_arxiv", {}, trace),
        "request_import_approval": traced_node("request_import_approval", {}, trace),
        "ingest_papers": traced_node("ingest_papers", {}, trace),
        "retrieve_augmented_library": traced_node("retrieve_augmented_library", {}, trace),
        "generate_report": traced_node("generate_report", {"status": "report_generated"}, trace),
        "explain_knowledge_gap": traced_node(
            "explain_knowledge_gap", {"status": "knowledge_gap_explained"}, trace
        ),
        "report_candidates": traced_node("report_candidates", {"status": "candidates_reported"}, trace),
        "request_clarification": traced_node(
            "request_clarification", {"status": "clarification_requested"}, trace
        ),
        "explain_out_of_scope": traced_node(
            "explain_out_of_scope", {"status": "out_of_scope_explained"}, trace
        ),
    }
    defaults.update(overrides)
    return replace(create_placeholder_nodes(), **defaults)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_local_evidence_branch_ends_at_report_without_external_search() -> None:
    trace: list[str] = []
    graph = build_research_graph(
        _nodes_for(
            trace,
            classify_query=traced_node("classify_query", {"intent": {"route": "research"}}, trace),
            assess_coverage=traced_node("assess_coverage", {"coverage": {"sufficient": True}}, trace),
        )
    )

    result = await graph.ainvoke(workflow_state())

    assert trace == ["classify_query", "retrieve_local", "assess_coverage", "generate_report"]
    assert result["status"] == "report_generated"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_rejected_search_ends_with_knowledge_gap_without_calling_arxiv() -> None:
    trace: list[str] = []
    graph = build_research_graph(
        _nodes_for(
            trace,
            classify_query=traced_node("classify_query", {"intent": {"route": "research"}}, trace),
            assess_coverage=traced_node("assess_coverage", {"coverage": {"sufficient": False}}, trace),
            request_search_approval=traced_node(
                "request_search_approval", {"search_approval": "rejected"}, trace
            ),
        )
    )

    result = await graph.ainvoke(workflow_state())

    assert trace == [
        "classify_query",
        "retrieve_local",
        "assess_coverage",
        "request_search_approval",
        "explain_knowledge_gap",
    ]
    assert result["status"] == "knowledge_gap_explained"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_approved_search_can_end_at_candidates_when_import_is_skipped() -> None:
    trace: list[str] = []
    graph = build_research_graph(
        _nodes_for(
            trace,
            classify_query=traced_node("classify_query", {"intent": {"route": "research"}}, trace),
            assess_coverage=traced_node("assess_coverage", {"coverage": {"sufficient": False}}, trace),
            request_search_approval=traced_node(
                "request_search_approval", {"search_approval": "approved"}, trace
            ),
            request_import_approval=traced_node(
                "request_import_approval", {"import_approval": "skipped"}, trace
            ),
        )
    )

    result = await graph.ainvoke(workflow_state())

    assert trace == [
        "classify_query",
        "retrieve_local",
        "assess_coverage",
        "request_search_approval",
        "build_arxiv_query",
        "search_arxiv",
        "request_import_approval",
        "report_candidates",
    ]
    assert result["status"] == "candidates_reported"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_direct_import_selected_path_indexes_then_reports() -> None:
    trace: list[str] = []
    graph = build_research_graph(
        _nodes_for(
            trace,
            classify_query=traced_node("classify_query", {"intent": {"route": "direct_import"}}, trace),
            request_import_approval=traced_node(
                "request_import_approval",
                {"import_approval": "selected", "selected_paper_ids": ["arxiv:2210.02747"]},
                trace,
            ),
        )
    )

    result = await graph.ainvoke(workflow_state())

    assert trace == [
        "classify_query",
        "request_import_approval",
        "ingest_papers",
        "retrieve_augmented_library",
        "generate_report",
    ]
    assert result["status"] == "report_generated"
