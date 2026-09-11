from typing import Any, cast

import pytest

from app.graph.nodes.contracts import NodeNotImplementedError, create_placeholder_nodes
from app.graph.research_graph import build_research_graph
from app.graph.routing import (
    WorkflowRoutingError,
    route_after_classification,
    route_after_coverage,
    route_after_import_approval,
    route_after_search_approval,
)
from app.graph.state import ResearchState


def workflow_state(**updates: Any) -> ResearchState:
    return cast(
        ResearchState,
        {
            "run_id": "run-1",
            "conversation_id": "conversation-1",
            "workspace_id": "default",
            "user_query": "流匹配有哪些代表性论文？",
            "status": "started",
            **updates,
        },
    )


def test_complete_graph_contains_every_planned_node() -> None:
    graph = build_research_graph(create_placeholder_nodes())
    node_names = set(graph.get_graph().nodes)

    assert {
        "classify_query",
        "retrieve_local",
        "assess_coverage",
        "request_search_approval",
        "build_arxiv_query",
        "search_arxiv",
        "request_import_approval",
        "ingest_papers",
        "retrieve_augmented_library",
        "generate_report",
        "explain_knowledge_gap",
        "report_candidates",
        "request_clarification",
        "explain_out_of_scope",
    } <= node_names


def test_external_search_branch_requires_explicit_approval() -> None:
    assert route_after_coverage(workflow_state(coverage={"sufficient": False})) == "request_search_approval"
    assert route_after_search_approval(workflow_state(search_approval="approved")) == "build_arxiv_query"
    assert route_after_search_approval(workflow_state(search_approval="rejected")) == "explain_knowledge_gap"

    with pytest.raises(WorkflowRoutingError):
        route_after_search_approval(workflow_state())


def test_classification_routes_non_research_and_ambiguous_requests_safely() -> None:
    assert route_after_classification(workflow_state(intent={"route": "research"})) == "retrieve_local"
    assert (
        route_after_classification(workflow_state(intent={"route": "direct_import"}))
        == "request_import_approval"
    )
    assert (
        route_after_classification(workflow_state(intent={"route": "needs_clarification"}))
        == "request_clarification"
    )
    assert (
        route_after_classification(workflow_state(intent={"route": "out_of_scope"}))
        == "explain_out_of_scope"
    )

    with pytest.raises(WorkflowRoutingError):
        route_after_classification(workflow_state(intent={"route": "unknown"}))


def test_import_branch_requires_a_selection() -> None:
    assert (
        route_after_import_approval(
            workflow_state(import_approval="selected", selected_paper_ids=["arxiv:2501.00001"])
        )
        == "ingest_papers"
    )
    assert route_after_import_approval(workflow_state(import_approval="skipped")) == "report_candidates"

    with pytest.raises(WorkflowRoutingError):
        route_after_import_approval(workflow_state(import_approval="selected", selected_paper_ids=[]))


@pytest.mark.asyncio
async def test_placeholder_graph_cannot_be_run_before_nodes_are_implemented() -> None:
    graph = build_research_graph(create_placeholder_nodes())

    with pytest.raises(NodeNotImplementedError, match="classify_query"):
        await graph.ainvoke(
            {
                "run_id": "run-1",
                "conversation_id": "conversation-1",
                "workspace_id": "default",
                "user_query": "流匹配有哪些代表性论文？",
                "status": "started",
            }
        )
