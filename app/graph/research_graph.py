"""Complete LangGraph topology for the controlled research Agent.

Only the graph and route rules live here. Individual node implementations are
injected through ``ResearchNodes`` and are developed separately.
"""

from __future__ import annotations

from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.contracts import ResearchNodes
from app.graph.routing import (
    route_after_classification,
    route_after_coverage,
    route_after_import_approval,
    route_after_search_approval,
)
from app.graph.state import ResearchState


def build_research_graph(
    nodes: ResearchNodes, *, checkpointer: Any | None = None
) -> Any:
    """Build the complete workflow without coupling it to node implementations.

    Invariant: external search and ingestion are reachable only through their
    respective human-approval nodes.
    """
    builder = StateGraph(ResearchState)

    def add_node(name: str, action: Any) -> None:
        # LangGraph supports sync and async callables, while its current type
        # overloads cannot express their union in a dependency container.
        builder.add_node(name, cast(Any, action))

    add_node("classify_query", nodes.classify_query)
    add_node("retrieve_local", nodes.retrieve_local)
    add_node("assess_coverage", nodes.assess_coverage)
    add_node("request_search_approval", nodes.request_search_approval)
    add_node("build_arxiv_query", nodes.build_arxiv_query)
    add_node("search_arxiv", nodes.search_arxiv)
    add_node("request_import_approval", nodes.request_import_approval)
    add_node("ingest_papers", nodes.ingest_papers)
    add_node("retrieve_augmented_library", nodes.retrieve_augmented_library)
    add_node("generate_report", nodes.generate_report)
    add_node("explain_knowledge_gap", nodes.explain_knowledge_gap)
    add_node("report_candidates", nodes.report_candidates)
    add_node("request_clarification", nodes.request_clarification)
    add_node("explain_out_of_scope", nodes.explain_out_of_scope)

    builder.add_edge(START, "classify_query")
    builder.add_conditional_edges(
        "classify_query",
        route_after_classification,
        {
            "retrieve_local": "retrieve_local",
            "request_import_approval": "request_import_approval",
            "request_clarification": "request_clarification",
            "explain_out_of_scope": "explain_out_of_scope",
        },
    )
    builder.add_edge("retrieve_local", "assess_coverage")
    builder.add_conditional_edges(
        "assess_coverage",
        route_after_coverage,
        {
            "generate_report": "generate_report",
            "request_search_approval": "request_search_approval",
        },
    )
    builder.add_conditional_edges(
        "request_search_approval",
        route_after_search_approval,
        {
            "build_arxiv_query": "build_arxiv_query",
            "explain_knowledge_gap": "explain_knowledge_gap",
        },
    )
    builder.add_edge("build_arxiv_query", "search_arxiv")
    builder.add_edge("search_arxiv", "request_import_approval")
    builder.add_conditional_edges(
        "request_import_approval",
        route_after_import_approval,
        {
            "ingest_papers": "ingest_papers",
            "report_candidates": "report_candidates",
        },
    )
    builder.add_edge("ingest_papers", "retrieve_augmented_library")
    builder.add_edge("retrieve_augmented_library", "generate_report")
    builder.add_edge("generate_report", END)
    builder.add_edge("explain_knowledge_gap", END)
    builder.add_edge("report_candidates", END)
    builder.add_edge("request_clarification", END)
    builder.add_edge("explain_out_of_scope", END)
    return builder.compile(checkpointer=checkpointer)
