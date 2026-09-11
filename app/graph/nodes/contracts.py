"""Node contracts for the complete research workflow topology.

The graph is intentionally assembled from this dependency container. This lets
us finalize routes and review gates before implementing any individual node.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.graph.state import ResearchState

NodeUpdate = dict[str, Any]
ResearchNode = Callable[[ResearchState], NodeUpdate | Awaitable[NodeUpdate]]


@dataclass(frozen=True)
class ResearchNodes:
    """All node implementations required by the research workflow."""

    classify_query: ResearchNode
    retrieve_local: ResearchNode
    assess_coverage: ResearchNode
    request_search_approval: ResearchNode
    build_arxiv_query: ResearchNode
    search_arxiv: ResearchNode
    request_import_approval: ResearchNode
    ingest_papers: ResearchNode
    retrieve_augmented_library: ResearchNode
    generate_report: ResearchNode
    explain_knowledge_gap: ResearchNode
    report_candidates: ResearchNode
    request_clarification: ResearchNode
    explain_out_of_scope: ResearchNode


class NodeNotImplementedError(NotImplementedError):
    """Raised when a graph skeleton is run before a node is implemented."""


def create_placeholder_nodes() -> ResearchNodes:
    """Create an executable-safe topology placeholder for design-time testing."""

    def placeholder(name: str) -> ResearchNode:
        async def node(_: ResearchState) -> NodeUpdate:
            raise NodeNotImplementedError(f"Workflow node '{name}' is not implemented yet.")

        return node

    return ResearchNodes(
        classify_query=placeholder("classify_query"),
        retrieve_local=placeholder("retrieve_local"),
        assess_coverage=placeholder("assess_coverage"),
        request_search_approval=placeholder("request_search_approval"),
        build_arxiv_query=placeholder("build_arxiv_query"),
        search_arxiv=placeholder("search_arxiv"),
        request_import_approval=placeholder("request_import_approval"),
        ingest_papers=placeholder("ingest_papers"),
        retrieve_augmented_library=placeholder("retrieve_augmented_library"),
        generate_report=placeholder("generate_report"),
        explain_knowledge_gap=placeholder("explain_knowledge_gap"),
        report_candidates=placeholder("report_candidates"),
        request_clarification=placeholder("request_clarification"),
        explain_out_of_scope=placeholder("explain_out_of_scope"),
    )
