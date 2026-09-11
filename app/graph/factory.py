"""Compose implemented Agent nodes into the complete research graph."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from app.agent.arxiv_query import build_arxiv_query_agent
from app.agent.classification import build_classification_agent
from app.agent.coverage import build_coverage_agent
from app.config import Settings
from app.graph.nodes.assess_coverage import create_assess_coverage_node
from app.graph.nodes.build_arxiv_query import create_build_arxiv_query_node
from app.graph.nodes.classify_query import create_classify_query_node
from app.graph.nodes.contracts import ResearchNode, ResearchNodes, create_placeholder_nodes
from app.graph.nodes.request_search_approval import request_search_approval
from app.graph.nodes.retrieve_local import create_retrieve_local_node
from app.graph.research_graph import build_research_graph
from app.llm.embeddings import create_dashscope_embeddings
from app.llm.minimax import create_minimax_chat_model
from app.repositories.chroma_store import ChromaStore
from app.services.local_retrieval import LocalEvidenceRetriever


def compose_research_nodes(
    *,
    classify_query: ResearchNode,
    retrieve_local: ResearchNode,
    assess_coverage: ResearchNode,
    request_search_approval: ResearchNode,
    build_arxiv_query: ResearchNode,
) -> ResearchNodes:
    """Replace only completed nodes while preserving explicit placeholders."""
    return replace(
        create_placeholder_nodes(),
        classify_query=classify_query,
        retrieve_local=retrieve_local,
        assess_coverage=assess_coverage,
        request_search_approval=request_search_approval,
        build_arxiv_query=build_arxiv_query,
    )


def build_research_nodes(settings: Settings) -> ResearchNodes:
    """Build the current node set; later steps replace additional placeholders."""
    minimax_model = create_minimax_chat_model(settings)
    classification_agent = build_classification_agent(minimax_model)
    classify_query = create_classify_query_node(classification_agent)
    coverage_agent = build_coverage_agent(minimax_model)
    assess_coverage = create_assess_coverage_node(coverage_agent)
    arxiv_query_agent = build_arxiv_query_agent(minimax_model)
    build_arxiv_query = create_build_arxiv_query_node(arxiv_query_agent)
    vector_store = ChromaStore(settings).as_langchain_vector_store(create_dashscope_embeddings(settings))
    retrieve_local = create_retrieve_local_node(LocalEvidenceRetriever(vector_store))
    return compose_research_nodes(
        classify_query=classify_query,
        retrieve_local=retrieve_local,
        assess_coverage=assess_coverage,
        request_search_approval=request_search_approval,
        build_arxiv_query=build_arxiv_query,
    )


def build_configured_research_graph(settings: Settings, *, checkpointer: Any | None = None) -> Any:
    """Build the complete graph with every node currently implemented."""
    return build_research_graph(build_research_nodes(settings), checkpointer=checkpointer)
