"""Compose implemented Agent nodes into the complete research graph."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from app.agent.arxiv_query import build_arxiv_query_agent
from app.agent.classification import build_classification_agent
from app.agent.coverage import build_coverage_agent
from app.agent.report import build_report_agent
from app.config import Settings
from app.db import Database
from app.graph.nodes.assess_coverage import create_assess_coverage_node
from app.graph.nodes.build_arxiv_query import create_build_arxiv_query_node
from app.graph.nodes.classify_query import create_classify_query_node
from app.graph.nodes.contracts import ResearchNode, ResearchNodes, create_placeholder_nodes
from app.graph.nodes.generate_report import create_generate_report_node
from app.graph.nodes.ingest_papers import create_ingest_papers_node
from app.graph.nodes.request_import_approval import request_import_approval
from app.graph.nodes.request_search_approval import request_search_approval
from app.graph.nodes.retrieve_augmented_library import create_retrieve_augmented_library_node
from app.graph.nodes.retrieve_local import create_retrieve_local_node
from app.graph.nodes.search_arxiv import create_search_arxiv_node
from app.graph.nodes.terminal_responses import (
    create_explain_knowledge_gap_node,
    create_explain_out_of_scope_node,
    create_report_candidates_node,
    create_request_clarification_node,
)
from app.graph.research_graph import build_research_graph
from app.llm.embeddings import create_dashscope_embeddings
from app.llm.minimax import create_minimax_chat_model
from app.repositories.chroma_store import ChromaStore
from app.repositories.ingestion_repository import SQLiteIngestionRepository
from app.services.arxiv_search import ArxivSearchClient, create_search_arxiv_tool
from app.services.local_retrieval import LocalEvidenceRetriever
from app.services.paper_indexing import PaperIndexer
from app.services.paper_ingestion import PaperDownloader, PaperIngestionCoordinator
from app.services.paper_parsing import PaperParser


def compose_research_nodes(
    *,
    classify_query: ResearchNode,
    retrieve_local: ResearchNode,
    assess_coverage: ResearchNode,
    request_search_approval: ResearchNode,
    build_arxiv_query: ResearchNode,
    search_arxiv: ResearchNode,
    request_import_approval: ResearchNode,
    ingest_papers: ResearchNode,
    retrieve_augmented_library: ResearchNode,
    generate_report: ResearchNode,
    explain_knowledge_gap: ResearchNode,
    report_candidates: ResearchNode,
    request_clarification: ResearchNode,
    explain_out_of_scope: ResearchNode,
) -> ResearchNodes:
    """Replace only completed nodes while preserving explicit placeholders."""
    return replace(
        create_placeholder_nodes(),
        classify_query=classify_query,
        retrieve_local=retrieve_local,
        assess_coverage=assess_coverage,
        request_search_approval=request_search_approval,
        build_arxiv_query=build_arxiv_query,
        search_arxiv=search_arxiv,
        request_import_approval=request_import_approval,
        ingest_papers=ingest_papers,
        retrieve_augmented_library=retrieve_augmented_library,
        generate_report=generate_report,
        explain_knowledge_gap=explain_knowledge_gap,
        report_candidates=report_candidates,
        request_clarification=request_clarification,
        explain_out_of_scope=explain_out_of_scope,
    )


def build_research_nodes(settings: Settings) -> ResearchNodes:
    """Build the current node set; later steps replace additional placeholders."""
    minimax_model = create_minimax_chat_model(settings)
    classification_agent = build_classification_agent(minimax_model)
    classify_query = create_classify_query_node(classification_agent)
    coverage_agent = build_coverage_agent(minimax_model)
    assess_coverage = create_assess_coverage_node(coverage_agent)
    report_agent = build_report_agent(minimax_model)
    generate_report = create_generate_report_node(report_agent)
    arxiv_query_agent = build_arxiv_query_agent(minimax_model)
    build_arxiv_query = create_build_arxiv_query_node(arxiv_query_agent)
    search_arxiv = create_search_arxiv_node(create_search_arxiv_tool(ArxivSearchClient()))
    vector_store = ChromaStore(settings).as_langchain_vector_store(create_dashscope_embeddings(settings))
    evidence_retriever = LocalEvidenceRetriever(vector_store)
    retrieve_local = create_retrieve_local_node(evidence_retriever)
    retrieve_augmented_library = create_retrieve_augmented_library_node(evidence_retriever)
    ingest_papers = create_ingest_papers_node(
        PaperIngestionCoordinator(
            downloader=PaperDownloader(settings.paper_directory),
            parser=PaperParser(),
            indexer=PaperIndexer(vector_store),
            records=SQLiteIngestionRepository(Database(settings.database_path)),
            parsed_directory=settings.parsed_directory,
        )
    )
    return compose_research_nodes(
        classify_query=classify_query,
        retrieve_local=retrieve_local,
        assess_coverage=assess_coverage,
        request_search_approval=request_search_approval,
        build_arxiv_query=build_arxiv_query,
        search_arxiv=search_arxiv,
        request_import_approval=request_import_approval,
        ingest_papers=ingest_papers,
        retrieve_augmented_library=retrieve_augmented_library,
        generate_report=generate_report,
        explain_knowledge_gap=create_explain_knowledge_gap_node(),
        report_candidates=create_report_candidates_node(),
        request_clarification=create_request_clarification_node(),
        explain_out_of_scope=create_explain_out_of_scope_node(),
    )


def build_configured_research_graph(settings: Settings, *, checkpointer: Any | None = None) -> Any:
    """Build the complete graph with every node currently implemented."""
    return build_research_graph(build_research_nodes(settings), checkpointer=checkpointer)
