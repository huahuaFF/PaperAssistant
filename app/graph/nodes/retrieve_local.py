"""LangGraph node that retrieves evidence from the local Chroma library."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from app.graph.nodes.contracts import NodeUpdate
from app.graph.state import ResearchState
from app.models.schemas import QueryIntent
from app.services.local_retrieval import LocalEvidenceRetriever, build_retrieval_query


def create_retrieve_local_node(
    retriever: LocalEvidenceRetriever,
) -> Callable[[ResearchState], Awaitable[NodeUpdate]]:
    """Create a no-LLM local-RAG node for the research branch."""

    async def retrieve_local(state: ResearchState) -> NodeUpdate:
        intent_data = state.get("intent")
        if not isinstance(intent_data, dict):
            raise TypeError("retrieve_local requires intent from classify_query.")
        intent = QueryIntent.model_validate(intent_data)
        query = build_retrieval_query(user_query=state["user_query"], intent=intent)
        result = await retriever.retrieve(query)
        evidence = [item.model_dump() for item in result.evidence]
        return {
            "retrieval_query": result.query,
            "local_evidence": evidence,
            "local_evidence_ids": [item["chunk_id"] for item in evidence],
            "local_retrieval_stats": {
                "candidate_count": result.candidate_count,
                "selected_count": len(evidence),
                "unique_paper_count": result.unique_paper_count,
            },
            "status": "local_retrieved",
        }

    return retrieve_local
