"""Smoke-test Agent-backed coverage assessment using actual local retrieval evidence."""

from __future__ import annotations

import asyncio
import sys
from typing import Any, cast

from app.agent.coverage import build_coverage_agent
from app.config import get_settings
from app.graph.nodes.assess_coverage import create_assess_coverage_node
from app.graph.nodes.retrieve_local import create_retrieve_local_node
from app.llm.embeddings import create_dashscope_embeddings
from app.llm.minimax import create_minimax_chat_model
from app.models.schemas import QueryIntent
from app.repositories.chroma_store import ChromaStore
from app.services.local_retrieval import LocalEvidenceRetriever


async def main() -> None:
    cast(Any, sys.stdout).reconfigure(encoding="utf-8")
    settings = get_settings()
    intent = QueryIntent(
        route="research",
        task_type="literature_question",
        topic="FlowCF",
        key_concepts=["behavior-guided prior", "collaborative filtering"],
        requires_literature_evidence=True,
    )
    state: dict[str, object] = {
        "run_id": "coverage-smoke",
        "conversation_id": "coverage-smoke",
        "workspace_id": "default",
        "user_query": "FlowCF 的 behavior-guided prior 如何帮助协同过滤？",
        "status": "intent_classified",
        "intent": intent.model_dump(),
    }
    vector_store = ChromaStore(settings).as_langchain_vector_store(create_dashscope_embeddings(settings))
    retrieve_node = create_retrieve_local_node(LocalEvidenceRetriever(vector_store))
    state.update(await retrieve_node(cast(Any, state)))
    coverage_node = create_assess_coverage_node(
        build_coverage_agent(create_minimax_chat_model(settings))
    )
    update = await coverage_node(cast(Any, state))
    print(f"retrieval_stats={state['local_retrieval_stats']}")
    print(f"coverage={update['coverage']}")


if __name__ == "__main__":
    asyncio.run(main())
