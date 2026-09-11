"""Run the actual retrieve_local node against the local Chroma library."""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any, cast

from app.config import get_settings
from app.graph.nodes.retrieve_local import create_retrieve_local_node
from app.llm.embeddings import create_dashscope_embeddings
from app.models.schemas import QueryIntent
from app.repositories.chroma_store import ChromaStore
from app.services.local_retrieval import LocalEvidenceRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test local retrieval with a pre-classified query.")
    parser.add_argument("query")
    parser.add_argument("--topic", default="FlowCF")
    parser.add_argument("--concept", action="append", default=["flow matching", "collaborative filtering"])
    return parser.parse_args()


async def main() -> None:
    cast(Any, sys.stdout).reconfigure(encoding="utf-8")
    args = parse_args()
    settings = get_settings()
    vector_store = ChromaStore(settings).as_langchain_vector_store(create_dashscope_embeddings(settings))
    node = create_retrieve_local_node(LocalEvidenceRetriever(vector_store))
    intent = QueryIntent(
        route="research",
        task_type="literature_question",
        topic=args.topic,
        key_concepts=args.concept,
        requires_literature_evidence=True,
    )
    update = await node(
        {
            "run_id": "retrieve-smoke",
            "conversation_id": "retrieve-smoke",
            "workspace_id": "default",
            "user_query": args.query,
            "intent": intent.model_dump(),
            "status": "intent_classified",
        }
    )
    print(f"retrieval_query={update['retrieval_query']}")
    print(f"stats={update['local_retrieval_stats']}")
    for item in update["local_evidence"]:
        print(
            f"score={item['score']:.3f} page={item['page_number']} "
            f"section={item.get('section', 'unknown')} chunk={item['chunk_id']}"
        )
        print(item["excerpt"][:240].replace("\n", " "))


if __name__ == "__main__":
    asyncio.run(main())
