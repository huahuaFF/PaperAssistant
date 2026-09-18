"""Read-only comparison of two persisted workflow checkpoints.

Use this when identical user queries appear to take different graph branches.
It reports only the state that is available immediately before and after local
retrieval, so it can distinguish an intent/query divergence from a retrieval
or coverage decision divergence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any, cast

from app.config import get_settings
from app.graph.checkpointer import open_sqlite_checkpointer
from app.graph.factory import build_configured_research_graph
from app.graph.state import ResearchState


def _evidence_summary(evidence: object) -> list[dict[str, object]]:
    if not isinstance(evidence, list):
        return []
    return [
        {
            key: item.get(key)
            for key in ("chunk_id", "paper_id", "title", "page_number", "section", "score")
        }
        for item in evidence
        if isinstance(item, dict)
    ]


def _summary(state: ResearchState) -> dict[str, object]:
    return {
        "user_query": state.get("user_query"),
        "status": state.get("status"),
        "intent": state.get("intent"),
        "retrieval_query": state.get("retrieval_query"),
        "local_retrieval_stats": state.get("local_retrieval_stats"),
        "local_evidence": _evidence_summary(state.get("local_evidence")),
        "coverage": state.get("coverage"),
    }


async def _load(thread_id: str) -> ResearchState:
    settings = get_settings()
    async with open_sqlite_checkpointer(settings) as checkpointer:
        graph = build_configured_research_graph(settings, checkpointer=checkpointer)
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    if not snapshot.values:
        raise LookupError(f"No saved graph state found for thread_id={thread_id!r}.")
    return cast(ResearchState, dict(snapshot.values))


async def main(first_thread_id: str, second_thread_id: str) -> None:
    cast(Any, sys.stdout).reconfigure(encoding="utf-8")
    first, second = await asyncio.gather(_load(first_thread_id), _load(second_thread_id))
    print(json.dumps({first_thread_id: _summary(first), second_thread_id: _summary(second)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two saved workflow checkpoints.")
    parser.add_argument("first_thread_id")
    parser.add_argument("second_thread_id")
    args = parser.parse_args()
    asyncio.run(main(args.first_thread_id, args.second_thread_id))
