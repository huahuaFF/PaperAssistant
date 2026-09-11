"""Smoke-test MiniMax structured classification without exposing credentials."""

from __future__ import annotations

import asyncio

from app.agent.classification import build_classification_agent
from app.agent.context import ClassificationContextBuilder
from app.config import get_settings
from app.graph.nodes.classify_query import create_classify_query_node
from app.llm.minimax import create_minimax_chat_model


async def main() -> None:
    settings = get_settings()
    model = create_minimax_chat_model(settings)
    agent = build_classification_agent(model)
    node = create_classify_query_node(agent, ClassificationContextBuilder())
    update = await node(
        {
            "run_id": "smoke-run",
            "conversation_id": "smoke-conversation",
            "workspace_id": "default",
            "user_query": "流匹配有哪些代表性论文？",
            "status": "started",
        }
    )
    print("Calling MiniMax query_classifier...")
    print(update["intent"])


if __name__ == "__main__":
    asyncio.run(main())
