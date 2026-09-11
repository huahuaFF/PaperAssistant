"""Smoke-test approved arXiv search planning without executing a network search."""

from __future__ import annotations

import asyncio
import sys
from typing import Any, cast

from app.agent.arxiv_query import build_arxiv_query_agent
from app.config import get_settings
from app.graph.nodes.build_arxiv_query import create_build_arxiv_query_node
from app.llm.minimax import create_minimax_chat_model


async def main() -> None:
    cast(Any, sys.stdout).reconfigure(encoding="utf-8")
    settings = get_settings()
    node = create_build_arxiv_query_node(
        build_arxiv_query_agent(create_minimax_chat_model(settings))
    )
    update = await node(
        {
            "run_id": "arxiv-query-smoke",
            "conversation_id": "arxiv-query-smoke",
            "workspace_id": "default",
            "user_query": "请列出流匹配在生成建模中的代表性论文，并说明发展脉络。",
            "status": "search_approved",
            "search_approval": "approved",
            "intent": {
                "schema_version": "query_intent_v1",
                "route": "research",
                "task_type": "literature_discovery",
                "topic": "Flow Matching for Generative Modeling",
                "key_concepts": ["flow matching", "generative modeling"],
                "requires_literature_evidence": True,
                "target_arxiv_ids": [],
                "target_dois": [],
                "target_urls": [],
                "referenced_paper_ids": [],
                "output_language": "zh",
                "clarification_question": None,
            },
            "coverage": {
                "sufficient": False,
                "confidence": 0.98,
                "reason": "本地文献库没有覆盖流匹配的代表性论文和发展脉络。",
                "missing_aspects": ["缺少多篇基础与后续代表论文", "缺少时间线证据"],
                "recommended_action": "ask_arxiv_permission",
            },
            "local_evidence": [],
        }
    )
    print(update["arxiv_search_plan"])


if __name__ == "__main__":
    asyncio.run(main())
