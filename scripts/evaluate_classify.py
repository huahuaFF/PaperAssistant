"""Run a manual, cost-bearing evaluation set against the classify-query Agent."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from app.agent.classification import build_classification_agent
from app.config import get_settings
from app.graph.nodes.classify_query import create_classify_query_node
from app.llm.minimax import create_minimax_chat_model

CASES_PATH = Path("tests/fixtures/classify_query_cases.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate query classification against MiniMax.")
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N cases.")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if args.limit is not None:
        cases = cases[: args.limit]

    node = create_classify_query_node(
        build_classification_agent(create_minimax_chat_model(get_settings()))
    )
    passed = 0
    for case in cases:
        update = await node(
            {
                "run_id": f"eval-{case['id']}",
                "conversation_id": "classification-eval",
                "workspace_id": "default",
                "user_query": case["query"],
                "status": "started",
            }
        )
        intent = update["intent"]
        actual = (intent["route"], intent["task_type"])
        expected = (case["expected_route"], case["expected_task_type"])
        success = actual == expected
        passed += int(success)
        print(f"{'PASS' if success else 'FAIL'} {case['id']}: expected={expected}, actual={actual}")

    print(f"\\n{passed}/{len(cases)} cases matched the exact routing expectation.")


if __name__ == "__main__":
    asyncio.run(main())
