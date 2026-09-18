"""Diagnose whether coverage latency comes from MiniMax or Agent structured output.

This script does not modify graph behavior. It runs two measurements over the
same coverage prompt: a raw MiniMax call and the existing coverage Agent. The
optional diagnostic timeout is local to this script and only prevents a manual
diagnostic session from waiting indefinitely.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from time import perf_counter
from typing import Any, cast

from app.agent.coverage import build_coverage_agent
from app.agent.coverage_context import CoverageContextBuilder
from app.config import get_settings
from app.graph.checkpointer import open_sqlite_checkpointer
from app.graph.factory import build_configured_research_graph
from app.graph.state import ResearchState
from app.llm.minimax import create_minimax_chat_model
from app.prompts.assess_coverage import COVERAGE_PROMPT


async def main(thread_id: str, timeout_seconds: float) -> None:
    cast(Any, sys.stdout).reconfigure(encoding="utf-8")
    settings = get_settings()
    if settings.minimax_api_key is None:
        raise RuntimeError("MINIMAX_API_KEY is required for this diagnostic.")
    state = await _load_checkpoint_state(settings, thread_id)
    context = CoverageContextBuilder().build(state)
    prompt = COVERAGE_PROMPT.invoke(context.prompt_values())
    print(f"thread_id={thread_id}")
    print(f"query={state['user_query']}")
    print(f"evidence_prompt_chars={len(context.evidence_json)}")
    print(f"diagnostic_timeout_seconds={timeout_seconds:g}")

    model = create_minimax_chat_model(settings)
    raw_completed = await _measure_raw_model(model, prompt.to_messages(), timeout_seconds)
    if not raw_completed:
        print("\nConclusion: the raw MiniMax request did not complete inside the diagnostic budget.")
        print("The structured-Agent experiment is skipped because provider or transport latency is primary.")
        return

    agent = build_coverage_agent(model)
    await _measure_coverage_agent(agent, prompt.to_messages(), timeout_seconds)


async def _load_checkpoint_state(settings: Any, thread_id: str) -> ResearchState:
    async with open_sqlite_checkpointer(settings) as checkpointer:
        graph = build_configured_research_graph(settings, checkpointer=checkpointer)
        snapshot = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    if not snapshot.values:
        raise LookupError(f"No saved graph state found for thread_id={thread_id!r}.")
    return cast(ResearchState, dict(snapshot.values))


async def _measure_raw_model(
    model: Any, messages: list[Any], timeout_seconds: float
) -> bool:
    print("\n=== Experiment A: raw MiniMax model ===")
    started_at = perf_counter()
    try:
        async with asyncio.timeout(timeout_seconds):
            response = await model.ainvoke(messages)
    except TimeoutError:
        print(f"raw_model_timeout={perf_counter() - started_at:.2f}s")
        return False
    except Exception as error:  # noqa: BLE001 - diagnostics must print unexpected provider errors.
        print(f"raw_model_error={type(error).__name__}: {error}")
        return False
    print(f"raw_model_completed={perf_counter() - started_at:.2f}s")
    print(f"raw_model_response_chars={len(str(response.content))}")
    return True


async def _measure_coverage_agent(
    agent: Any, messages: list[Any], timeout_seconds: float
) -> None:
    print("\n=== Experiment B: create_agent + ToolStrategy ===")
    started_at = perf_counter()
    model_calls = 0
    try:
        async with asyncio.timeout(timeout_seconds):
            async for event in agent.astream_events({"messages": messages}, version="v2"):
                event_name = event.get("event")
                if event_name == "on_chat_model_start":
                    model_calls += 1
                    print(f"chat_model_start #{model_calls} at {perf_counter() - started_at:.2f}s")
                elif event_name == "on_chat_model_end":
                    print(f"chat_model_end   #{model_calls} at {perf_counter() - started_at:.2f}s")
                elif event_name == "on_chat_model_error":
                    print(f"chat_model_error #{model_calls} at {perf_counter() - started_at:.2f}s")
    except TimeoutError:
        print(
            f"coverage_agent_timeout={perf_counter() - started_at:.2f}s "
            f"after_chat_model_calls={model_calls}"
        )
        return
    except Exception as error:  # noqa: BLE001 - diagnostics must print unexpected Agent errors.
        print(
            f"coverage_agent_error={type(error).__name__}: {error} "
            f"after_chat_model_calls={model_calls}"
        )
        return
    print(
        f"coverage_agent_completed={perf_counter() - started_at:.2f}s "
        f"chat_model_calls={model_calls}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose coverage-assessment latency.")
    parser.add_argument("--thread-id", required=True, help="Existing manual workflow thread_id.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=70.0,
        help="Maximum wait for each independent diagnostic experiment.",
    )
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    asyncio.run(main(args.thread_id, args.timeout_seconds))
