"""LangChain Agent factory for structured arXiv search planning."""

from __future__ import annotations

from typing import cast

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from app.models.schemas import ArxivSearchPlan
from app.prompts.build_arxiv_query import ARXIV_QUERY_PROMPT_VERSION, ARXIV_QUERY_SYSTEM_PROMPT


def build_arxiv_query_agent(model: ChatOpenAI) -> Runnable[dict[str, object], dict[str, object]]:
    """Create a bounded Agent that plans but never executes an arXiv search."""
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt=ARXIV_QUERY_SYSTEM_PROMPT,
        response_format=ToolStrategy(ArxivSearchPlan, handle_errors=True),
        name="arxiv_query_planner",
    )
    configured_agent = agent.with_retry(stop_after_attempt=2).with_config(
        run_name="build_arxiv_query",
        tags=["agent", "build_arxiv_query", ARXIV_QUERY_PROMPT_VERSION],
        metadata={"prompt_version": ARXIV_QUERY_PROMPT_VERSION},
    )
    return cast(Runnable[dict[str, object], dict[str, object]], configured_agent)
