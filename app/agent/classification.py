"""LangChain Agent factory for classify-query structured output."""

from __future__ import annotations

from typing import cast

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from app.models.schemas import QueryIntent
from app.prompts.classify_query import CLASSIFY_QUERY_PROMPT_VERSION, CLASSIFY_QUERY_SYSTEM_PROMPT


def build_classification_agent(model: ChatOpenAI) -> Runnable[dict[str, object], dict[str, object]]:
    """Create a bounded LangChain Agent that returns ``QueryIntent`` as state.

    ``ToolStrategy`` forces schema-as-tool output, which matches MiniMax's
    documented tool-calling capability without granting external tools.
    """
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt=CLASSIFY_QUERY_SYSTEM_PROMPT,
        response_format=ToolStrategy(QueryIntent, handle_errors=True),
        name="query_classifier",
    )
    configured_agent = agent.with_retry(stop_after_attempt=2).with_config(
        run_name="classify_query",
        tags=["agent", "classify_query", CLASSIFY_QUERY_PROMPT_VERSION],
        metadata={"prompt_version": CLASSIFY_QUERY_PROMPT_VERSION},
    )
    return cast(Runnable[dict[str, object], dict[str, object]], configured_agent)
