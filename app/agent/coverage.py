"""LangChain Agent factory for structured local-evidence coverage assessment."""

from __future__ import annotations

from typing import cast

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from app.models.schemas import CoverageAssessment
from app.prompts.assess_coverage import COVERAGE_PROMPT_VERSION, COVERAGE_SYSTEM_PROMPT


def build_coverage_agent(model: ChatOpenAI) -> Runnable[dict[str, object], dict[str, object]]:
    """Create a bounded Agent that returns only a coverage decision."""
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt=COVERAGE_SYSTEM_PROMPT,
        response_format=ToolStrategy(CoverageAssessment, handle_errors=True),
        name="coverage_assessor",
    )
    configured_agent = agent.with_retry(stop_after_attempt=2).with_config(
        run_name="assess_coverage",
        tags=["agent", "assess_coverage", COVERAGE_PROMPT_VERSION],
        metadata={"prompt_version": COVERAGE_PROMPT_VERSION},
    )
    return cast(Runnable[dict[str, object], dict[str, object]], configured_agent)
