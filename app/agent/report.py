"""LangChain Agent factory for structured, source-grounded report drafts."""

from __future__ import annotations

from typing import cast

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from app.models.schemas import GroundedReportDraft
from app.prompts.generate_report import REPORT_PROMPT_VERSION, REPORT_SYSTEM_PROMPT


def build_report_agent(model: ChatOpenAI) -> Runnable[dict[str, object], dict[str, object]]:
    """Create a no-tools Agent constrained to a report draft schema."""
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt=REPORT_SYSTEM_PROMPT,
        response_format=ToolStrategy(GroundedReportDraft, handle_errors=True),
        name="grounded_report_writer",
    )
    configured_agent = agent.with_retry(stop_after_attempt=2).with_config(
        run_name="generate_report",
        tags=["agent", "generate_report", REPORT_PROMPT_VERSION],
        metadata={"prompt_version": REPORT_PROMPT_VERSION},
    )
    return cast(Runnable[dict[str, object], dict[str, object]], configured_agent)
