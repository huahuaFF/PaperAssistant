"""Tests for the LLM-backed memory summarizer."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda

from app.memory.models import ChatTurn
from app.memory.summarizer import MAX_SUMMARY_CHARS, MemorySummarizer


@pytest.mark.asyncio
async def test_summarizer_returns_model_text() -> None:
    model: Runnable[list[BaseMessage], BaseMessage] = RunnableLambda(
        lambda _: AIMessage(content="压缩后的摘要")
    )
    summarizer = MemorySummarizer(model)

    summary = await summarizer.summarize(
        "旧摘要",
        [ChatTurn(role="user", content="找流匹配论文")],
    )

    assert summary == "压缩后的摘要"


@pytest.mark.asyncio
async def test_summarizer_truncates_to_max_chars() -> None:
    long_text = "长" * (MAX_SUMMARY_CHARS + 100)
    model: Runnable[list[BaseMessage], BaseMessage] = RunnableLambda(
        lambda _: AIMessage(content=long_text)
    )
    summarizer = MemorySummarizer(model)

    summary = await summarizer.summarize("", [ChatTurn(role="user", content="q")])

    assert len(summary) == MAX_SUMMARY_CHARS
