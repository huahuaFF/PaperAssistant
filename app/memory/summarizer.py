"""LLM-backed rolling conversation summarizer."""

from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from app.memory.models import ChatTurn
from app.prompts.summarize_memory import SUMMARIZE_MEMORY_SYSTEM_PROMPT

MAX_SUMMARY_CHARS = 2_000


class MemorySummarizer:
    """Compress older turns plus a previous summary into one bounded summary."""

    def __init__(self, model: Runnable[list[BaseMessage], BaseMessage]) -> None:
        self._model = model

    async def summarize(self, previous_summary: str, turns: list[ChatTurn]) -> str:
        history = "\n".join(f"- {turn.role}: {turn.content}" for turn in turns)
        human = (
            "<previous_summary>\n"
            f"{previous_summary or '无'}\n"
            "</previous_summary>\n\n"
            "<new_turns>\n"
            f"{history}\n"
            "</new_turns>\n\n"
            "请基于以上内容生成更新后的滚动摘要。"
        )
        response = await self._model.ainvoke(
            [
                SystemMessage(content=SUMMARIZE_MEMORY_SYSTEM_PROMPT),
                HumanMessage(content=human),
            ]
        )
        content = response.content
        text = content if isinstance(content, str) else str(content)
        return text[:MAX_SUMMARY_CHARS]
