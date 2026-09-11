"""Bounded, serializable conversation context for Agent nodes."""

from __future__ import annotations

from html import escape
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from app.graph.state import ResearchState

MAX_QUERY_CHARS = 4_000
MAX_SUMMARY_CHARS = 2_000
MAX_ACTIVE_PAPER_ABSTRACT_CHARS = 500
MAX_ACTIVE_PAPERS = 5


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ActivePaperReference(BaseModel):
    paper_id: str
    title: str
    arxiv_id: str | None = None
    abstract: str | None = None


class ClassificationContext(BaseModel):
    query: str
    conversation_summary: str
    chat_history: list[BaseMessage]
    active_papers: str

    model_config = {"arbitrary_types_allowed": True}

    def prompt_values(self) -> dict[str, object]:
        return {
            "query": self.query,
            "conversation_summary": self.conversation_summary,
            "chat_history": self.chat_history,
            "active_papers": self.active_papers,
        }


class ClassificationContextBuilder:
    """Apply the research-assistant context budget before rendering a prompt."""

    def build(self, state: ResearchState) -> ClassificationContext:
        turns = [ChatTurn.model_validate(item) for item in state.get("chat_history", [])]
        active_papers = [
            ActivePaperReference.model_validate(item) for item in state.get("active_papers", [])
        ]
        return ClassificationContext(
            query=_truncate(state["user_query"], MAX_QUERY_CHARS),
            conversation_summary=_truncate(state.get("conversation_summary", "无"), MAX_SUMMARY_CHARS),
            chat_history=[_to_message(turn) for turn in turns[-4:]],
            active_papers=_render_active_papers(active_papers[:MAX_ACTIVE_PAPERS]),
        )


def _to_message(turn: ChatTurn) -> BaseMessage:
    if turn.role == "user":
        return HumanMessage(content=turn.content)
    return AIMessage(content=turn.content)


def _render_active_papers(papers: list[ActivePaperReference]) -> str:
    if not papers:
        return "无"

    entries: list[str] = []
    for paper in papers:
        abstract = _truncate(paper.abstract or "", MAX_ACTIVE_PAPER_ABSTRACT_CHARS)
        entries.append(
            "\n".join(
                (
                    f"- paper_id: {escape(paper.paper_id)}",
                    f"  title: {escape(paper.title)}",
                    f"  arxiv_id: {escape(paper.arxiv_id or '未知')}",
                    f"  abstract: {escape(abstract or '无')}",
                )
            )
        )
    return "\n".join(entries)


def _truncate(value: str, max_chars: int) -> str:
    return value if len(value) <= max_chars else f"{value[:max_chars]}…"
