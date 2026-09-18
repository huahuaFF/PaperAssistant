"""Domain types for conversational memory."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    """A single raw message in the conversation."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class SummaryEntry(BaseModel):
    """A cached rolling summary plus the number of messages it already covers."""

    summary: str
    covered_until: int = Field(ge=0)


class MemoryContext(BaseModel):
    """The bounded memory slice injected into a prompt."""

    summary: str
    recent_turns: list[ChatTurn]
