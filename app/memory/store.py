"""Storage abstractions for conversational memory."""

from __future__ import annotations

from typing import Protocol

from app.memory.models import ChatTurn, SummaryEntry


class MessageStore(Protocol):
    """Source of truth for raw conversation turns (backed by the messages table)."""

    async def append(
        self, conversation_id: str, run_id: str, role: str, content: str
    ) -> None: ...

    async def list_since(self, conversation_id: str, after_index: int) -> list[ChatTurn]: ...

    async def has_run(self, conversation_id: str, run_id: str) -> bool: ...


class SummaryCache(Protocol):
    """Derived, rebuildable summary cache (in-memory now, Redis later)."""

    def get(self, conversation_id: str) -> SummaryEntry | None: ...

    def set(self, conversation_id: str, entry: SummaryEntry) -> None: ...
