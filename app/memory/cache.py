"""In-process summary cache."""

from __future__ import annotations

from app.memory.models import SummaryEntry


class InMemorySummaryCache:
    """Store one rolling summary per conversation in process memory."""

    def __init__(self) -> None:
        self._entries: dict[str, SummaryEntry] = {}

    def get(self, conversation_id: str) -> SummaryEntry | None:
        return self._entries.get(conversation_id)

    def set(self, conversation_id: str, entry: SummaryEntry) -> None:
        self._entries[conversation_id] = entry
