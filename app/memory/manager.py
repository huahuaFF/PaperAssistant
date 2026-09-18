"""Orchestrate conversational-memory read (inject) and write (record)."""

from __future__ import annotations

from app.memory.models import ChatTurn, MemoryContext, SummaryEntry
from app.memory.store import MessageStore, SummaryCache
from app.memory.summarizer import MemorySummarizer


def estimate_tokens(text: str) -> int:
    """Rough token estimate for budget checks (~4 chars per token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


class MemoryManager:
    """Own the memory read path (lazy compaction) and write path (append)."""

    def __init__(
        self,
        store: MessageStore,
        cache: SummaryCache,
        summarizer: MemorySummarizer,
        *,
        tail_keep: int = 4,
        tail_count_limit: int = 8,
        token_limit: int = 2_500,
    ) -> None:
        self._store = store
        self._cache = cache
        self._summarizer = summarizer
        self._tail_keep = tail_keep
        self._tail_count_limit = tail_count_limit
        self._token_limit = token_limit

    async def inject(self, conversation_id: str) -> MemoryContext:
        """Load memory: summary + the recent raw tail, compacting if over budget."""
        entry = self._cache.get(conversation_id)
        summary = entry.summary if entry else ""
        covered = entry.covered_until if entry else 0
        tail = await self._store.list_since(conversation_id, covered)

        while self._over_budget(summary, tail) and len(tail) > self._tail_keep:
            overflow = tail[: len(tail) - self._tail_keep]
            summary = await self._summarizer.summarize(summary, overflow)
            covered += len(overflow)
            tail = tail[len(overflow) :]

        self._cache.set(conversation_id, SummaryEntry(summary=summary, covered_until=covered))
        return MemoryContext(summary=summary, recent_turns=tail)

    async def record(
        self, conversation_id: str, run_id: str, user_query: str, report: str
    ) -> None:
        """Append one turn (user query + assistant report), idempotent by run id."""
        if await self._store.has_run(conversation_id, run_id):
            return
        await self._store.append(conversation_id, run_id, "user", user_query)
        await self._store.append(conversation_id, run_id, "assistant", report)

    def _over_budget(self, summary: str, tail: list[ChatTurn]) -> bool:
        total_tokens = estimate_tokens(summary) + sum(
            estimate_tokens(turn.content) for turn in tail
        )
        return total_tokens > self._token_limit or len(tail) > self._tail_count_limit
