"""Tests for the memory manager read/write orchestration."""

from __future__ import annotations

from typing import Literal, cast

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda

from app.memory.cache import InMemorySummaryCache
from app.memory.manager import MemoryManager
from app.memory.models import ChatTurn
from app.memory.store import MessageStore
from app.memory.summarizer import MemorySummarizer


class FakeMessageStore:
    """In-memory MessageStore for deterministic manager tests."""

    def __init__(self, turns: list[ChatTurn] | None = None) -> None:
        self._turns = list(turns or [])
        self._runs: set[str] = set()

    async def append(
        self, conversation_id: str, run_id: str, role: str, content: str
    ) -> None:
        self._runs.add(run_id)
        self._turns.append(ChatTurn(role=cast(Literal["user", "assistant"], role), content=content))

    async def list_since(self, conversation_id: str, after_index: int) -> list[ChatTurn]:
        return self._turns[after_index:]

    async def has_run(self, conversation_id: str, run_id: str) -> bool:
        return run_id in self._runs


def _make_manager(
    store: MessageStore,
    *,
    tail_keep: int = 4,
    tail_count_limit: int = 8,
    token_limit: int = 2_500,
) -> MemoryManager:
    model: Runnable[list[BaseMessage], BaseMessage] = RunnableLambda(
        lambda _: AIMessage(content="COMPACTED")
    )
    return MemoryManager(
        store,
        InMemorySummaryCache(),
        MemorySummarizer(model),
        tail_keep=tail_keep,
        tail_count_limit=tail_count_limit,
        token_limit=token_limit,
    )


@pytest.mark.asyncio
async def test_record_appends_user_and_assistant() -> None:
    store = FakeMessageStore()
    manager = _make_manager(store)

    await manager.record("c1", "run-1", "找流匹配论文", "报告内容")

    turns = await store.list_since("c1", 0)
    assert [turn.role for turn in turns] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_record_is_idempotent_by_run_id() -> None:
    store = FakeMessageStore()
    manager = _make_manager(store)

    await manager.record("c1", "run-1", "q", "a")
    await manager.record("c1", "run-1", "q", "a")

    turns = await store.list_since("c1", 0)
    assert len(turns) == 2


@pytest.mark.asyncio
async def test_inject_returns_empty_on_cold_start() -> None:
    store = FakeMessageStore()
    manager = _make_manager(store)

    context = await manager.inject("c1")

    assert context.summary == ""
    assert context.recent_turns == []


@pytest.mark.asyncio
async def test_inject_returns_tail_without_compaction_when_under_budget() -> None:
    turns = [ChatTurn(role="user", content=f"m{i}") for i in range(4)]
    store = FakeMessageStore(turns)
    manager = _make_manager(store)

    context = await manager.inject("c1")

    assert context.summary == ""
    assert len(context.recent_turns) == 4


@pytest.mark.asyncio
async def test_inject_compacts_when_tail_exceeds_count_limit() -> None:
    turns = [ChatTurn(role="user", content=f"m{i}") for i in range(9)]
    store = FakeMessageStore(turns)
    manager = _make_manager(store, tail_keep=4, tail_count_limit=8)

    context = await manager.inject("c1")

    assert context.summary == "COMPACTED"
    assert len(context.recent_turns) == 4
    assert context.recent_turns[0].content == "m5"


@pytest.mark.asyncio
async def test_inject_compacts_when_token_limit_exceeded() -> None:
    turns = [ChatTurn(role="user", content="x" * 40) for _ in range(4)]
    store = FakeMessageStore(turns)
    manager = _make_manager(store, tail_keep=2, tail_count_limit=8, token_limit=15)

    context = await manager.inject("c1")

    assert context.summary == "COMPACTED"
    assert len(context.recent_turns) == 2
