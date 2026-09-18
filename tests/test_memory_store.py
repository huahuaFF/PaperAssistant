"""Tests for the SQLite message store and in-memory summary cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.db import Database
from app.memory.cache import InMemorySummaryCache
from app.memory.models import SummaryEntry
from app.memory.sqlite_store import SQLiteMessageStore
from app.models.entities import Conversation, ResearchRun


async def _seed_parents(database: Database, conversation_id: str, run_id: str) -> None:
    await database.initialize()
    async with database.session() as session:
        session.add(Conversation(id=conversation_id))
        session.add(
            ResearchRun(
                id=run_id,
                conversation_id=conversation_id,
                graph_thread_id=f"thread-{run_id}",
                user_query="seed query",
            )
        )
        await session.commit()


@pytest.mark.asyncio
async def test_append_and_list_since_preserve_order(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    store = SQLiteMessageStore(database)
    await _seed_parents(database, "conversation-1", "run-1")

    await store.append("conversation-1", "run-1", "user", "帮我找流匹配论文")
    await store.append("conversation-1", "run-1", "assistant", "已生成报告")

    turns = await store.list_since("conversation-1", 0)
    await database.dispose()

    assert [turn.role for turn in turns] == ["user", "assistant"]
    assert turns[0].content == "帮我找流匹配论文"


@pytest.mark.asyncio
async def test_list_since_skips_covered_messages(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    store = SQLiteMessageStore(database)
    await _seed_parents(database, "conversation-1", "run-1")

    await store.append("conversation-1", "run-1", "user", "u1")
    await store.append("conversation-1", "run-1", "assistant", "a1")
    await store.append("conversation-1", "run-1", "user", "u2")

    turns = await store.list_since("conversation-1", 2)
    await database.dispose()

    assert [turn.content for turn in turns] == ["u2"]


@pytest.mark.asyncio
async def test_has_run_guards_idempotency(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    store = SQLiteMessageStore(database)
    await _seed_parents(database, "conversation-1", "run-1")

    await store.append("conversation-1", "run-1", "user", "u")

    assert await store.has_run("conversation-1", "run-1") is True
    assert await store.has_run("conversation-1", "run-2") is False
    await database.dispose()


def test_in_memory_cache_roundtrip() -> None:
    cache = InMemorySummaryCache()
    assert cache.get("conversation-1") is None

    entry = SummaryEntry(summary="摘要", covered_until=3)
    cache.set("conversation-1", entry)

    assert cache.get("conversation-1") == entry
