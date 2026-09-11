"""Durable SQLite checkpointer lifecycle for interrupt-enabled LangGraph runs."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import Settings


@asynccontextmanager
async def open_sqlite_checkpointer(settings: Settings) -> AsyncIterator[AsyncSqliteSaver]:
    """Open and initialize the durable checkpointer used by approval interrupts."""
    settings.ensure_local_directories()
    async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_path)) as checkpointer:
        await checkpointer.setup()
        yield checkpointer
