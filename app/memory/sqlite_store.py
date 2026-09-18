"""SQLite-backed message store."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select

from app.db import Database
from app.memory.models import ChatTurn
from app.models.entities import Message


class SQLiteMessageStore:
    """Persist raw conversation turns in the ``messages`` table."""

    def __init__(self, database: Database) -> None:
        self._database = database
        self._initialized = False

    async def append(
        self, conversation_id: str, run_id: str, role: str, content: str
    ) -> None:
        await self._ensure_initialized()
        message = Message(
            conversation_id=conversation_id,
            run_id=run_id,
            role=role,
            content=content,
            created_at=datetime.now(UTC),
        )
        async with self._database.session() as session:
            session.add(message)
            await session.commit()

    async def list_since(self, conversation_id: str, after_index: int) -> list[ChatTurn]:
        await self._ensure_initialized()
        async with self._database.session() as session:
            result = await session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at, Message.id)
                .offset(after_index)
            )
            messages = list(result.all())
        return [_to_chat_turn(message) for message in messages]

    async def has_run(self, conversation_id: str, run_id: str) -> bool:
        await self._ensure_initialized()
        async with self._database.session() as session:
            result = await session.scalars(
                select(Message.id)
                .where(
                    Message.conversation_id == conversation_id,
                    Message.run_id == run_id,
                )
                .limit(1)
            )
            return result.first() is not None

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self._database.initialize()
            self._initialized = True


def _to_chat_turn(message: Message) -> ChatTurn:
    if message.role == "user":
        return ChatTurn(role="user", content=message.content)
    if message.role == "assistant":
        return ChatTurn(role="assistant", content=message.content)
    raise ValueError(f"Unknown message role: {message.role}")
