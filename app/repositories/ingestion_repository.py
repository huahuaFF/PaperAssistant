"""SQLite persistence for ingestion attempts and imported-paper metadata."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import Database
from app.models.entities import IngestionTask, Paper, PaperVersion


class SQLiteIngestionRepository:
    """Record import progress separately from LangGraph's checkpoint state."""

    def __init__(self, database: Database) -> None:
        self._database = database
        self._initialized = False

    async def start(self, *, run_id: str, source_key: str, source_url: str) -> str:
        await self._ensure_initialized()
        task = IngestionTask(run_id=run_id, source_key=source_key, source_url=source_url)
        async with self._database.session() as session:
            session.add(task)
            await session.commit()
            await session.refresh(task)
        return task.id

    async def complete(
        self,
        *,
        task_id: str,
        paper_id: str,
        title: str,
        arxiv_id: str | None,
        source_url: str,
        version_label: str,
        file_hash: str,
    ) -> None:
        await self._ensure_initialized()
        async with self._database.session() as session:
            task = await session.get(IngestionTask, task_id)
            if task is None:
                raise LookupError(f"Unknown ingestion task: {task_id}")
            paper = await _find_paper(session, arxiv_id)
            if paper is None:
                paper = Paper(id=paper_id, title=title, arxiv_id=arxiv_id)
                session.add(paper)
            else:
                paper.title = title
            version = await session.scalar(
                select(PaperVersion).where(
                    PaperVersion.paper_id == paper.id, PaperVersion.file_hash == file_hash
                )
            )
            if version is None:
                session.add(
                    PaperVersion(
                        paper_id=paper.id,
                        version_label=version_label,
                        source_url=source_url,
                        file_hash=file_hash,
                        ingestion_status="completed",
                    )
                )
            else:
                version.ingestion_status = "completed"
            task.paper_id = paper.id
            task.status = "completed"
            task.error_message = None
            task.completed_at = datetime.now(UTC)
            await session.commit()

    async def fail(self, *, task_id: str, error_message: str) -> None:
        await self._ensure_initialized()
        async with self._database.session() as session:
            task = await session.get(IngestionTask, task_id)
            if task is None:
                raise LookupError(f"Unknown ingestion task: {task_id}")
            task.status = "failed"
            task.error_message = error_message[:8_000]
            task.completed_at = datetime.now(UTC)
            await session.commit()

    async def _ensure_initialized(self) -> None:
        if not self._initialized:
            await self._database.initialize()
            self._initialized = True


async def _find_paper(session: AsyncSession, arxiv_id: str | None) -> Paper | None:
    if arxiv_id is None:
        return None
    scalar = await session.scalar(select(Paper).where(Paper.arxiv_id == arxiv_id))
    return scalar if isinstance(scalar, Paper) else None
