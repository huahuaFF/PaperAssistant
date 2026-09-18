from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from sqlalchemy import select

from app.db import Database
from app.graph.nodes.ingest_papers import create_ingest_papers_node
from app.models.entities import IngestionTask, Paper, PaperVersion
from app.models.schemas import ImportApprovalCandidate
from app.repositories.ingestion_repository import SQLiteIngestionRepository
from app.services.paper_indexing import IndexingResult
from app.services.paper_ingestion import (
    DownloadedPaper,
    IngestionOutcome,
    PaperDownloader,
    PaperIngestionCoordinator,
    UnsupportedImportSourceError,
)
from app.services.paper_parsing import PaperChunk, ParsedPaper


def arxiv_candidate() -> ImportApprovalCandidate:
    return ImportApprovalCandidate(
        paper_id="arxiv:2210.02747",
        source_type="arxiv",
        source_url="https://arxiv.org/pdf/2210.02747",
        title="Flow Matching",
    )


@pytest.mark.asyncio
async def test_downloader_writes_arxiv_pdf_atomically_and_reuses_valid_file(tmp_path: Path) -> None:
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        assert request.url == "https://arxiv.org/pdf/2210.02747"
        return httpx.Response(200, content=b"%PDF-1.7\nminimal test PDF")

    downloader = PaperDownloader(tmp_path, transport=httpx.MockTransport(handler))
    first = await downloader.download(arxiv_candidate())
    second = await downloader.download(arxiv_candidate())

    assert first.path == tmp_path / "arxiv-2210-02747.pdf"
    assert second.path == first.path
    assert first.path.read_bytes().startswith(b"%PDF-")
    assert request_count == 1
    assert not (tmp_path / "arxiv-2210-02747.pdf.part").exists()


@pytest.mark.asyncio
async def test_downloader_rejects_unimplemented_source_type(tmp_path: Path) -> None:
    downloader = PaperDownloader(tmp_path)
    candidate = arxiv_candidate().model_copy(update={"paper_id": "doi:10.1000/example", "source_type": "doi"})

    with pytest.raises(UnsupportedImportSourceError, match="not implemented"):
        await downloader.download(candidate)


class FakeDownloader:
    async def download(self, candidate: ImportApprovalCandidate) -> DownloadedPaper:
        return DownloadedPaper(path=Path("unused.pdf"), source_url=candidate.source_url)


class FakeParser:
    def parse(self, source: Any) -> ParsedPaper:
        return ParsedPaper(
            paper_id=source.paper_id,
            source_path=str(source.local_path),
            file_sha256="a" * 64,
            title="Parsed Flow Matching",
            page_count=1,
            chunks=[
                PaperChunk(
                    chunk_id=f"{source.paper_id}:section_token_v2:p0001:body:c0001",
                    paper_id=source.paper_id,
                    title="Parsed Flow Matching",
                    page_number=1,
                    chunk_index=1,
                    source_type="arxiv",
                    arxiv_id=source.arxiv_id,
                    text="Enough parsed source text for the fake index.",
                )
            ],
        )


class FakeIndexer:
    def index(self, parsed: ParsedPaper) -> IndexingResult:
        return IndexingResult(
            paper_id=parsed.paper_id,
            requested_chunk_count=len(parsed.chunks),
            upserted_chunk_ids=[chunk.chunk_id for chunk in parsed.chunks],
        )


class FakeRecords:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def start(self, **_: str) -> str:
        self.events.append("started")
        return "task-1"

    async def complete(self, **_: str | None) -> None:
        self.events.append("completed")

    async def fail(self, **_: str) -> None:
        self.events.append("failed")


@pytest.mark.asyncio
async def test_coordinator_parses_indexes_and_records_completion(tmp_path: Path) -> None:
    records = FakeRecords()
    coordinator = PaperIngestionCoordinator(
        downloader=cast(Any, FakeDownloader()),
        parser=cast(Any, FakeParser()),
        indexer=cast(Any, FakeIndexer()),
        records=records,
        parsed_directory=tmp_path,
    )

    outcome = await coordinator.ingest(run_id="run-1", candidate=arxiv_candidate())

    assert outcome.paper_id == "arxiv-2210-02747"
    assert outcome.chunk_count == 1
    assert outcome.parsed_path.is_file()
    assert records.events == ["started", "completed"]


@pytest.mark.asyncio
async def test_ingest_node_only_uses_human_selected_candidates() -> None:
    class FakeCoordinator:
        async def ingest(self, *, run_id: str, candidate: ImportApprovalCandidate) -> IngestionOutcome:
            assert run_id == "run-1"
            assert candidate.paper_id == "arxiv:2210.02747"
            return IngestionOutcome(
                paper_id="arxiv-2210-02747",
                arxiv_id="2210.02747",
                title="Flow Matching",
                parsed_path=Path("data/parsed/paper.json"),
                chunk_count=2,
                indexed_chunk_ids=["chunk-1", "chunk-2"],
            )

    node = create_ingest_papers_node(cast(Any, FakeCoordinator()))
    update = await node(
        cast(
            Any,
            {
                "run_id": "run-1",
                "conversation_id": "conversation-1",
                "workspace_id": "default",
                "user_query": "导入这篇流匹配论文",
                "status": "import_selected",
                "import_approval": "selected",
                "selected_paper_ids": ["arxiv:2210.02747"],
                "arxiv_candidates": [
                    {
                        "arxiv_id": "2210.02747",
                        "title": "Flow Matching",
                        "authors": ["Ada Lovelace"],
                        "abstract": "A paper.",
                        "categories": ["cs.LG"],
                        "published_at": "2022-10-06T17:58:00Z",
                        "updated_at": "2022-10-06T17:58:00Z",
                        "abs_url": "https://arxiv.org/abs/2210.02747",
                        "pdf_url": "https://arxiv.org/pdf/2210.02747",
                    }
                ],
            },
        )
    )

    assert update["imported_paper_ids"] == ["arxiv-2210-02747"]
    assert update["ingestion_results"][0]["chunk_count"] == 2


@pytest.mark.asyncio
async def test_sqlite_repository_records_completed_task_and_reuses_paper(tmp_path: Path) -> None:
    database = Database(tmp_path / "app.db")
    repository = SQLiteIngestionRepository(database)
    task_id = await repository.start(
        run_id="run-1", source_key="arxiv:2210.02747", source_url="https://arxiv.org/pdf/2210.02747"
    )
    await repository.complete(
        task_id=task_id,
        paper_id="arxiv-2210-02747",
        title="Flow Matching",
        arxiv_id="2210.02747",
        source_url="https://arxiv.org/pdf/2210.02747",
        version_label="2210.02747",
        file_hash="a" * 64,
    )
    second_task_id = await repository.start(
        run_id="run-2", source_key="arxiv:2210.02747", source_url="https://arxiv.org/pdf/2210.02747"
    )
    await repository.complete(
        task_id=second_task_id,
        paper_id="arxiv-2210-02747",
        title="Flow Matching (updated title)",
        arxiv_id="2210.02747",
        source_url="https://arxiv.org/pdf/2210.02747",
        version_label="2210.02747",
        file_hash="a" * 64,
    )

    async with database.session() as session:
        tasks = list((await session.scalars(select(IngestionTask))).all())
        papers = list((await session.scalars(select(Paper))).all())
        versions = list((await session.scalars(select(PaperVersion))).all())

    await database.dispose()
    assert [task.status for task in tasks] == ["completed", "completed"]
    assert len(papers) == 1
    assert papers[0].title == "Flow Matching (updated title)"
    assert len(versions) == 1
