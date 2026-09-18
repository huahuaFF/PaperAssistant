"""Download, parse, index, and record approved arXiv papers as one idempotent unit."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import httpx

from app.models.schemas import ImportApprovalCandidate
from app.services.paper_indexing import IndexingResult, PaperIndexer
from app.services.paper_parsing import PaperParser, PaperSource, ParsedPaper, write_parsed_paper

MAX_PDF_BYTES = 60 * 1024 * 1024


class IngestionError(RuntimeError):
    """Raised when an approved source cannot be safely made searchable."""


class UnsupportedImportSourceError(IngestionError):
    """Raised for source types that do not yet have a trustworthy PDF resolver."""


@dataclass(frozen=True)
class DownloadedPaper:
    path: Path
    source_url: str


@dataclass(frozen=True)
class IngestionOutcome:
    paper_id: str
    arxiv_id: str
    title: str
    parsed_path: Path
    chunk_count: int
    indexed_chunk_ids: list[str]

    def to_state(self) -> dict[str, object]:
        return {
            "paper_id": self.paper_id,
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "parsed_path": str(self.parsed_path),
            "chunk_count": self.chunk_count,
            "indexed_chunk_ids": self.indexed_chunk_ids,
        }


class IngestionRecordStore(Protocol):
    async def start(self, *, run_id: str, source_key: str, source_url: str) -> str: ...

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
    ) -> None: ...

    async def fail(self, *, task_id: str, error_message: str) -> None: ...


class PaperDownloader:
    """Download arXiv PDFs atomically, with a fixed upper bound on file size."""

    def __init__(
        self,
        storage_directory: Path,
        *,
        max_bytes: int = MAX_PDF_BYTES,
        timeout_seconds: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._storage_directory = storage_directory
        self._max_bytes = max_bytes
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    async def download(self, candidate: ImportApprovalCandidate) -> DownloadedPaper:
        if candidate.source_type != "arxiv":
            raise UnsupportedImportSourceError(
                f"Automatic PDF resolution for '{candidate.source_type}' is not implemented."
            )
        arxiv_id = _arxiv_id(candidate.paper_id)
        destination = self._storage_directory / f"arxiv-{_safe_filename(arxiv_id)}.pdf"
        self._storage_directory.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and _looks_like_pdf(destination):
            return DownloadedPaper(path=destination, source_url=_arxiv_pdf_url(arxiv_id))

        temporary = destination.with_suffix(".pdf.part")
        temporary.unlink(missing_ok=True)
        source_url = _arxiv_pdf_url(arxiv_id)
        try:
            async with (
                httpx.AsyncClient(
                    timeout=self._timeout_seconds,
                    follow_redirects=True,
                    transport=self._transport,
                    headers={"User-Agent": "PaperAssistant/0.1 (research workflow)"},
                ) as client,
                client.stream("GET", source_url) as response,
            ):
                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > self._max_bytes:
                    raise IngestionError("PDF exceeds the configured download size limit.")
                received = 0
                with temporary.open("wb") as stream:
                    async for block in response.aiter_bytes():
                        received += len(block)
                        if received > self._max_bytes:
                            raise IngestionError("PDF exceeds the configured download size limit.")
                        stream.write(block)
            if not _looks_like_pdf(temporary):
                raise IngestionError("Downloaded arXiv content is not a PDF.")
            temporary.replace(destination)
        except IngestionError:
            temporary.unlink(missing_ok=True)
            raise
        except (httpx.HTTPError, OSError, ValueError) as error:
            temporary.unlink(missing_ok=True)
            raise IngestionError("Unable to download arXiv PDF.") from error
        return DownloadedPaper(path=destination, source_url=source_url)


class PaperIngestionCoordinator:
    """Run approved arXiv ingestion sequentially so retries remain idempotent."""

    def __init__(
        self,
        *,
        downloader: PaperDownloader,
        parser: PaperParser,
        indexer: PaperIndexer,
        records: IngestionRecordStore,
        parsed_directory: Path,
    ) -> None:
        self._downloader = downloader
        self._parser = parser
        self._indexer = indexer
        self._records = records
        self._parsed_directory = parsed_directory

    async def ingest(self, *, run_id: str, candidate: ImportApprovalCandidate) -> IngestionOutcome:
        arxiv_id = _arxiv_id(candidate.paper_id)
        source_url = _arxiv_pdf_url(arxiv_id)
        task_id = await self._records.start(
            run_id=run_id, source_key=candidate.paper_id, source_url=source_url
        )
        try:
            downloaded = await self._downloader.download(candidate)
            paper_id = _stable_paper_id(arxiv_id)
            source = PaperSource(
                paper_id=paper_id,
                local_path=downloaded.path,
                source_type="arxiv",
                arxiv_id=arxiv_id,
                version_label=arxiv_id,
            )
            parsed = await asyncio.to_thread(self._parser.parse, source)
            parsed_path = await asyncio.to_thread(write_parsed_paper, parsed, self._parsed_directory)
            indexing = await asyncio.to_thread(self._indexer.index, parsed)
            await self._records.complete(
                task_id=task_id,
                paper_id=paper_id,
                title=parsed.title,
                arxiv_id=arxiv_id,
                source_url=downloaded.source_url,
                version_label=arxiv_id,
                file_hash=parsed.file_sha256,
            )
            return _outcome(parsed, parsed_path, indexing, arxiv_id)
        except Exception as error:
            await self._records.fail(task_id=task_id, error_message=str(error))
            raise


def _outcome(
    parsed: ParsedPaper, parsed_path: Path, indexing: IndexingResult, arxiv_id: str
) -> IngestionOutcome:
    return IngestionOutcome(
        paper_id=parsed.paper_id,
        arxiv_id=arxiv_id,
        title=parsed.title,
        parsed_path=parsed_path,
        chunk_count=len(parsed.chunks),
        indexed_chunk_ids=indexing.upserted_chunk_ids,
    )


def _arxiv_id(paper_id: str) -> str:
    prefix, separator, identifier = paper_id.partition(":")
    if prefix != "arxiv" or not separator or not identifier:
        raise UnsupportedImportSourceError(f"Expected an approved arXiv paper ID, got: {paper_id}")
    return identifier


def _arxiv_pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}"


def _stable_paper_id(arxiv_id: str) -> str:
    return f"arxiv-{_safe_filename(arxiv_id)}"


def _safe_filename(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-")
    if not normalized:
        raise IngestionError("Cannot derive a safe local file name from arXiv ID.")
    return normalized


def _looks_like_pdf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 5:
        return False
    with path.open("rb") as stream:
        return stream.read(5) == b"%PDF-"
