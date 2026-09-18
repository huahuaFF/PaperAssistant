"""arxiv.py-backed search adapter exposed through a LangChain tool."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import arxiv
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.models.schemas import ArxivCandidate, ArxivSearchPlan

ARXIV_ID_PATTERN = re.compile(
    r"(?P<identifier>(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7}))(?:v\d+)?$", re.IGNORECASE
)
SAFE_CATEGORY = re.compile(r"^[a-z-]+\.[A-Za-z-]+$")


class ArxivSearchError(RuntimeError):
    """Raised when arxiv.py cannot return a valid result set."""


class ArxivSearchToolInput(BaseModel):
    """Tool contract: semantic keywords, never raw arXiv API syntax."""

    keywords: list[str] = Field(min_length=1, max_length=6)
    categories: list[str] = Field(default_factory=list, max_length=5)
    max_results: int = Field(ge=1, le=30)


class ArxivLibraryClient(Protocol):
    """The small synchronous arxiv.py surface needed by this adapter."""

    def results(self, search: arxiv.Search) -> Iterator[arxiv.Result]: ...


@dataclass(frozen=True)
class ArxivSearchResult:
    query: str
    candidates: list[ArxivCandidate]
    total_results: int | None = None


class ArxivSearchClient:
    """Async boundary over the maintained ``arxiv`` Python client library."""

    def __init__(self, *, client: ArxivLibraryClient | None = None) -> None:
        self._client = client or arxiv.Client(page_size=30, delay_seconds=3.0, num_retries=2)

    async def search(self, plan: ArxivSearchPlan) -> ArxivSearchResult:
        query = build_arxiv_library_query(plan)
        search = arxiv.Search(
            query=query,
            max_results=plan.max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        try:
            results = await asyncio.to_thread(_collect_results, self._client, search, plan.max_results)
        except Exception as error:  # arxiv.py owns transport, pagination, and its exception types.
            raise ArxivSearchError("arXiv search through arxiv.py failed.") from error
        return ArxivSearchResult(query=query, candidates=[_to_candidate(result) for result in results])


def build_arxiv_library_query(plan: ArxivSearchPlan) -> str:
    """Build a small, validated API expression without exposing DSL generation to the Agent."""
    phrases = [_normalized_phrase(keyword) for keyword in plan.keywords]
    keyword_clause = " OR ".join(f'all:"{phrase}"' for phrase in dict.fromkeys(phrases))
    categories = [category.strip() for category in plan.categories if category.strip()]
    if not categories:
        return f"({keyword_clause})"
    invalid_categories = [category for category in categories if SAFE_CATEGORY.fullmatch(category) is None]
    if invalid_categories:
        raise ValueError(f"Invalid arXiv categories: {', '.join(invalid_categories)}")
    category_clause = " OR ".join(f"cat:{category}" for category in dict.fromkeys(categories))
    return f"({keyword_clause}) AND ({category_clause})"


def create_search_arxiv_tool(client: ArxivSearchClient) -> BaseTool:
    """Expose arxiv.py search through the existing LangChain Tool boundary."""

    @tool("search_arxiv", args_schema=ArxivSearchToolInput)
    async def search_arxiv(
        keywords: list[str], categories: list[str], max_results: int
    ) -> list[dict[str, Any]]:
        """Search arXiv by semantic keywords and return normalized paper metadata."""
        plan = ArxivSearchPlan(
            keywords=keywords,
            rationale="Workflow-approved arXiv search.",
            categories=categories,
            max_results=max_results,
        )
        result = await client.search(plan)
        return [candidate.model_dump(mode="json") for candidate in result.candidates]

    return search_arxiv


def _normalized_phrase(value: str) -> str:
    phrase = " ".join(value.split())
    if not phrase:
        raise ValueError("arXiv search keywords cannot be blank.")
    if '"' in phrase or "\\" in phrase:
        raise ValueError("arXiv search keywords cannot contain quotes or backslashes.")
    return phrase


def _collect_results(
    client: ArxivLibraryClient, search: arxiv.Search, max_results: int
) -> list[arxiv.Result]:
    return list(client.results(search))[:max_results]


def _to_candidate(result: arxiv.Result) -> ArxivCandidate:
    short_id = result.get_short_id()
    match = ARXIV_ID_PATTERN.fullmatch(short_id)
    if match is None:
        raise ArxivSearchError(f"arxiv.py returned an unrecognized paper identifier: {short_id}")
    arxiv_id = match.group("identifier")
    return ArxivCandidate(
        arxiv_id=arxiv_id,
        title=" ".join(result.title.split()),
        authors=[author.name for author in result.authors],
        abstract=" ".join(result.summary.split()),
        categories=list(result.categories),
        published_at=_as_datetime(result.published),
        updated_at=_as_datetime(result.updated),
        abs_url=result.entry_id,
        pdf_url=result.pdf_url or f"https://arxiv.org/pdf/{short_id}",
    )


def _as_datetime(value: datetime) -> datetime:
    return value
