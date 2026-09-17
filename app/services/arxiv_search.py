"""Typed arXiv API client and LangChain tool used by the search workflow node."""

from __future__ import annotations

import asyncio
import re
import xml.etree.ElementTree as element_tree
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

from app.models.schemas import ArxivCandidate, ArxivSearchPlan

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NAMESPACE = "{http://www.w3.org/2005/Atom}"
ARXIV_NAMESPACE = "{http://arxiv.org/schemas/atom}"
ARXIV_ID_PATTERN = re.compile(
    r"(?:abs/)?(?P<identifier>(?:\d{4}\.\d{4,5}|[a-z-]+/\d{7}))(?:v\d+)?/?$",
    re.IGNORECASE,
)


class ArxivSearchError(RuntimeError):
    """Raised when arXiv cannot provide a valid search response."""


class ArxivSearchToolInput(BaseModel):
    """Explicit, bounded input contract for the deterministic LangChain tool."""

    query: str = Field(min_length=1, max_length=1_000)
    categories: list[str] = Field(default_factory=list, max_length=5)
    max_results: int = Field(ge=1, le=30)


@dataclass(frozen=True)
class ArxivSearchResult:
    query: str
    candidates: list[ArxivCandidate]
    total_results: int | None


class ArxivSearchClient:
    """Small async client for arXiv's Atom API, isolated from graph concerns."""

    def __init__(
        self,
        *,
        endpoint: str = ARXIV_API_URL,
        timeout_seconds: float = 15.0,
        retry_attempts: int = 2,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._timeout_seconds = timeout_seconds
        self._retry_attempts = retry_attempts
        self._transport = transport

    async def search(self, plan: ArxivSearchPlan) -> ArxivSearchResult:
        query = build_arxiv_api_query(plan)
        response = await self._request(query, plan.max_results)
        return parse_arxiv_response(response.text, query=query)

    async def _request(self, query: str, max_results: int) -> httpx.Response:
        params: Mapping[str, str | int] = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        async with httpx.AsyncClient(
            timeout=self._timeout_seconds,
            transport=self._transport,
            headers={"User-Agent": "PaperAssistant/0.1 (research workflow)"},
        ) as client:
            for attempt in range(self._retry_attempts + 1):
                try:
                    response = await client.get(self._endpoint, params=params)
                    response.raise_for_status()
                    return response
                except (httpx.TransportError, httpx.HTTPStatusError) as error:
                    should_retry = attempt < self._retry_attempts and _is_retryable(error)
                    if not should_retry:
                        raise ArxivSearchError("arXiv search request failed.") from error
                    await asyncio.sleep(0.25 * (attempt + 1))
        raise AssertionError("The retry loop must either return or raise.")


def build_arxiv_api_query(plan: ArxivSearchPlan) -> str:
    """Combine semantic query and category filters without giving them LLM control."""
    categories = [category.strip() for category in plan.categories if category.strip()]
    if not categories:
        return plan.query.strip()
    category_clause = " OR ".join(f"cat:{category}" for category in dict.fromkeys(categories))
    return f"({plan.query.strip()}) AND ({category_clause})"


def parse_arxiv_response(xml_text: str, *, query: str) -> ArxivSearchResult:
    """Parse the documented Atom feed into provenance-preserving candidate objects."""
    try:
        root = element_tree.fromstring(xml_text)
    except element_tree.ParseError as error:
        raise ArxivSearchError("arXiv returned invalid Atom XML.") from error

    total_results_text = _text(root, f"{ARXIV_NAMESPACE}totalResults", required=False)
    total_results = int(total_results_text) if total_results_text and total_results_text.isdigit() else None
    candidates = [_parse_entry(entry) for entry in root.findall(f"{ATOM_NAMESPACE}entry")]
    return ArxivSearchResult(query=query, candidates=candidates, total_results=total_results)


def create_search_arxiv_tool(client: ArxivSearchClient) -> BaseTool:
    """Expose the official-API client through LangChain's standard tool interface."""

    @tool("search_arxiv", args_schema=ArxivSearchToolInput)
    async def search_arxiv(
        query: str, categories: list[str], max_results: int
    ) -> list[dict[str, Any]]:
        """Search arXiv and return normalized candidate-paper metadata."""
        plan = ArxivSearchPlan(
            query=query,
            rationale="Workflow-approved arXiv search.",
            categories=categories,
            max_results=max_results,
        )
        result = await client.search(plan)
        return [candidate.model_dump(mode="json") for candidate in result.candidates]

    return search_arxiv


def _parse_entry(entry: element_tree.Element) -> ArxivCandidate:
    raw_identifier = _text(entry, f"{ATOM_NAMESPACE}id")
    match = ARXIV_ID_PATTERN.search(raw_identifier)
    if match is None:
        raise ArxivSearchError("arXiv response contains an unrecognized paper identifier.")
    arxiv_id = match.group("identifier")
    authors = [_text(author, f"{ATOM_NAMESPACE}name") for author in entry.findall(f"{ATOM_NAMESPACE}author")]
    pdf_url = next(
        (
            link.attrib["href"]
            for link in entry.findall(f"{ATOM_NAMESPACE}link")
            if link.attrib.get("title") == "pdf" and "href" in link.attrib
        ),
        f"https://arxiv.org/pdf/{arxiv_id}",
    )
    return ArxivCandidate(
        arxiv_id=arxiv_id,
        title=_text(entry, f"{ATOM_NAMESPACE}title"),
        authors=authors,
        abstract=_text(entry, f"{ATOM_NAMESPACE}summary"),
        categories=[category.attrib["term"] for category in entry.findall(f"{ATOM_NAMESPACE}category") if "term" in category.attrib],
        published_at=_parse_datetime(_text(entry, f"{ATOM_NAMESPACE}published")),
        updated_at=_parse_datetime(_text(entry, f"{ATOM_NAMESPACE}updated")),
        abs_url=raw_identifier,
        pdf_url=pdf_url,
    )


def _text(element: element_tree.Element, path: str, *, required: bool = True) -> str:
    child = element.find(path)
    value = " ".join((child.text or "").split()) if child is not None else ""
    if required and not value:
        raise ArxivSearchError(f"arXiv response is missing required field: {path}.")
    return value


def _parse_datetime(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ArxivSearchError("arXiv response contains an invalid timestamp.") from error


def _is_retryable(error: httpx.TransportError | httpx.HTTPStatusError) -> bool:
    if isinstance(error, httpx.TransportError):
        return True
    return error.response.status_code == 429 or error.response.status_code >= 500
