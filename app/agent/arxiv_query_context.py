"""Validated prompt context for arXiv search planning."""

from __future__ import annotations

from pydantic import BaseModel

from app.graph.state import ResearchState
from app.models.schemas import CoverageAssessment, QueryIntent


class ArxivQueryContext(BaseModel):
    query: str
    intent_json: str
    coverage_json: str

    def prompt_values(self) -> dict[str, str]:
        return self.model_dump()


class ArxivQueryContextBuilder:
    """Validate approved-search context before it is rendered into a prompt."""

    def build(self, state: ResearchState) -> ArxivQueryContext:
        intent_data = state.get("intent")
        coverage_data = state.get("coverage")
        if not isinstance(intent_data, dict):
            raise TypeError("build_arxiv_query requires intent from classify_query.")
        if not isinstance(coverage_data, dict):
            raise TypeError("build_arxiv_query requires coverage from assess_coverage.")
        intent = QueryIntent.model_validate(intent_data)
        coverage = CoverageAssessment.model_validate(coverage_data)
        if coverage.sufficient:
            raise ValueError("An arXiv search plan is only valid when local coverage is insufficient.")
        return ArxivQueryContext(
            query=state["user_query"][:4_000],
            intent_json=intent.model_dump_json(),
            coverage_json=coverage.model_dump_json(),
        )
