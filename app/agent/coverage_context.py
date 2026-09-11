"""Bounded evidence context for the coverage-assessment Agent."""

from __future__ import annotations

import json

from pydantic import BaseModel

from app.graph.state import ResearchState
from app.models.schemas import EvidenceItem, QueryIntent

MAX_COVERAGE_EVIDENCE = 8
MAX_COVERAGE_EXCERPT_CHARS = 1_200


class CoverageContext(BaseModel):
    query: str
    intent_json: str
    evidence_json: str

    def prompt_values(self) -> dict[str, str]:
        return self.model_dump()


class CoverageContextBuilder:
    """Validate and bound retrieval evidence before it enters the assessment prompt."""

    def build(self, state: ResearchState) -> CoverageContext:
        intent_data = state.get("intent")
        if not isinstance(intent_data, dict):
            raise TypeError("assess_coverage requires intent from classify_query.")
        raw_evidence = state.get("local_evidence", [])
        if not isinstance(raw_evidence, list):
            raise TypeError("local_evidence must be a list.")
        intent = QueryIntent.model_validate(intent_data)
        evidence = [EvidenceItem.model_validate(item) for item in raw_evidence]
        prompt_evidence = [
            item.model_copy(update={"excerpt": item.excerpt[:MAX_COVERAGE_EXCERPT_CHARS]}).model_dump()
            for item in evidence[:MAX_COVERAGE_EVIDENCE]
        ]
        return CoverageContext(
            query=state["user_query"][:4_000],
            intent_json=intent.model_dump_json(),
            evidence_json=json.dumps(prompt_evidence, ensure_ascii=False),
        )
