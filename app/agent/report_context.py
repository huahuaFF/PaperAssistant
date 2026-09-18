"""Bounded, injection-safe evidence context for the report-generation Agent."""

from __future__ import annotations

import json

from pydantic import BaseModel

from app.graph.state import ResearchState
from app.models.schemas import EvidenceItem, QueryIntent

MAX_REPORT_EVIDENCE = 10
MAX_REPORT_EXCERPT_CHARS = 1_400


class ReportContext(BaseModel):
    query: str
    intent_json: str
    evidence_json: str

    def prompt_values(self) -> dict[str, str]:
        return self.model_dump()


class ReportContextBuilder:
    """Validate report evidence and bound it before it reaches the model."""

    def build(self, state: ResearchState) -> ReportContext:
        intent_data = state.get("intent")
        if not isinstance(intent_data, dict):
            raise TypeError("generate_report requires intent from classify_query.")
        raw_evidence = state.get("final_evidence", state.get("local_evidence", []))
        if not isinstance(raw_evidence, list) or not raw_evidence:
            raise ValueError("generate_report requires at least one final or local evidence item.")
        intent = QueryIntent.model_validate(intent_data)
        evidence = [EvidenceItem.model_validate(item) for item in raw_evidence]
        model_evidence = [
            item.model_copy(update={"excerpt": item.excerpt[:MAX_REPORT_EXCERPT_CHARS]}).model_dump()
            for item in evidence[:MAX_REPORT_EVIDENCE]
        ]
        return ReportContext(
            query=state["user_query"][:4_000],
            intent_json=intent.model_dump_json(),
            evidence_json=json.dumps(model_evidence, ensure_ascii=False),
        )

    def evidence_by_id(self, state: ResearchState) -> dict[str, EvidenceItem]:
        raw_evidence = state.get("final_evidence", state.get("local_evidence", []))
        if not isinstance(raw_evidence, list):
            raise TypeError("Report evidence must be a list.")
        evidence = [EvidenceItem.model_validate(item) for item in raw_evidence]
        return {item.chunk_id: item for item in evidence[:MAX_REPORT_EVIDENCE]}
