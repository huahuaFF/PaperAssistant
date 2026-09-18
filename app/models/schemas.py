"""Pydantic schemas shared by API endpoints and workflow nodes."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    app_name: str
    environment: str


class CreateConversationRequest(BaseModel):
    title: str = Field(default="新对话", min_length=1, max_length=255)


class CreateRunRequest(BaseModel):
    conversation_id: str
    query: str = Field(min_length=1, max_length=20_000)


class ResumeRunRequest(BaseModel):
    action: Literal["approve_search", "reject_search", "select_papers", "skip_import"]
    selected_paper_ids: list[str] = Field(default_factory=list)


class RunResponse(BaseModel):
    id: str
    conversation_id: str
    status: str
    created_at: datetime | None = None


class EvidenceItem(BaseModel):
    """A retrievable excerpt with enough provenance to support a claim."""

    chunk_id: str
    paper_id: str
    title: str
    excerpt: str
    page_number: int | None = None
    section: str | None = None
    score: float | None = None


class QueryIntent(BaseModel):
    """Validated routing decision returned by the classify-query Agent node."""

    schema_version: Literal["query_intent_v1"] = "query_intent_v1"
    route: Literal["research", "direct_import", "out_of_scope", "needs_clarification"]
    task_type: Literal[
        "literature_question",
        "literature_discovery",
        "paper_analysis",
        "paper_import",
        "paper_comparison",
        "follow_up",
        "other",
    ]
    topic: str = Field(max_length=500)
    key_concepts: list[str] = Field(default_factory=list, max_length=8)
    requires_literature_evidence: bool
    time_scope: str | None = None
    target_arxiv_ids: list[str] = Field(default_factory=list)
    target_dois: list[str] = Field(default_factory=list)
    target_urls: list[str] = Field(default_factory=list)
    referenced_paper_ids: list[str] = Field(default_factory=list)
    output_language: Literal["zh", "en"] = "zh"
    clarification_question: str | None = None

    @model_validator(mode="after")
    def validate_route_contract(self) -> QueryIntent:
        has_import_target = bool(self.target_arxiv_ids or self.target_dois or self.target_urls)

        if self.route == "needs_clarification" and not self.clarification_question:
            raise ValueError("needs_clarification requires clarification_question.")
        if self.route != "needs_clarification" and self.clarification_question is not None:
            raise ValueError("clarification_question is only allowed for needs_clarification.")
        if self.task_type == "paper_import" and self.route != "direct_import":
            raise ValueError("paper_import must use the direct_import route.")
        if self.route == "direct_import" and not has_import_target:
            raise ValueError("direct_import requires an arXiv ID, DOI, or URL.")
        if self.route == "out_of_scope" and self.requires_literature_evidence:
            raise ValueError("out_of_scope requests cannot require literature evidence.")
        return self


class CoverageAssessment(BaseModel):
    sufficient: bool
    confidence: float = Field(ge=0, le=1)
    reason: str
    missing_aspects: list[str] = Field(default_factory=list)
    recommended_action: Literal["answer", "ask_arxiv_permission"]

    @model_validator(mode="after")
    def validate_action_matches_coverage(self) -> CoverageAssessment:
        if self.sufficient and self.recommended_action != "answer":
            raise ValueError("Sufficient coverage must recommend answer.")
        if not self.sufficient and self.recommended_action != "ask_arxiv_permission":
            raise ValueError("Insufficient coverage must request arXiv permission.")
        return self


class SearchApprovalRequest(BaseModel):
    """JSON-serializable payload exposed when the graph pauses before arXiv search."""

    approval_type: Literal["search_arxiv"] = "search_arxiv"
    query: str
    topic: str
    coverage_reason: str
    missing_aspects: list[str]
    local_evidence_count: int = Field(ge=0)


class SearchApprovalResume(BaseModel):
    """Validated human response supplied through ``Command(resume=...)``."""

    decision: Literal["approve", "reject"]


class ArxivSearchPlan(BaseModel):
    keywords: list[str] = Field(min_length=1, max_length=6)
    rationale: str = Field(min_length=1, max_length=2_000)
    categories: list[str] = Field(default_factory=list, max_length=5)
    max_results: int = Field(default=10, ge=1, le=30)


class ArxivCandidate(BaseModel):
    """Normalized paper metadata returned by the arXiv search tool."""

    arxiv_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=2_000)
    authors: list[str] = Field(min_length=1, max_length=100)
    abstract: str = Field(min_length=1, max_length=20_000)
    categories: list[str] = Field(default_factory=list, max_length=30)
    published_at: datetime
    updated_at: datetime
    abs_url: str = Field(min_length=1, max_length=2_000)
    pdf_url: str = Field(min_length=1, max_length=2_000)


class ImportApprovalCandidate(BaseModel):
    """A user-selectable import target with a stable, workflow-local identifier."""

    paper_id: str = Field(min_length=1, max_length=2_500)
    source_type: Literal["arxiv", "doi", "url"]
    source_url: str = Field(min_length=1, max_length=2_000)
    title: str | None = Field(default=None, max_length=2_000)
    authors: list[str] = Field(default_factory=list, max_length=100)
    abstract: str | None = Field(default=None, max_length=20_000)
    categories: list[str] = Field(default_factory=list, max_length=30)


class ImportApprovalRequest(BaseModel):
    """JSON-serializable payload shown before any paper download or ingestion."""

    approval_type: Literal["import_papers"] = "import_papers"
    source: Literal["arxiv_search", "direct_import"]
    query: str
    candidates: list[ImportApprovalCandidate] = Field(max_length=30)


class ImportApprovalResume(BaseModel):
    """Validated human response supplied through ``Command(resume=...)``."""

    decision: Literal["select", "skip"]
    selected_paper_ids: list[str] = Field(default_factory=list, max_length=30)

    @model_validator(mode="after")
    def validate_selected_ids_match_decision(self) -> ImportApprovalResume:
        if self.decision == "select" and not self.selected_paper_ids:
            raise ValueError("Selecting papers requires at least one paper ID.")
        if self.decision == "skip" and self.selected_paper_ids:
            raise ValueError("Skipping import cannot include selected paper IDs.")
        return self


class GroundedClaim(BaseModel):
    """A narrow claim paired with a verbatim quotation from its evidence chunk."""

    statement: str = Field(min_length=1, max_length=2_000)
    evidence_id: str = Field(min_length=1)
    supporting_quote: str = Field(min_length=8, max_length=600)


class ReportSection(BaseModel):
    """A thematic group of individually verifiable report claims."""

    heading: str = Field(min_length=1, max_length=300)
    claims: list[GroundedClaim] = Field(min_length=1, max_length=4)


class GroundedReportDraft(BaseModel):
    """Structured report content before the workflow renders citation labels."""

    sections: list[ReportSection] = Field(min_length=1, max_length=6)
    limitations: list[str] = Field(default_factory=list, max_length=6)
