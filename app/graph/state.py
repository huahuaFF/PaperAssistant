"""Serializable state passed between nodes in the research workflow."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class ResearchState(TypedDict):
    run_id: str
    conversation_id: str
    workspace_id: str
    user_query: str
    status: str
    conversation_summary: NotRequired[str]
    chat_history: NotRequired[list[dict[str, str]]]
    active_papers: NotRequired[list[dict[str, object]]]
    local_evidence: NotRequired[list[dict[str, object]]]
    retrieval_query: NotRequired[str]
    local_retrieval_stats: NotRequired[dict[str, object]]
    intent: NotRequired[dict[str, object]]
    local_evidence_ids: NotRequired[list[str]]
    coverage: NotRequired[dict[str, object]]
    search_approval: NotRequired[Literal["approved", "rejected"]]
    arxiv_query: NotRequired[str]
    arxiv_search_plan: NotRequired[dict[str, object]]
    arxiv_candidates: NotRequired[list[dict[str, object]]]
    import_approval: NotRequired[Literal["selected", "skipped"]]
    selected_paper_ids: NotRequired[list[str]]
    imported_paper_ids: NotRequired[list[str]]
    final_evidence_ids: NotRequired[list[str]]
    report_id: NotRequired[str]
    errors: NotRequired[list[str]]
