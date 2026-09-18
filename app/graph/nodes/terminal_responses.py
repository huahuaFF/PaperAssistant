"""Deterministic terminal nodes for non-retrieval workflow branches."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from app.graph.nodes.contracts import NodeUpdate
from app.graph.state import ResearchState
from app.models.schemas import ArxivCandidate, CoverageAssessment, ImportApprovalCandidate, QueryIntent


def create_request_clarification_node() -> Callable[[ResearchState], Awaitable[NodeUpdate]]:
    """Return the classifier's bounded clarification question as the terminal response."""

    async def request_clarification(state: ResearchState) -> NodeUpdate:
        intent = _intent(state)
        if intent.route != "needs_clarification" or not intent.clarification_question:
            raise ValueError("request_clarification requires a clarification intent.")
        return {
            "report_content": intent.clarification_question,
            "status": "clarification_requested",
        }

    return request_clarification


def create_explain_out_of_scope_node() -> Callable[[ResearchState], Awaitable[NodeUpdate]]:
    """End unsupported requests without pretending to have researched them."""

    async def explain_out_of_scope(state: ResearchState) -> NodeUpdate:
        intent = _intent(state)
        if intent.route != "out_of_scope":
            raise ValueError("explain_out_of_scope requires an out_of_scope intent.")
        return {
            "report_content": "该请求不属于当前科研文献助手的处理范围。请改为提出论文检索、导入、阅读或基于本地文献的科研问题。",
            "status": "out_of_scope_explained",
        }

    return explain_out_of_scope


def create_explain_knowledge_gap_node() -> Callable[[ResearchState], Awaitable[NodeUpdate]]:
    """Explain why an answer was not generated after search permission is declined."""

    async def explain_knowledge_gap(state: ResearchState) -> NodeUpdate:
        if state.get("search_approval") != "rejected":
            raise PermissionError("explain_knowledge_gap requires rejected arXiv search approval.")
        coverage_data = state.get("coverage")
        if not isinstance(coverage_data, dict):
            raise TypeError("explain_knowledge_gap requires coverage assessment.")
        coverage = CoverageAssessment.model_validate(coverage_data)
        missing = "；".join(coverage.missing_aspects) or "本地可引用证据不足"
        return {
            "report_content": (
                "当前未生成正式回答：本地文献证据不足，且你未同意搜索 arXiv。\n\n"
                f"证据缺口：{missing}\n\n"
                f"判断依据：{coverage.reason}"
            ),
            "status": "knowledge_gap_explained",
        }

    return explain_knowledge_gap


def create_report_candidates_node() -> Callable[[ResearchState], Awaitable[NodeUpdate]]:
    """Return search results when the user elects not to import them."""

    async def report_candidates(state: ResearchState) -> NodeUpdate:
        if state.get("import_approval") != "skipped":
            raise PermissionError("report_candidates requires skipped import approval.")
        raw_offered_candidates = state.get("import_candidates")
        if raw_offered_candidates is not None:
            if not isinstance(raw_offered_candidates, list):
                raise TypeError("import_candidates must be a list.")
            candidates = [
                ImportApprovalCandidate.model_validate(item) for item in raw_offered_candidates
            ]
            entries = [
                f"{index}. {candidate.title or candidate.paper_id}\n"
                f"   标识: {candidate.paper_id}\n   {candidate.source_url}"
                for index, candidate in enumerate(candidates, start=1)
            ]
        else:
            raw_candidates = state.get("arxiv_candidates", [])
            if not isinstance(raw_candidates, list):
                raise TypeError("arxiv_candidates must be a list.")
            arxiv_candidates = [ArxivCandidate.model_validate(item) for item in raw_candidates]
            entries = [
                f"{index}. {candidate.title}\n   arXiv: {candidate.arxiv_id}\n   {candidate.abs_url}"
                for index, candidate in enumerate(arxiv_candidates, start=1)
            ]
        if not entries:
            content = "未找到可供导入的 arXiv 候选论文。"
        else:
            content = "已跳过导入。以下是本次检索到的候选论文：\n\n" + "\n\n".join(entries)
        return {"report_content": content, "status": "candidates_reported"}

    return report_candidates


def _intent(state: ResearchState) -> QueryIntent:
    intent_data = state.get("intent")
    if not isinstance(intent_data, dict):
        raise TypeError("terminal response requires intent from classify_query.")
    return QueryIntent.model_validate(intent_data)
