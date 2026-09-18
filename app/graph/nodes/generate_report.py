"""Generate a citation-validated research report from Chroma evidence."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.runnables import Runnable

from app.agent.report_context import ReportContextBuilder
from app.graph.nodes.contracts import NodeUpdate
from app.graph.state import ResearchState
from app.models.schemas import EvidenceItem, GroundedReportDraft
from app.prompts.generate_report import REPORT_PROMPT, REPORT_PROMPT_VERSION


def create_generate_report_node(
    agent: Runnable[dict[str, object], dict[str, object]],
    context_builder: ReportContextBuilder | None = None,
) -> Callable[[ResearchState], Awaitable[NodeUpdate]]:
    """Create a report Agent node that validates every cited chunk after generation."""
    builder = context_builder or ReportContextBuilder()

    async def generate_report(state: ResearchState) -> NodeUpdate:
        context = builder.build(state)
        evidence_by_id = builder.evidence_by_id(state)
        prompt_value = REPORT_PROMPT.invoke(context.prompt_values())
        agent_result = await agent.ainvoke({"messages": prompt_value.to_messages()})
        draft = agent_result.get("structured_response")
        if not isinstance(draft, GroundedReportDraft):
            raise TypeError("grounded_report_writer did not return a GroundedReportDraft structured response.")
        _validate_draft_grounding(draft, evidence_by_id)
        content, citations = _render_report(draft, evidence_by_id, title=state["user_query"])
        return {
            "report_content": content,
            "report_draft": draft.model_dump(),
            "report_citations": citations,
            "final_evidence_ids": list(evidence_by_id),
            "status": "report_generated",
            "prompt_version": REPORT_PROMPT_VERSION,
        }

    return generate_report


def _validate_draft_grounding(
    draft: GroundedReportDraft, evidence_by_id: dict[str, EvidenceItem]
) -> None:
    claims = [claim for section in draft.sections for claim in section.claims]
    unknown_ids = {claim.evidence_id for claim in claims if claim.evidence_id not in evidence_by_id}
    if unknown_ids:
        display = ", ".join(sorted(unknown_ids))
        raise ValueError(f"Report draft cited evidence IDs not supplied to the Agent: {display}")
    for claim in claims:
        evidence = evidence_by_id[claim.evidence_id]
        if _normalized_text(claim.supporting_quote) not in _normalized_text(evidence.excerpt):
            raise ValueError(
                "Report claim supporting_quote must be copied from its cited evidence: "
                f"{claim.evidence_id}"
            )


def _normalized_text(value: str) -> str:
    return "".join(value.casefold().split())


def _render_report(
    draft: GroundedReportDraft, evidence_by_id: dict[str, EvidenceItem], *, title: str
) -> tuple[str, list[dict[str, object]]]:
    lines = [f"# {title}"]
    citations: list[dict[str, object]] = []
    cited_chunk_ids: set[str] = set()
    for section in draft.sections:
        lines.extend(["", f"## {section.heading}", ""])
        for claim in section.claims:
            evidence = evidence_by_id[claim.evidence_id]
            label = _citation_label(evidence)
            lines.extend([claim.statement, "", f"证据：{label}", ""])
            if claim.evidence_id not in cited_chunk_ids:
                cited_chunk_ids.add(claim.evidence_id)
                citations.append(
                    {
                        "chunk_id": evidence.chunk_id,
                        "citation_label": label,
                        "page_number": evidence.page_number,
                    }
                )
    if draft.limitations:
        lines.extend(["", "## 局限与证据边界", ""])
        lines.extend(f"- {limitation}" for limitation in draft.limitations)
    return "\n".join(lines), citations


def _citation_label(evidence: EvidenceItem) -> str:
    page = f"p. {evidence.page_number}" if evidence.page_number is not None else "页码未知"
    section = f"，{evidence.section}" if evidence.section else ""
    return f"【{evidence.title}{section}，{page}】"
