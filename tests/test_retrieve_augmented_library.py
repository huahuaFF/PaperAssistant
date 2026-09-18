from typing import Any, cast

import pytest
from langchain_core.documents import Document

from app.graph.nodes.retrieve_augmented_library import create_retrieve_augmented_library_node
from app.services.local_retrieval import LocalEvidenceRetriever


def make_document(paper_id: str, chunk_id: str) -> Document:
    return Document(
        page_content=f"Embedding text for {chunk_id}",
        metadata={
            "paper_id": paper_id,
            "chunk_id": chunk_id,
            "title": f"Title for {paper_id}",
            "page_number": 1,
            "section": "Method",
            "source_text": f"Source text for {chunk_id}",
        },
    )


class FilterAwareVectorStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, object] | None] = []

    def similarity_search_with_relevance_scores(
        self,
        _: str,
        *,
        k: int,
        score_threshold: float | None,
        filter: dict[str, object] | None = None,
    ) -> list[tuple[Document, float]]:
        assert k == 12
        assert score_threshold is None
        self.calls.append(filter)
        if filter is not None:
            return [(make_document("arxiv-2210-02747", "new-1"), 0.92)]
        return [
            (make_document("old-paper", "old-1"), 0.96),
            (make_document("arxiv-2210-02747", "new-1"), 0.92),
        ]


def augmented_state() -> dict[str, object]:
    return {
        "run_id": "run-1",
        "conversation_id": "conversation-1",
        "workspace_id": "default",
        "user_query": "流匹配方法如何工作？",
        "status": "papers_ingested",
        "import_approval": "selected",
        "imported_paper_ids": ["arxiv-2210-02747"],
        "intent": {
            "schema_version": "query_intent_v1",
            "route": "research",
            "task_type": "literature_question",
            "topic": "Flow Matching",
            "key_concepts": ["flow matching"],
            "requires_literature_evidence": True,
            "target_arxiv_ids": [],
            "target_dois": [],
            "target_urls": [],
            "referenced_paper_ids": [],
            "output_language": "zh",
            "clarification_question": None,
        },
    }


@pytest.mark.asyncio
async def test_augmented_retrieval_prioritizes_newly_imported_evidence() -> None:
    store = FilterAwareVectorStore()
    node = create_retrieve_augmented_library_node(LocalEvidenceRetriever(store))

    update = await node(cast(Any, augmented_state()))

    assert store.calls == [{"paper_id": {"$in": ["arxiv-2210-02747"]}}, None]
    assert update["final_evidence_ids"] == ["new-1", "old-1"]
    assert update["new_paper_evidence_ids"] == ["new-1"]
    assert update["augmented_retrieval_stats"] == {
        "imported_paper_count": 1,
        "imported_candidate_count": 1,
        "imported_selected_count": 1,
        "library_candidate_count": 2,
        "library_selected_count": 2,
        "final_selected_count": 2,
        "new_paper_evidence_count": 1,
    }
    assert update["status"] == "augmented_library_retrieved"


@pytest.mark.asyncio
async def test_augmented_retrieval_requires_imported_papers() -> None:
    node = create_retrieve_augmented_library_node(LocalEvidenceRetriever(FilterAwareVectorStore()))
    state = augmented_state()
    state["imported_paper_ids"] = []

    with pytest.raises(TypeError, match="imported_paper_ids"):
        await node(cast(Any, state))
