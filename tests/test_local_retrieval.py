import pytest
from langchain_core.documents import Document

from app.graph.nodes.retrieve_local import create_retrieve_local_node
from app.models.schemas import QueryIntent
from app.services.local_retrieval import (
    LocalEvidenceRetriever,
    RetrievalDataError,
    build_retrieval_query,
)


class FakeVectorStore:
    def __init__(self, candidates: list[tuple[Document, float]]) -> None:
        self.candidates = candidates
        self.calls: list[tuple[str, int, float]] = []

    def similarity_search_with_relevance_scores(
        self, query: str, *, k: int, score_threshold: float
    ) -> list[tuple[Document, float]]:
        self.calls.append((query, k, score_threshold))
        return self.candidates


def document(*, paper_id: str, chunk_id: str, score_text: str, page_number: int = 1) -> Document:
    return Document(
        page_content=score_text,
        metadata={
            "paper_id": paper_id,
            "chunk_id": chunk_id,
            "title": f"Title for {paper_id}",
            "page_number": page_number,
            "section": "Method",
            "source_text": f"Raw source for {chunk_id}",
        },
    )


def test_retrieval_query_is_deterministic_and_deduplicates_concepts() -> None:
    intent = QueryIntent(
        route="research",
        task_type="literature_discovery",
        topic="Flow Matching",
        key_concepts=["flow matching", "flow matching", "generative modeling"],
        requires_literature_evidence=True,
        referenced_paper_ids=["paper-1"],
    )

    query = build_retrieval_query(user_query="代表性论文有哪些？", intent=intent)

    assert query == (
        "代表性论文有哪些？\n主题: Flow Matching\n"
        "关键词: flow matching, generative modeling\n已引用论文: paper-1"
    )


@pytest.mark.asyncio
async def test_retriever_filters_low_scores_deduplicates_chunks_and_caps_per_paper() -> None:
    store = FakeVectorStore(
        [
            (document(paper_id="paper-a", chunk_id="a-1", score_text="first"), 0.95),
            (document(paper_id="paper-a", chunk_id="a-2", score_text="second"), 0.90),
            (document(paper_id="paper-a", chunk_id="a-3", score_text="third"), 0.85),
            (document(paper_id="paper-b", chunk_id="b-1", score_text="duplicate"), 0.80),
            (document(paper_id="paper-b", chunk_id="b-1", score_text="duplicate"), 0.79),
            (document(paper_id="paper-c", chunk_id="c-1", score_text="too low"), 0.20),
        ]
    )
    result = await LocalEvidenceRetriever(store, score_threshold=0.35).retrieve("flow matching")

    assert [item.chunk_id for item in result.evidence] == ["a-1", "a-2", "b-1"]
    assert result.candidate_count == 6
    assert result.unique_paper_count == 2
    assert store.calls == [("flow matching", 12, 0.35)]
    assert result.evidence[0].section == "Method"
    assert result.evidence[0].excerpt == "Raw source for a-1"


@pytest.mark.asyncio
async def test_retrieve_node_returns_empty_evidence_for_an_empty_local_library() -> None:
    node = create_retrieve_local_node(LocalEvidenceRetriever(FakeVectorStore([])))
    intent = QueryIntent(
        route="research",
        task_type="literature_question",
        topic="Flow Matching",
        requires_literature_evidence=True,
    )

    update = await node(
        {
            "run_id": "run-1",
            "conversation_id": "conversation-1",
            "workspace_id": "default",
            "user_query": "Flow Matching 是什么？",
            "status": "intent_classified",
            "intent": intent.model_dump(),
        }
    )

    assert update["local_evidence"] == []
    assert update["local_evidence_ids"] == []
    assert update["local_retrieval_stats"] == {
        "candidate_count": 0,
        "selected_count": 0,
        "unique_paper_count": 0,
    }
    assert update["status"] == "local_retrieved"


def test_retriever_rejects_chunks_without_citation_provenance() -> None:
    store = FakeVectorStore([(Document(page_content="orphan chunk", metadata={}), 0.95)])

    with pytest.raises(RetrievalDataError, match="paper_id"):
        LocalEvidenceRetriever(store)._select_evidence(store.candidates)
