import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError

from app.agent.context import ClassificationContextBuilder
from app.graph.nodes.classify_query import create_classify_query_node
from app.models.schemas import EvidenceItem, QueryIntent
from app.prompts.classify_query import CLASSIFY_QUERY_PROMPT


@pytest.mark.asyncio
async def test_classify_node_returns_a_typed_agent_decision() -> None:
    expected_intent = QueryIntent(
        route="research",
        task_type="literature_discovery",
        topic="flow matching",
        key_concepts=["flow matching", "generative modeling"],
        requires_literature_evidence=True,
        output_language="zh",
    )
    agent: Runnable[dict[str, object], dict[str, object]] = RunnableLambda(
        lambda _: {"structured_response": expected_intent}
    )
    node = create_classify_query_node(agent)

    update = await node(
        {
            "run_id": "run-1",
            "conversation_id": "conversation-1",
            "workspace_id": "default",
            "user_query": "流匹配有哪些代表性论文？",
            "status": "started",
        }
    )

    assert update["intent"] == expected_intent.model_dump()
    assert update["status"] == "intent_classified"


def test_context_builder_bounds_history_and_renders_active_papers() -> None:
    context = ClassificationContextBuilder().build(
        {
            "run_id": "run-1",
            "conversation_id": "conversation-1",
            "workspace_id": "default",
            "user_query": "比较第二篇论文的方法。",
            "status": "started",
            "conversation_summary": "用户正在研究流匹配。",
            "chat_history": [
                {"role": "user", "content": "问题一"},
                {"role": "assistant", "content": "回答一"},
                {"role": "user", "content": "问题二"},
                {"role": "assistant", "content": "回答二"},
                {"role": "user", "content": "问题三"},
            ],
            "active_papers": [
                {
                    "paper_id": "paper-2",
                    "title": "Flow Matching for Generative Modeling",
                    "arxiv_id": "2210.02747",
                    "abstract": "A paper about flow matching.",
                }
            ],
        }
    )

    assert len(context.chat_history) == 4
    assert isinstance(context.chat_history[0], AIMessage)
    assert isinstance(context.chat_history[-1], HumanMessage)
    assert "paper-2" in context.active_papers
    assert "2210.02747" in context.active_papers


def test_prompt_template_keeps_history_as_a_message_placeholder() -> None:
    prompt_value = CLASSIFY_QUERY_PROMPT.invoke(
        {
            "query": "流匹配有哪些代表性论文？",
            "conversation_summary": "无",
            "chat_history": [HumanMessage(content="上一轮问题")],
            "active_papers": "无",
        }
    )

    messages = prompt_value.to_messages()
    assert isinstance(messages[0], HumanMessage)
    assert "上一轮问题" == messages[0].content


def test_evidence_item_has_required_provenance() -> None:
    evidence = EvidenceItem(
        chunk_id="paper-1:chunk-3",
        paper_id="paper-1",
        title="Flow Matching for Generative Modeling",
        page_number=2,
        excerpt="We introduce flow matching.",
    )

    assert evidence.chunk_id == "paper-1:chunk-3"


def test_direct_import_intent_requires_an_explicit_paper_identifier() -> None:
    intent = QueryIntent(
        route="direct_import",
        task_type="paper_import",
        topic="Flow Matching for Generative Modeling",
        key_concepts=["flow matching"],
        requires_literature_evidence=True,
        target_arxiv_ids=["2210.02747"],
    )

    assert intent.target_arxiv_ids == ["2210.02747"]

    with pytest.raises(ValidationError, match="direct_import requires"):
        QueryIntent(
            route="direct_import",
            task_type="paper_import",
            topic="一篇未指定的论文",
            requires_literature_evidence=True,
        )
