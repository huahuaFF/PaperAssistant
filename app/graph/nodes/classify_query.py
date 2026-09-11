"""LangGraph node for controlled query classification."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from langchain_core.runnables import Runnable

from app.agent.context import ClassificationContextBuilder
from app.graph.nodes.contracts import NodeUpdate
from app.graph.state import ResearchState
from app.models.schemas import QueryIntent
from app.prompts.classify_query import CLASSIFY_QUERY_PROMPT, CLASSIFY_QUERY_PROMPT_VERSION


def create_classify_query_node(
    agent: Runnable[dict[str, object], dict[str, object]],
    context_builder: ClassificationContextBuilder | None = None,
) -> Callable[[ResearchState], Awaitable[NodeUpdate]]:
    """Create the first complete Agent node with injectable LangChain dependencies."""
    builder = context_builder or ClassificationContextBuilder()

    async def classify_query(state: ResearchState) -> NodeUpdate:
        context = builder.build(state)
        prompt_value = CLASSIFY_QUERY_PROMPT.invoke(context.prompt_values())
        agent_result = await agent.ainvoke({"messages": prompt_value.to_messages()})
        intent = agent_result.get("structured_response")
        if not isinstance(intent, QueryIntent):
            raise TypeError("query_classifier did not return a QueryIntent structured response.")
        return {
            "intent": intent.model_dump(),
            "status": "intent_classified",
            "prompt_version": CLASSIFY_QUERY_PROMPT_VERSION,
        }

    return classify_query
