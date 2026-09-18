"""Factories for deterministic LangGraph contract tests."""

from __future__ import annotations

from collections.abc import Callable

from app.graph.nodes.contracts import NodeUpdate, ResearchNode
from app.graph.state import ResearchState


def workflow_state(**updates: object) -> ResearchState:
    """Return the smallest valid state accepted by the research graph."""
    return {
        "run_id": "test-run",
        "conversation_id": "test-conversation",
        "workspace_id": "test-workspace",
        "user_query": "测试问题",
        "status": "started",
        **updates,
    }


def traced_node(
    name: str, update: NodeUpdate, trace: list[str]
) -> ResearchNode:
    """Create a side-effect-free node that records graph execution order."""

    async def node(_: ResearchState) -> NodeUpdate:
        trace.append(name)
        return update

    return node
