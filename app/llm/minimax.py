"""MiniMax OpenAI-compatible chat-model factory."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import Settings


class MissingModelConfigurationError(RuntimeError):
    """Raised only when a model-backed graph node is invoked without credentials."""


def create_minimax_chat_model(settings: Settings) -> ChatOpenAI:
    """Create the single MiniMax model used by v0.1 Agent nodes."""
    if settings.minimax_api_key is None:
        raise MissingModelConfigurationError("MINIMAX_API_KEY is required for Agent nodes.")

    return ChatOpenAI(
        model=settings.minimax_model,
        api_key=settings.minimax_api_key,
        base_url=settings.minimax_base_url,
        temperature=0,
        timeout=settings.minimax_timeout_seconds,
        max_retries=settings.minimax_max_retries,
    )
