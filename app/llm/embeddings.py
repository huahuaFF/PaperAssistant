"""DashScope embedding factory used by retrieval and ingestion services."""

from __future__ import annotations

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings

from app.config import Settings


def create_dashscope_embeddings(settings: Settings) -> Embeddings:
    """Create the LangChain adapter for the configured DashScope embedding model."""
    api_key = settings.dashscope_api_key
    if api_key is None:
        raise ValueError("DASHSCOPE_API_KEY must be configured before using local retrieval.")

    return DashScopeEmbeddings(
        model=settings.dashscope_embedding_model,
        dashscope_api_key=api_key.get_secret_value(),
    )
