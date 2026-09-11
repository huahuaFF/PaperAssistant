"""The only application boundary allowed to access Chroma directly."""

from __future__ import annotations

import chromadb
from chromadb.api.models.Collection import Collection
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.config import Settings


class ChromaStore:
    """Persist vector data locally while retaining collection metadata."""

    collection_name = "paper_chunks_embedding_v1"

    def __init__(self, settings: Settings) -> None:
        settings.ensure_local_directories()
        self._settings = settings
        self._client = chromadb.PersistentClient(path=str(settings.chroma_directory))

    def get_collection(self) -> Collection:
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "embedding_model": self._settings.dashscope_embedding_model,
                "embedding_dimensions": self._settings.embedding_dimensions,
            },
        )

    def as_langchain_vector_store(self, embeddings: Embeddings) -> Chroma:
        """Expose this collection through LangChain without leaking the Chroma client."""
        return Chroma(
            client=self._client,
            collection_name=self.collection_name,
            embedding_function=embeddings,
            collection_metadata={
                "embedding_model": self._settings.dashscope_embedding_model,
                "embedding_dimensions": self._settings.embedding_dimensions,
            },
        )

    def heartbeat(self) -> int:
        return self._client.heartbeat()
