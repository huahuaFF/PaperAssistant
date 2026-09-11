"""Index a reviewed ParsedPaper JSON into local Chroma using DashScope embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_settings
from app.llm.embeddings import create_dashscope_embeddings
from app.repositories.chroma_store import ChromaStore
from app.services.paper_indexing import PaperIndexer
from app.services.paper_parsing import ParsedPaper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index a reviewed parsed-paper JSON artifact.")
    parser.add_argument("parsed_json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parsed = ParsedPaper.model_validate_json(args.parsed_json.read_text(encoding="utf-8"))
    settings = get_settings()
    chroma_store = ChromaStore(settings)
    result = PaperIndexer(
        chroma_store.as_langchain_vector_store(create_dashscope_embeddings(settings))
    ).index(parsed)
    print(f"paper_id={result.paper_id}")
    print(f"upserted_chunks={len(result.upserted_chunk_ids)}")
    print(f"collection_count={chroma_store.get_collection().count()}")


if __name__ == "__main__":
    main()
