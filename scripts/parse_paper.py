"""Create a reviewable parsed-paper artifact without embedding or indexing it."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.config import get_settings
from app.services.paper_parsing import PaperParser, PaperSource, write_parsed_paper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse a local research-paper PDF into reviewable chunks.")
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--arxiv-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parsed = PaperParser().parse(
        PaperSource(
            paper_id=args.paper_id,
            local_path=args.pdf_path,
            arxiv_id=args.arxiv_id,
        )
    )
    output_path = write_parsed_paper(parsed, get_settings().parsed_directory)
    print(f"title={parsed.title}")
    print(f"pages={parsed.page_count}")
    print(f"chunks={len(parsed.chunks)}")
    print(f"artifact={output_path}")


if __name__ == "__main__":
    main()
