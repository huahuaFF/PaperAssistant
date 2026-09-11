"""Page-preserving PDF parsing and chunking for research-paper ingestion."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

PARSER_VERSION = "pymupdf_layout_v2"
CHUNKER_VERSION = "section_token_v2"

_KNOWN_HEADING_PATTERN = re.compile(
    r"(?mi)^\s*(?:\d+(?:\.\d+){0,3}\s+)?"
    r"(abstract|introduction|background|related work|preliminaries|method(?:ology)?|"
    r"approach|experiments?|evaluation|results?|discussion|conclusion(?:s)?|"
    r"limitations?|references|appendix)\s*$"
)
_NUMBERED_HEADING_PATTERN = re.compile(
    r"(?m)^\s*(\d+(?:\.\d+){0,3})\s*(?:\n\s*)?"
    r"([A-Z][A-Za-z0-9 ,:;()/'&+\-]{2,100})\s*$"
)
_CAPTION_PATTERN = re.compile(r"(?i)^(?:figure|table)\s+\d+(?:[.:])")
_REFERENCES = {"references"}
_LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"})
_DASHES = str.maketrans({"–": "-", "—": "-", "−": "-"})


class PaperSource(BaseModel):
    """A local source whose identity has already been assigned by the ingestion coordinator."""

    paper_id: str = Field(min_length=1)
    local_path: Path
    source_type: str = "local_pdf"
    arxiv_id: str | None = None
    version_label: str | None = None


@dataclass(frozen=True)
class LayoutBlock:
    """A body-text or figure-caption block recovered from a PDF page."""

    text: str
    content_type: Literal["body", "figure_caption"]


class PaperChunk(BaseModel):
    """A citation-ready chunk before it is embedded into Chroma."""

    chunk_id: str
    paper_id: str
    title: str
    page_number: int
    section: str | None = None
    chunk_index: int
    content_type: Literal["body", "figure_caption"] = "body"
    parser_version: str = PARSER_VERSION
    chunker_version: str = CHUNKER_VERSION
    source_type: str
    arxiv_id: str | None = None
    text: str

    @property
    def embedding_text(self) -> str:
        """Add stable retrieval context without changing the displayed source text."""
        context = [f"Title: {self.title}"]
        if self.section:
            context.append(f"Section: {self.section}")
        context.append(f"Content type: {self.content_type.replace('_', ' ')}")
        context.append(f"Page: {self.page_number}")
        context.append(self.text)
        return "\n".join(context)


class ParsedPaper(BaseModel):
    """Inspectable intermediate artifact emitted before vectorization starts."""

    schema_version: str = "parsed_paper_v1"
    paper_id: str
    source_path: str
    file_sha256: str
    title: str
    page_count: int
    parser_version: str = PARSER_VERSION
    chunker_version: str = CHUNKER_VERSION
    chunks: list[PaperChunk]


class PaperParseError(ValueError):
    """Raised for invalid or textless PDFs that cannot safely enter the vector library."""


class PaperParser:
    """Parse digital PDFs page by page and produce stable, section-aware chunks."""

    def __init__(
        self,
        *,
        chunk_size_tokens: int = 800,
        chunk_overlap_tokens: int = 120,
        min_chunk_tokens: int = 120,
    ) -> None:
        self._min_chunk_tokens = min_chunk_tokens
        self._splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size_tokens,
            chunk_overlap=chunk_overlap_tokens,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def parse(self, source: PaperSource) -> ParsedPaper:
        path = source.local_path.resolve()
        if not path.is_file() or path.suffix.lower() != ".pdf":
            raise PaperParseError(f"Expected an existing PDF file, got: {path}")

        title = self._extract_title_from_pdf(path)
        pages = self._load_pages(path)
        if not pages:
            raise PaperParseError("PDF contains no pages.")

        chunks: list[PaperChunk] = []
        current_section: str | None = None
        global_chunk_index = 0
        for page_index, blocks in enumerate(pages, start=1):
            page_chunk_counts: defaultdict[str, int] = defaultdict(int)
            for block in blocks:
                sectioned_text: list[tuple[str | None, str]]
                if block.content_type == "figure_caption":
                    sectioned_text = [("Figure caption", block.text)]
                    minimum_tokens = 20
                else:
                    sectioned_text = split_page_sections(block.text, current_section)
                    minimum_tokens = self._min_chunk_tokens
                for section, text in sectioned_text:
                    if block.content_type == "body":
                        current_section = section or current_section
                    if section and section.casefold() in _REFERENCES:
                        continue
                    for chunk_text in self._splitter.split_text(text):
                        if token_count(chunk_text) < minimum_tokens:
                            continue
                        global_chunk_index += 1
                        page_chunk_counts[block.content_type] += 1
                        chunks.append(
                            PaperChunk(
                                chunk_id=(
                                    f"{source.paper_id}:{CHUNKER_VERSION}:p{page_index:04d}:"
                                    f"{block.content_type}:c{page_chunk_counts[block.content_type]:04d}"
                                ),
                                paper_id=source.paper_id,
                                title=title,
                                page_number=page_index,
                                section=section or current_section,
                                chunk_index=global_chunk_index,
                                content_type=block.content_type,
                                source_type=source.source_type,
                                arxiv_id=source.arxiv_id,
                                text=chunk_text,
                            )
                        )

        if not chunks:
            raise PaperParseError("No chunks met the minimum text threshold; OCR may be required.")
        return ParsedPaper(
            paper_id=source.paper_id,
            source_path=str(path),
            file_sha256=file_sha256(path),
            title=title,
            page_count=len(pages),
            chunks=chunks,
        )

    @staticmethod
    def _extract_title_from_pdf(source_path: Path) -> str:
        with pymupdf.open(source_path) as pdf:
            blocks = sorted(pdf[0].get_text("blocks"), key=lambda block: block[1])
        for _, y0, _, _, raw_text, *_ in blocks:
            if y0 > 240:
                break
            for line in clean_pdf_text(raw_text).splitlines():
                candidate = line.strip()
                if len(candidate) >= 12 and "@" not in candidate and not candidate.casefold().startswith("arxiv:"):
                    return candidate
        return source_path.stem

    @staticmethod
    def _load_pages(path: Path) -> list[list[LayoutBlock]]:
        """Build layout-aware page blocks before passing text to the LangChain splitter."""
        pages: list[list[LayoutBlock]] = []
        with pymupdf.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                blocks = _ordered_body_blocks(page, is_first_page=page_number == 1)
                pages.append(blocks)
        return pages


def clean_pdf_text(text: str) -> str:
    """Normalize common PDF extraction artifacts while preserving paragraph boundaries."""
    text = (
        text.translate(_LIGATURES)
        .translate(_DASHES)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _ordered_body_blocks(page: pymupdf.Page, *, is_first_page: bool) -> list[LayoutBlock]:
    """Read two-column pages left-to-right by column while dropping margin boilerplate."""
    page_width = page.rect.width
    page_height = page.rect.height
    left_column: list[tuple[float, str, Literal["body", "figure_caption"]]] = []
    right_column: list[tuple[float, str, Literal["body", "figure_caption"]]] = []
    wide_blocks: list[tuple[float, str, Literal["body", "figure_caption"]]] = []
    raw_blocks = page.get_text("blocks")
    first_body_y = _first_page_body_y(raw_blocks) if is_first_page else None
    for x0, y0, x1, y1, raw_text, *_ in raw_blocks:
        text = clean_pdf_text(raw_text)
        if first_body_y is not None and y0 < first_body_y:
            continue
        if _is_visual_label(text):
            continue
        if not text or _is_margin_or_boilerplate(
            text=text,
            x0=x0,
            y0=y0,
            y1=y1,
            page_height=page_height,
        ):
            continue
        content_type: Literal["body", "figure_caption"] = (
            "figure_caption" if _is_caption(text) else "body"
        )
        if x1 - x0 > page_width * 0.7:
            wide_blocks.append((y0, text, content_type))
        elif x0 < page_width / 2:
            left_column.append((y0, text, content_type))
        else:
            right_column.append((y0, text, content_type))

    ordered = sorted(wide_blocks) + sorted(left_column) + sorted(right_column)
    return [LayoutBlock(text=text, content_type=content_type) for _, text, content_type in ordered]


def _first_page_body_y(blocks: list[tuple[object, ...]]) -> float | None:
    """Find the Abstract label and use it to exclude a title/author area on page one."""
    for _, y0, _, _, raw_text, *_ in blocks:
        if clean_pdf_text(str(raw_text)).casefold() == "abstract" and isinstance(y0, int | float):
            return float(y0)
    return None


def _is_visual_label(text: str) -> bool:
    """Drop short diagram labels while retaining section headings and figure captions."""
    normalized = " ".join(text.split())
    if not normalized or _find_section_matches(normalized):
        return False
    if normalized.casefold().startswith(("figure ", "table ")):
        return False
    words = re.findall(r"[A-Za-z]+", normalized)
    return len(words) <= 12 and "." not in normalized and ":" not in normalized


def _is_caption(text: str) -> bool:
    """Distinguish a real figure/table caption from a prose cross-reference."""
    return bool(_CAPTION_PATTERN.match(text.strip()))


def _is_margin_or_boilerplate(
    *, text: str, x0: float, y0: float, y1: float, page_height: float
) -> bool:
    normalized = " ".join(text.casefold().split())
    if x0 < 45 or y0 < 65 or y1 > page_height - 55:
        return True
    if normalized.startswith(("acm reference format:", "ccs concepts", "keywords")):
        return True
    return any(
        marker in normalized
        for marker in (
            "this work is licensed under",
            "copyright held by the owner",
            "acm isbn",
            "kdd '25, august",
            "kdd \u201925, august",
        )
    )


def split_page_sections(text: str, inherited_section: str | None) -> list[tuple[str | None, str]]:
    """Split a page around recognizable headings and inherit its prior section otherwise."""
    matches = _find_section_matches(text)
    if not matches:
        return [(inherited_section, text)]

    sections: list[tuple[str | None, str]] = []
    active_section = inherited_section
    cursor = 0
    for start, end, heading in matches:
        before = text[cursor:start].strip()
        if before:
            sections.append((active_section, before))
        active_section = heading
        cursor = end
    tail = text[cursor:].strip()
    if tail:
        sections.append((active_section, tail))
    return sections


def _find_section_matches(text: str) -> list[tuple[int, int, str]]:
    matches: list[tuple[int, int, str]] = []
    for match in _KNOWN_HEADING_PATTERN.finditer(text):
        matches.append((match.start(), match.end(), match.group(1).strip().title()))
    for match in _NUMBERED_HEADING_PATTERN.finditer(text):
        matches.append((match.start(), match.end(), match.group(2).strip()))
    return sorted(matches, key=lambda item: item[0])


def token_count(text: str) -> int:
    """Use the splitter's tokenizer family for deterministic minimum-length filtering."""
    return len(re.findall(r"\w+|[^\w\s]", text))


def file_sha256(path: Path) -> str:
    """Compute the content identifier used for idempotent import decisions."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_parsed_paper(parsed: ParsedPaper, output_directory: Path) -> Path:
    """Persist an inspectable intermediate artifact outside the vector store."""
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / (
        f"{parsed.paper_id}-{parsed.file_sha256[:12]}-"
        f"{parsed.parser_version}-{parsed.chunker_version}.json"
    )
    output_path.write_text(parsed.model_dump_json(indent=2), encoding="utf-8")
    return output_path
