from pathlib import Path

from app.services.paper_parsing import (
    CHUNKER_VERSION,
    PaperChunk,
    _is_caption,
    _is_visual_label,
    clean_pdf_text,
    split_page_sections,
    write_parsed_paper,
)


def test_clean_pdf_text_repairs_ligatures_hyphenation_and_whitespace() -> None:
    text = "A \ufb01eld con-\ndition  -  with   spaces\n\n\nNext paragraph"

    assert clean_pdf_text(text) == "A field condition - with spaces\n\nNext paragraph"


def test_section_splitter_inherits_then_updates_sections() -> None:
    text = "Continuation text.\n\n2 Method\n\nMethod text.\n\n3 Experiments\n\nExperiment text."

    assert split_page_sections(text, "Introduction") == [
        ("Introduction", "Continuation text."),
        ("Method", "Method text."),
        ("Experiments", "Experiment text."),
    ]


def test_section_splitter_detects_a_domain_specific_numbered_heading() -> None:
    text = "3\nFlow-Based Recommender - FlowCF\n\nWe describe the model."

    assert split_page_sections(text, "Preliminaries") == [
        ("Flow-Based Recommender - FlowCF", "We describe the model.")
    ]


def test_visual_label_filter_keeps_captions_but_drops_diagram_legends() -> None:
    assert _is_visual_label("Positive Item Negative Item Predicted Score")
    assert not _is_visual_label("Figure 1: Overview of the model.")
    assert not _is_visual_label("2 Method")


def test_caption_detection_does_not_misclassify_prose_cross_references() -> None:
    assert _is_caption("Figure 1: Overview of the model.")
    assert _is_caption("Table 2. Ablation study.")
    assert not _is_caption("Table 1 presents the overall performance comparison.")


def test_chunk_embedding_text_has_context_but_preserves_raw_text() -> None:
    chunk = PaperChunk(
        chunk_id="paper-1:section_token_v2:p0002:body:c0001",
        paper_id="paper-1",
        title="A Paper",
        page_number=2,
        section="Method",
        chunk_index=1,
        source_type="local_pdf",
        text="Original source text.",
    )

    assert chunk.chunk_id.startswith(f"paper-1:{CHUNKER_VERSION}")
    assert chunk.text == "Original source text."
    assert chunk.embedding_text == (
        "Title: A Paper\nSection: Method\nContent type: body\nPage: 2\nOriginal source text."
    )


def test_parsed_artifact_is_written_as_inspectable_json(tmp_path: Path) -> None:
    from app.services.paper_parsing import ParsedPaper

    parsed = ParsedPaper(
        paper_id="paper-1",
        source_path="paper.pdf",
        file_sha256="a" * 64,
        title="A Paper",
        page_count=1,
        chunks=[],
    )

    output = write_parsed_paper(parsed, tmp_path)

    assert output.name == "paper-1-aaaaaaaaaaaa-pymupdf_layout_v2-section_token_v2.json"
    assert '"title": "A Paper"' in output.read_text(encoding="utf-8")
