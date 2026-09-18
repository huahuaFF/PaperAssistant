from __future__ import annotations

from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage

from scripts.manual_invoke_workflow import (
    ModelProgressPrinter,
    _model_call_label,
    _structured_output_diagnostic,
)


def test_model_call_label_redacts_prompt_content_and_uses_workflow_tag() -> None:
    label = _model_call_label(
        {"name": "ChatOpenAI", "kwargs": {"model_name": "MiniMax-M2.7-highspeed"}},
        ["agent", "assess_coverage"],
    )

    assert label == "node=assess_coverage model=MiniMax-M2.7-highspeed"


@pytest.mark.asyncio
async def test_model_progress_printer_reports_start_and_error(capsys: pytest.CaptureFixture[str]) -> None:
    printer = ModelProgressPrinter()
    run_id = uuid4()

    await printer.on_chat_model_start({"name": "ChatOpenAI"}, [[]], run_id=run_id)
    await printer.on_llm_error(RuntimeError("provider unavailable"), run_id=run_id)

    output = capsys.readouterr().out
    assert "[llm:start]" in output
    assert "[llm:error]" in output
    assert "provider unavailable" in output


def test_structured_diagnostic_reports_invalid_coverage_fields() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "CoverageAssessment",
                "args": {"sufficient": "yes", "confidence": 2},
                "id": "call-1",
            }
        ],
    )

    diagnostic = _structured_output_diagnostic(
        type("Response", (), {"generations": [[type("Generation", (), {"message": message})()]]})(),
        ["assess_coverage"],
    )

    assert diagnostic is not None
    assert diagnostic.startswith("invalid CoverageAssessment:")
    assert "confidence" in diagnostic
    assert "reason" in diagnostic


def test_structured_diagnostic_shows_a_short_plain_text_excerpt_when_no_tool_is_called() -> None:
    message = AIMessage(content="我无法按工具格式回答。")

    diagnostic = _structured_output_diagnostic(
        type("Response", (), {"generations": [[type("Generation", (), {"message": message})()]]})(),
        ["assess_coverage"],
    )

    assert diagnostic is not None
    assert "expected CoverageAssessment" in diagnostic
    assert "text='我无法按工具格式回答。'" in diagnostic
