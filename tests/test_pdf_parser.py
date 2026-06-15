"""Tests for the deterministic born-digital PDF parser (Agent B)."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("pdfplumber")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_bundles  # noqa: E402

from app.services.pdf_parser import PdfParseError, PdfParser  # noqa: E402

SCENARIO_03_PDF = (
    REPO_ROOT / "data" / "bundles" / "scenario_03_net_90_payment_terms" / "contract.pdf"
)


def test_parse_uses_1_based_pages_and_text() -> None:
    parsed = PdfParser().parse(SCENARIO_03_PDF)

    assert parsed.page_count >= 1
    assert parsed.pages[0].page_number == 1
    # Pages are numbered 1..N (never 0-based).
    assert [page.page_number for page in parsed.pages] == list(
        range(1, parsed.page_count + 1)
    )
    assert "payment terms" in parsed.full_text.lower()


def test_words_have_bbox_coordinates() -> None:
    parsed = PdfParser().parse(SCENARIO_03_PDF)
    words = parsed.pages[0].words

    assert words, "expected at least one extracted word"
    for word in words:
        assert word.text != ""
        assert word.x1 >= word.x0
        assert word.bottom >= word.top


def test_lines_carry_page_number_and_bbox() -> None:
    parsed = PdfParser().parse(SCENARIO_03_PDF)
    lines = parsed.pages[0].lines

    assert lines
    for line in lines:
        assert line.page_number == 1
        assert line.x1 >= line.x0
        assert line.bottom >= line.top


def test_empty_pdf_is_treated_as_parse_failure(tmp_path: Path) -> None:
    # A real PDF whose only content is blank text must be a parsing failure.
    blank = tmp_path / "blank.pdf"
    blank.write_bytes(build_bundles.build_text_pdf([""]))

    with pytest.raises(PdfParseError):
        PdfParser().parse(blank)


def test_missing_pdf_raises(tmp_path: Path) -> None:
    with pytest.raises(PdfParseError):
        PdfParser().parse(tmp_path / "does_not_exist.pdf")


def test_unreadable_pdf_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.pdf"
    bogus.write_bytes(b"this is not a pdf")

    with pytest.raises(PdfParseError):
        PdfParser().parse(bogus)


def test_parsing_is_deterministic() -> None:
    first = PdfParser().parse(SCENARIO_03_PDF)
    second = PdfParser().parse(SCENARIO_03_PDF)

    assert first.full_text == second.full_text
    assert len(first.pages[0].words) == len(second.pages[0].words)
    first_coords = [
        (w.text, w.x0, w.top, w.x1, w.bottom) for w in first.pages[0].words
    ]
    second_coords = [
        (w.text, w.x0, w.top, w.x1, w.bottom) for w in second.pages[0].words
    ]
    assert first_coords == second_coords
