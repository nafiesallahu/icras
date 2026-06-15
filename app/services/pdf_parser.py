"""Deterministic born-digital PDF parser service for ICRAS Agent B.

This module wraps :mod:`pdfplumber` behind a small, deterministic API that
extracts text, words, and text-lines (each with bounding-box coordinates) from a
born-digital contract PDF. The output is a plain, dependency-free internal model
(:class:`ParsedPdf`) that the clause detector consumes; nothing here performs
clause detection, risk scoring, or any LLM work.

Design rules:

* Page numbers are **1-based** so they match human-readable PDF pages.
* Parsing is deterministic: pdfplumber preserves document order and we never
  sort or reorder words/lines beyond what the library returns.
* Failures are loud: a missing/unreadable PDF, or a PDF whose extracted text is
  empty, raises :class:`PdfParseError`. Agent B converts that into a synthetic
  fallback.

The bounding-box field names match :mod:`pdfplumber` exactly (``x0``, ``top``,
``x1``, ``bottom``) which maps directly onto the
:class:`~app.schemas.extracted_contract.BoundingBox` schema.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)


class PdfParseError(Exception):
    """Raised when a PDF cannot be opened, read, or yields no text.

    Agent B treats this as the trigger for its parser-failure synthetic
    fallback path.
    """


@dataclass(frozen=True)
class ParsedWord:
    """A single word with its pdfplumber bounding-box coordinates."""

    text: str
    x0: float
    top: float
    x1: float
    bottom: float


@dataclass(frozen=True)
class ParsedLine:
    """A single text line with its union bounding box on a page."""

    text: str
    page_number: int
    x0: float
    top: float
    x1: float
    bottom: float


@dataclass(frozen=True)
class ParsedPage:
    """Everything extracted from a single (1-based) PDF page."""

    page_number: int
    page_text: str
    words: list[ParsedWord] = field(default_factory=list)
    lines: list[ParsedLine] = field(default_factory=list)


@dataclass(frozen=True)
class ParsedPdf:
    """The full deterministic parse result for a PDF."""

    source_file: str
    pages: list[ParsedPage] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        """All page text joined by newlines, in page order."""
        return "\n".join(page.page_text for page in self.pages)


class PdfParser:
    """Parses a born-digital PDF into a :class:`ParsedPdf`.

    The parser is stateless, so a single instance can parse many PDFs.
    """

    def parse(self, pdf_path: str | Path) -> ParsedPdf:
        """Parse ``pdf_path`` into a deterministic :class:`ParsedPdf`.

        Args:
            pdf_path: Path to the born-digital contract PDF.

        Returns:
            The :class:`ParsedPdf` containing per-page text, words, and lines.

        Raises:
            PdfParseError: If the path is empty/missing, the file cannot be read
                by pdfplumber, or the extracted text is empty (an empty parse is
                treated as a parsing failure).
        """
        resolved = self._validate_path(pdf_path)

        try:
            pages = self._extract_pages(resolved)
        except PdfParseError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize any reader error
            logger.error("Failed to parse PDF %s: %s", resolved, exc)
            raise PdfParseError(
                f"Failed to parse PDF: {resolved} ({exc})"
            ) from exc

        parsed = ParsedPdf(source_file=resolved.name, pages=pages)

        if not parsed.full_text.strip():
            logger.error("PDF %s produced no extractable text.", resolved)
            raise PdfParseError(
                f"PDF produced no extractable text: {resolved}. "
                "Treating empty extraction as a parsing failure."
            )

        logger.info(
            "Parsed PDF %s (pages=%d, words=%d).",
            resolved,
            parsed.page_count,
            sum(len(page.words) for page in parsed.pages),
        )
        return parsed

    @staticmethod
    def _validate_path(pdf_path: str | Path) -> Path:
        """Validate that ``pdf_path`` points at an existing file."""
        if pdf_path is None or not str(pdf_path).strip():
            raise PdfParseError("pdf_path cannot be empty.")

        resolved = Path(pdf_path)
        if not resolved.exists():
            raise PdfParseError(f"PDF does not exist: {resolved}")
        if not resolved.is_file():
            raise PdfParseError(f"PDF path is not a file: {resolved}")
        return resolved

    @staticmethod
    def _extract_pages(resolved: Path) -> list[ParsedPage]:
        """Extract every page's text, words, and lines using pdfplumber."""
        pages: list[ParsedPage] = []
        with pdfplumber.open(resolved) as document:
            for index, page in enumerate(document.pages, start=1):
                page_text = page.extract_text() or ""

                words = [
                    ParsedWord(
                        text=str(word.get("text", "")),
                        x0=float(word["x0"]),
                        top=float(word["top"]),
                        x1=float(word["x1"]),
                        bottom=float(word["bottom"]),
                    )
                    for word in page.extract_words()
                ]

                lines = [
                    ParsedLine(
                        text=str(line.get("text", "")),
                        page_number=index,
                        x0=float(line["x0"]),
                        top=float(line["top"]),
                        x1=float(line["x1"]),
                        bottom=float(line["bottom"]),
                    )
                    for line in page.extract_text_lines()
                ]

                pages.append(
                    ParsedPage(
                        page_number=index,
                        page_text=page_text,
                        words=words,
                        lines=lines,
                    )
                )
        return pages
