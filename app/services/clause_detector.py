"""Deterministic clause detection for ICRAS Agent B.

Given a :class:`~app.services.pdf_parser.ParsedPdf`, this module splits the
contract into logical sections, classifies each section into one of the core
clause types, extracts normalized ``structured_fields``, computes bounding-box
locations per page (supporting multi-page aggregation), and assigns a
deterministic confidence score.

Hard rules (deliberately enforced here):

* No LLM, no network, no risk scoring, no approval routing, no legal judgement.
  Agent B only *extracts and structures*; it flags low-confidence clauses for
  manual review but never decides whether a clause is risky.
* Detection is deterministic: the same PDF always yields the same clauses, in
  the same order, with the same confidence values.

The detector returns a :class:`DetectionResult` carrying ready-to-serialize
:class:`~app.schemas.extracted_contract.ContractClause` objects (with a legacy
``evidence_ref`` placeholder); the extraction agent later attaches the
clause-level ``evidence_ids`` produced while updating ``evidence_index.json``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas.extracted_contract import (
    BoundingBox,
    ClauseLocation,
    ClauseType,
    ContractClause,
)
from app.services.pdf_parser import ParsedLine, ParsedPdf

# Default confidence thresholds used when the playbook does not define them.
DEFAULT_MINIMUM_CLAUSE_CONFIDENCE = 0.70
DEFAULT_MINIMUM_SIGNATURE_CONFIDENCE = 0.75

# Deterministic confidence tiers.
_CONFIDENCE_HIGH = 0.95
_CONFIDENCE_MEDIUM = 0.80
_CONFIDENCE_LOW = 0.55
_CONFIDENCE_SIGNATURE = 0.92
_CONFIDENCE_COUNTERPARTY = 0.90

# Max characters retained for a per-page text excerpt.
_EXCERPT_MAX_CHARS = 240

# A numbered clause heading, e.g. "2. Payment Terms".
_NUMBERED_HEADING = re.compile(r"^\s*(\d+)\.\s+(.+?)\s*$")

# Markers that the signature block begins (case-insensitive).
_WITNESS_MARKER = re.compile(r"in\s+witness\s+whereof", re.IGNORECASE)
_SIGNATURE_LINE = re.compile(r"^\s*signature\s*:", re.IGNORECASE)

# Signature degradation markers that force a low-confidence signature clause.
_DEGRADED_MARKERS = ("illegible", "unclear", "[unclear]", "degraded")

# Known governing-law jurisdictions, in match priority order (most specific
# first so "Russian Federation" wins over "Russia", "New York" over
# "United States", etc.).
_JURISDICTIONS = (
    "Russian Federation",
    "Islamic Republic of Iran",
    "New York",
    "United Kingdom",
    "United States",
    "Germany",
    "Russia",
    "Iran",
)

# Spelled-out number words used in payment clauses, mapped to day counts.
_NUMBER_WORDS = {
    "thirty": 30,
    "sixty": 60,
    "ninety": 90,
    "one hundred twenty": 120,
    "one hundred and twenty": 120,
}


@dataclass
class _Section:
    """A contiguous block of lines forming one logical contract section."""

    kind: str  # "preamble" | "numbered" | "signature"
    heading: str
    lines: list[ParsedLine]


@dataclass(frozen=True)
class DetectionResult:
    """The result of clause detection over a single PDF."""

    clauses: list[ContractClause]
    overall_confidence_score: float


class ClauseDetector:
    """Detects structured clauses from a parsed born-digital PDF."""

    def detect(
        self,
        parsed_pdf: ParsedPdf,
        *,
        minimum_clause_confidence: float = DEFAULT_MINIMUM_CLAUSE_CONFIDENCE,
        minimum_signature_confidence: float = DEFAULT_MINIMUM_SIGNATURE_CONFIDENCE,
    ) -> DetectionResult:
        """Detect clauses in ``parsed_pdf``.

        Args:
            parsed_pdf: The parsed PDF produced by :class:`PdfParser`.
            minimum_clause_confidence: Threshold below which a non-signature
                clause is flagged for manual review.
            minimum_signature_confidence: Threshold below which a signature
                clause is flagged for manual review.

        Returns:
            A :class:`DetectionResult` with the detected clauses and the overall
            confidence score (mean of clause confidences, rounded to 2 dp).
        """
        sections = self._split_sections(parsed_pdf)
        clauses: list[ContractClause] = []
        type_counters: dict[ClauseType, int] = {}

        for section in sections:
            detected = self._classify_section(section)
            if detected is None:
                continue
            clause_type, confidence_signals = detected

            text = self._section_text(section, clause_type)
            if not text:
                continue

            structured_fields = self._structured_fields(clause_type, text, section)
            confidence = self._confidence(clause_type, text, confidence_signals)

            type_counters[clause_type] = type_counters.get(clause_type, 0) + 1
            clause_id = f"{clause_type.value}_{type_counters[clause_type]:03d}"

            locations = self._build_locations(section)
            if not locations:
                # No usable lines means no traceable location; skip the clause.
                type_counters[clause_type] -= 1
                continue

            # Deduplicate identical clauses (same type + text): merge locations
            # into the existing clause instead of emitting a duplicate (e.g. for
            # repeated header text).
            existing = self._find_duplicate(clauses, clause_type, text)
            if existing is not None:
                merged_locations = list(existing.locations) + locations
                clauses[clauses.index(existing)] = existing.model_copy(
                    update={"locations": merged_locations}
                )
                type_counters[clause_type] -= 1
                continue

            threshold = (
                minimum_signature_confidence
                if clause_type == ClauseType.SIGNATURE_BLOCK
                else minimum_clause_confidence
            )
            requires_manual_review = confidence < threshold

            first = locations[0]
            clauses.append(
                ContractClause(
                    clause_id=clause_id,
                    clause_type=clause_type,
                    text=text,
                    page_number=first.page,
                    confidence_score=confidence,
                    evidence_ref=(
                        f"{parsed_pdf.source_file}#page={first.page}"
                        f"&clause={clause_id}"
                    ),
                    structured_fields=structured_fields,
                    bbox=first.bbox,
                    locations=locations,
                    evidence_ids=[],
                    requires_manual_review=requires_manual_review,
                )
            )

        overall = self._overall_confidence(clauses)
        return DetectionResult(clauses=clauses, overall_confidence_score=overall)

    # ------------------------------------------------------------------
    # Section splitting
    # ------------------------------------------------------------------
    @staticmethod
    def _split_sections(parsed_pdf: ParsedPdf) -> list[_Section]:
        """Split all PDF lines into preamble / numbered / signature sections."""
        sections: list[_Section] = []
        current: _Section | None = None

        for page in parsed_pdf.pages:
            for line in page.lines:
                heading_match = _NUMBERED_HEADING.match(line.text)
                in_signature = current is not None and current.kind == "signature"

                if heading_match and not in_signature:
                    current = _Section(
                        kind="numbered",
                        heading=heading_match.group(2),
                        lines=[line],
                    )
                    sections.append(current)
                    continue

                starts_signature = not in_signature and (
                    _WITNESS_MARKER.search(line.text)
                    or _SIGNATURE_LINE.match(line.text)
                )
                if starts_signature:
                    current = _Section(
                        kind="signature", heading="Signature", lines=[line]
                    )
                    sections.append(current)
                    continue

                if current is None:
                    current = _Section(kind="preamble", heading="", lines=[line])
                    sections.append(current)
                else:
                    current.lines.append(line)

        return sections

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    def _classify_section(
        self, section: _Section
    ) -> tuple[ClauseType, dict[str, bool]] | None:
        """Classify a section into a clause type, or ``None`` to skip it.

        Returns a tuple of the clause type and a dict of confidence signals
        (``heading``, ``keyword``) used by :meth:`_confidence`.
        """
        if section.kind == "signature":
            return ClauseType.SIGNATURE_BLOCK, {"heading": True, "keyword": True}

        body = self._joined_text(section.lines).lower()

        if section.kind == "preamble":
            if 'counterparty"' in body or "counterparty)" in body or (
                "counterparty" in body and "between" in body
            ):
                return ClauseType.COUNTERPARTY, {"heading": False, "keyword": True}
            return None

        heading = section.heading.lower()
        heading_type = self._classify_by_heading(heading)
        keyword_type = self._classify_by_keyword(body)

        clause_type = heading_type or keyword_type
        if clause_type is None:
            return None

        # GDPR must only be assigned on a genuine data-protection clause, never
        # on a section that merely mentions personal data (e.g. a scope clause).
        if clause_type == ClauseType.GDPR_DATA_PROCESSING and not self._is_gdpr(body):
            # The heading suggested data protection but the body lacks strong
            # signals; fall back to any keyword type, else skip.
            if keyword_type and keyword_type != ClauseType.GDPR_DATA_PROCESSING:
                clause_type = keyword_type
            else:
                return None

        signals = {
            "heading": heading_type is not None,
            "keyword": keyword_type == clause_type,
        }
        return clause_type, signals

    @staticmethod
    def _classify_by_heading(heading: str) -> ClauseType | None:
        if "payment" in heading:
            return ClauseType.PAYMENT_TERMS
        if "liability" in heading:
            return ClauseType.LIABILITY_CAP
        if "governing law" in heading:
            return ClauseType.GOVERNING_LAW
        if "termination" in heading:
            return ClauseType.TERMINATION
        if "auto-renewal" in heading or "auto renewal" in heading or "renewal" in heading:
            return ClauseType.AUTO_RENEWAL
        if "data protection" in heading or "gdpr" in heading or "data processing" in heading:
            return ClauseType.GDPR_DATA_PROCESSING
        if "confidential" in heading or "non-disclosure" in heading:
            return ClauseType.CONFIDENTIALITY
        if "signature" in heading:
            return ClauseType.SIGNATURE_BLOCK
        return None

    @classmethod
    def _classify_by_keyword(cls, body: str) -> ClauseType | None:
        if (
            "payment terms" in body
            or re.search(r"net\s*\d", body)
            or re.search(r"within\s+\w+\s*\(\d+\)\s*days", body)
            or re.search(r"within\s+\d+\s*days", body)
        ):
            return ClauseType.PAYMENT_TERMS
        if (
            "limitation of liability" in body
            or "liability cap" in body
            or "liability shall be capped" in body
            or "total liability" in body
            or "aggregate liability" in body
        ):
            return ClauseType.LIABILITY_CAP
        if (
            "governing law" in body
            or "governed by" in body
            or "laws of" in body
            or "exclusive jurisdiction" in body
            or "courts of" in body
        ):
            return ClauseType.GOVERNING_LAW
        if (
            "auto-renewal" in body
            or "automatically renew" in body
            or "renew for successive" in body
        ):
            return ClauseType.AUTO_RENEWAL
        if cls._is_gdpr(body):
            return ClauseType.GDPR_DATA_PROCESSING
        if (
            "confidential information" in body
            or "confidentiality" in body
            or "non-disclosure" in body
        ):
            return ClauseType.CONFIDENTIALITY
        if (
            "notice of termination" in body
            or "terminate this agreement" in body
            or "may terminate" in body
        ):
            return ClauseType.TERMINATION
        return None

    @staticmethod
    def _is_gdpr(body: str) -> bool:
        return (
            "gdpr" in body
            or "data protection" in body
            or "2016/679" in body
            or "data processing" in body
        )

    # ------------------------------------------------------------------
    # Structured fields
    # ------------------------------------------------------------------
    def _structured_fields(
        self, clause_type: ClauseType, text: str, section: _Section
    ) -> dict:
        if clause_type == ClauseType.PAYMENT_TERMS:
            days = self._payment_days(text)
            return {"payment_terms_days": days} if days is not None else {}

        if clause_type == ClauseType.GOVERNING_LAW:
            jurisdiction = self._governing_law(text)
            return {"governing_law": jurisdiction} if jurisdiction else {}

        if clause_type == ClauseType.LIABILITY_CAP:
            return {
                "liability_cap_present": True,
                "liability_cap_text": text,
            }

        if clause_type == ClauseType.AUTO_RENEWAL:
            return {
                "auto_renewal_present": True,
                "opt_out_window_days": self._opt_out_days(text),
            }

        if clause_type == ClauseType.GDPR_DATA_PROCESSING:
            return {
                "gdpr_clause_present": True,
                "personal_data_referenced": "personal data" in text.lower(),
            }

        if clause_type == ClauseType.CONFIDENTIALITY:
            return {"confidentiality_present": True}

        if clause_type == ClauseType.SIGNATURE_BLOCK:
            return {"signature_block_present": True}

        if clause_type == ClauseType.COUNTERPARTY:
            name = self._counterparty_name(self._joined_text(section.lines))
            return {"counterparty_name": name} if name else {}

        return {}

    @classmethod
    def _payment_days(cls, text: str) -> int | None:
        low = text.lower()
        match = re.search(r"net\s*(\d{1,3})", low)
        if match:
            return int(match.group(1))
        match = re.search(r"\((\d{1,3})\)\s*days", low)
        if match:
            return int(match.group(1))
        match = re.search(r"within\s+(\d{1,3})\s+days", low)
        if match:
            return int(match.group(1))
        for word, value in cls._NUMBER_WORDS_SORTED():
            if word in low:
                return value
        return None

    @staticmethod
    def _NUMBER_WORDS_SORTED() -> list[tuple[str, int]]:
        # Longest phrases first so "one hundred twenty" wins over "twenty".
        return sorted(_NUMBER_WORDS.items(), key=lambda kv: -len(kv[0]))

    @staticmethod
    def _governing_law(text: str) -> str | None:
        for jurisdiction in _JURISDICTIONS:
            if jurisdiction.lower() in text.lower():
                return jurisdiction
        return None

    @staticmethod
    def _opt_out_days(text: str) -> int | None:
        low = text.lower()
        if "opt-out" not in low and "opt out" not in low:
            return None
        match = re.search(r"(\d{1,3})\s*days", low)
        return int(match.group(1)) if match else None

    @staticmethod
    def _counterparty_name(preamble: str) -> str | None:
        match = re.search(
            r"and\s+([A-Za-z0-9 .,&'\-]+?)\s*\(\s*[\"\u201c]?Counterparty",
            preamble,
        )
        if match:
            return match.group(1).strip(" .,")
        return None

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------
    def _confidence(
        self, clause_type: ClauseType, text: str, signals: dict[str, bool]
    ) -> float:
        if clause_type == ClauseType.SIGNATURE_BLOCK:
            low = text.lower()
            if any(marker in low for marker in _DEGRADED_MARKERS):
                return _CONFIDENCE_LOW
            return _CONFIDENCE_SIGNATURE

        if clause_type == ClauseType.COUNTERPARTY:
            return _CONFIDENCE_COUNTERPARTY

        if signals.get("heading") and signals.get("keyword"):
            return _CONFIDENCE_HIGH
        if signals.get("heading") or signals.get("keyword"):
            return _CONFIDENCE_MEDIUM
        return _CONFIDENCE_MEDIUM

    @staticmethod
    def _overall_confidence(clauses: list[ContractClause]) -> float:
        if not clauses:
            return 0.0
        total = sum(clause.confidence_score for clause in clauses)
        return round(total / len(clauses), 2)

    # ------------------------------------------------------------------
    # Locations & text
    # ------------------------------------------------------------------
    @classmethod
    def _build_locations(cls, section: _Section) -> list[ClauseLocation]:
        """Build one location per page the section spans (multi-page aware)."""
        by_page: dict[int, list[ParsedLine]] = {}
        for line in section.lines:
            by_page.setdefault(line.page_number, []).append(line)

        locations: list[ClauseLocation] = []
        for page in sorted(by_page):
            page_lines = by_page[page]
            excerpt = cls._joined_text(page_lines)[:_EXCERPT_MAX_CHARS].strip()
            if not excerpt:
                continue
            bbox = BoundingBox(
                x0=min(line.x0 for line in page_lines),
                top=min(line.top for line in page_lines),
                x1=max(line.x1 for line in page_lines),
                bottom=max(line.bottom for line in page_lines),
            )
            locations.append(
                ClauseLocation(page=page, bbox=bbox, text_excerpt=excerpt)
            )
        return locations

    @classmethod
    def _section_text(cls, section: _Section, clause_type: ClauseType) -> str:
        """Aggregate the section body text (excluding the bare heading line)."""
        if section.kind == "numbered":
            body_lines = [
                line
                for line in section.lines
                if not _NUMBERED_HEADING.match(line.text)
            ]
            text = cls._joined_text(body_lines) or section.heading
        else:
            text = cls._joined_text(section.lines)
        return text.strip()

    @staticmethod
    def _joined_text(lines: list[ParsedLine]) -> str:
        return " ".join(line.text.strip() for line in lines if line.text.strip())

    @staticmethod
    def _find_duplicate(
        clauses: list[ContractClause], clause_type: ClauseType, text: str
    ) -> ContractClause | None:
        for clause in clauses:
            if clause.clause_type == clause_type and clause.text == text:
                return clause
        return None
