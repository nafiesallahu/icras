"""Tests for the born-digital contract PDFs produced by scripts/build_bundles.py.

These tests prove that every scenario ``contract.pdf`` is a real, text-based PDF
(not an empty placeholder) and that the extractable text contains the
scenario-specific clauses Agent B is expected to parse. The bundles are
regenerated once per test session so the assertions always reflect the current
generator output.
"""

import sys
from pathlib import Path

import pytest

pdfplumber = pytest.importorskip("pdfplumber")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_bundles  # noqa: E402  (path set up above)

BUNDLES_DIR = REPO_ROOT / "data" / "bundles"

# Per-scenario expectations: substrings that must be present, and substrings
# that must be absent (the deliberate omissions that define the scenario).
# Matching is case-insensitive.
SCENARIO_EXPECTATIONS: dict[str, dict[str, list[str]]] = {
    "scenario_01_clean_nda": {
        "must_contain": [
            "MUTUAL NON-DISCLOSURE AGREEMENT",
            "Confidentiality",
            "Germany",
            "Signature",
        ],
        "must_not_contain": ["Net 90", "Net 120"],
    },
    "scenario_02_missing_liability_cap": {
        "must_contain": ["SUPPLY AGREEMENT", "Net 30", "Governing Law"],
        # The defining omission: no limitation-of-liability / cap clause.
        "must_not_contain": ["liability"],
    },
    "scenario_03_net_90_payment_terms": {
        "must_contain": ["ninety (90)", "Net 90", "Payment Terms"],
        "must_not_contain": [],
    },
    "scenario_04_high_risk_jurisdiction": {
        "must_contain": ["Russia", "Governing Law"],
        "must_not_contain": [],
    },
    "scenario_05_auto_renewal_no_opt_out": {
        "must_contain": ["automatically renew", "twelve (12) month"],
        "must_not_contain": ["opt-out"],
    },
    "scenario_06_conflicting_governing_law": {
        "must_contain": ["Germany", "New York", "Governing Law"],
        "must_not_contain": [],
    },
    "scenario_07_low_confidence_signature": {
        "must_contain": ["NON-DISCLOSURE", "illegible"],
        "must_not_contain": [],
    },
    "scenario_08_missing_gdpr_clause": {
        "must_contain": ["personal data", "Net 30"],
        # Personal data is processed, but there is no GDPR / data-protection clause.
        "must_not_contain": ["GDPR", "2016/679", "Data Protection"],
    },
    "scenario_09_clean_services_agreement": {
        "must_contain": [
            "STANDARD SERVICES AGREEMENT",
            "Net 30",
            "Limitation of Liability",
            "GDPR",
        ],
        "must_not_contain": ["Net 90", "Net 120"],
    },
    "scenario_10_multiple_high_risks": {
        "must_contain": [
            "one hundred twenty (120)",
            "Net 120",
            "Iran",
            "personal data",
        ],
        # Multiple deliberate omissions: no liability cap, no GDPR clause.
        "must_not_contain": ["Limitation of Liability", "GDPR", "2016/679"],
    },
}

ALL_BUNDLE_IDS = sorted(build_bundles.SCENARIO_MANIFESTS)


@pytest.fixture(scope="session", autouse=True)
def _generated_bundles() -> None:
    """Regenerate all bundles once so tests run against fresh generator output."""
    build_bundles.main()


def _extract_text(bundle_id: str) -> str:
    """Return the concatenated text extracted from a bundle's contract.pdf."""
    pdf_path = BUNDLES_DIR / bundle_id / "contract.pdf"
    with pdfplumber.open(pdf_path) as document:
        return "\n".join(page.extract_text() or "" for page in document.pages)


def _normalized_text(bundle_id: str) -> str:
    """Lowercased text with all whitespace runs collapsed to single spaces.

    Line wrapping inside the PDF inserts newlines mid-phrase, so phrase matching
    must ignore whitespace boundaries.
    """
    return " ".join(_extract_text(bundle_id).split()).lower()


def test_all_scenarios_have_expectations() -> None:
    # Guard against a scenario being added without a matching expectation entry.
    assert set(SCENARIO_EXPECTATIONS) == set(ALL_BUNDLE_IDS)


@pytest.mark.parametrize("bundle_id", ALL_BUNDLE_IDS)
def test_contract_pdf_is_born_digital_text(bundle_id: str) -> None:
    pdf_path = BUNDLES_DIR / bundle_id / "contract.pdf"
    assert pdf_path.is_file()
    # Born-digital marker: a real PDF header.
    assert pdf_path.read_bytes().startswith(b"%PDF-")

    text = _extract_text(bundle_id)
    # A genuine contract has substantial extractable text, not a blank page.
    assert len(text.strip()) > 600, f"{bundle_id} has too little extractable text"


@pytest.mark.parametrize("bundle_id", ALL_BUNDLE_IDS)
def test_contract_pdf_mentions_parties(bundle_id: str) -> None:
    text = _normalized_text(bundle_id)
    counterparty = build_bundles.SCENARIO_MANIFESTS[bundle_id]["counterparty_name"]
    assert counterparty.lower() in text
    assert build_bundles.CUSTOMER_NAME.lower() in text
    assert "signature" in text


@pytest.mark.parametrize("bundle_id", ALL_BUNDLE_IDS)
def test_contract_pdf_scenario_clauses(bundle_id: str) -> None:
    text = _normalized_text(bundle_id)
    expectations = SCENARIO_EXPECTATIONS[bundle_id]

    for needle in expectations["must_contain"]:
        assert needle.lower() in text, (
            f"{bundle_id}: expected clause text {needle!r} missing from PDF"
        )

    for needle in expectations["must_not_contain"]:
        assert needle.lower() not in text, (
            f"{bundle_id}: text {needle!r} should be absent from PDF"
        )


def test_pdf_writer_roundtrips_text(tmp_path: Path) -> None:
    # Unit test for the dependency-free PDF writer, including escaped parentheses.
    pdf_bytes = build_bundles.build_text_pdf(
        ["Hello World", "", "Payment is due Net 90 (ninety (90) days)."]
    )
    out = tmp_path / "sample.pdf"
    out.write_bytes(pdf_bytes)
    with pdfplumber.open(out) as document:
        text = "\n".join(page.extract_text() or "" for page in document.pages)
    normalized = " ".join(text.split())
    assert "Hello World" in normalized
    assert "Net 90 (ninety (90) days)" in normalized
