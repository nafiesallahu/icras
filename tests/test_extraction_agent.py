"""Tests for Agent B — the Extraction Agent.

These tests exercise the real born-digital PDF extraction path, the synthetic
fallback paths (environment-forced and parser-failure), and the shared artifact
updates (evidence index, audit log, metrics). Every test runs Agent A first into
a temporary runs root so the suite is hermetic and never touches the repository
``runs/`` directory. Environment routing is controlled per-test via monkeypatch.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("pdfplumber")

from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_bundles  # noqa: E402

from app.agents.extraction_agent import (  # noqa: E402
    PARSER_FAILURE_FALLBACK_CONFIDENCE,
    ExtractionAgent,
)
from app.agents.intake_agent import IntakeAgent  # noqa: E402
from app.schemas.extracted_contract import ExtractedContract  # noqa: E402

BUNDLES_DIR = REPO_ROOT / "data" / "bundles"


@pytest.fixture(autouse=True)
def _clear_synthetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default to live mode; individual tests opt into synthetic mode."""
    monkeypatch.delenv("ENV_MODE", raising=False)
    monkeypatch.delenv("MOCK_PIPELINE", raising=False)


def _make_run(tmp_path: Path, bundle_id: str) -> str:
    """Run Agent A on a real bundle into a temp runs root; return run dir."""
    intake = IntakeAgent(runs_root=tmp_path / "runs")
    packet = intake.run(str(BUNDLES_DIR / bundle_id))
    return packet.run_directory


def _clause_types(contract: ExtractedContract) -> set[str]:
    return {clause.clause_type.value for clause in contract.clauses}


# ---------------------------------------------------------------------------
# 1. Live PDF extraction success
# ---------------------------------------------------------------------------
def test_live_extraction_success(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, "scenario_03_net_90_payment_terms")
    contract = ExtractionAgent().run(run_dir)

    assert contract.extraction_mode.value == "pdf_parser"
    assert (Path(run_dir) / "extracted_contract.json").is_file()

    payment = [
        c for c in contract.clauses if c.clause_type.value == "payment_terms"
    ]
    assert payment, "expected a payment_terms clause"
    assert payment[0].structured_fields["payment_terms_days"] == 90


# ---------------------------------------------------------------------------
# 2. Required core clause extraction (per scenario)
# ---------------------------------------------------------------------------
def test_scenario_01_core_clauses(tmp_path: Path) -> None:
    contract = ExtractionAgent().run(_make_run(tmp_path, "scenario_01_clean_nda"))
    types = _clause_types(contract)
    assert {"confidentiality", "governing_law", "signature_block"} <= types


def test_scenario_05_auto_renewal(tmp_path: Path) -> None:
    contract = ExtractionAgent().run(
        _make_run(tmp_path, "scenario_05_auto_renewal_no_opt_out")
    )
    auto = [c for c in contract.clauses if c.clause_type.value == "auto_renewal"]
    assert auto
    assert auto[0].structured_fields["auto_renewal_present"] is True
    assert auto[0].structured_fields["opt_out_window_days"] is None


def test_scenario_06_conflicting_governing_law(tmp_path: Path) -> None:
    contract = ExtractionAgent().run(
        _make_run(tmp_path, "scenario_06_conflicting_governing_law")
    )
    governing = [
        c for c in contract.clauses if c.clause_type.value == "governing_law"
    ]
    assert len(governing) >= 2
    jurisdictions = {
        c.structured_fields.get("governing_law") for c in governing
    }
    assert "Germany" in jurisdictions
    assert "New York" in jurisdictions


def test_scenario_08_has_no_gdpr_clause(tmp_path: Path) -> None:
    contract = ExtractionAgent().run(
        _make_run(tmp_path, "scenario_08_missing_gdpr_clause")
    )
    assert "gdpr_data_processing" not in _clause_types(contract)


def test_scenario_10_payment_and_high_risk_jurisdiction(tmp_path: Path) -> None:
    contract = ExtractionAgent().run(
        _make_run(tmp_path, "scenario_10_multiple_high_risks")
    )
    payment = [
        c for c in contract.clauses if c.clause_type.value == "payment_terms"
    ]
    assert payment
    assert payment[0].structured_fields["payment_terms_days"] == 120

    governing = [
        c for c in contract.clauses if c.clause_type.value == "governing_law"
    ]
    assert governing
    assert "Iran" in (governing[0].structured_fields.get("governing_law") or "")


# ---------------------------------------------------------------------------
# 3. Confidence scoring & validation
# ---------------------------------------------------------------------------
def test_every_clause_has_confidence_and_overall(tmp_path: Path) -> None:
    contract = ExtractionAgent().run(
        _make_run(tmp_path, "scenario_03_net_90_payment_terms")
    )
    assert 0.0 < contract.overall_confidence_score <= 1.0
    for clause in contract.clauses:
        assert 0.0 < clause.confidence_score <= 1.0


def test_low_confidence_signature_flagged(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, "scenario_07_low_confidence_signature")
    contract = ExtractionAgent().run(run_dir)

    signature = [
        c for c in contract.clauses if c.clause_type.value == "signature_block"
    ]
    assert signature
    # Below the default minimum signature confidence (0.75) and flagged.
    assert signature[0].confidence_score < 0.75
    assert signature[0].requires_manual_review is True

    metrics = json.loads((Path(run_dir) / "metrics.json").read_text())
    assert metrics["low_confidence_clauses_count"] >= 1


# ---------------------------------------------------------------------------
# 4. Bounding boxes
# ---------------------------------------------------------------------------
def test_clauses_have_locations_and_bbox(tmp_path: Path) -> None:
    contract = ExtractionAgent().run(
        _make_run(tmp_path, "scenario_03_net_90_payment_terms")
    )
    for clause in contract.clauses:
        assert clause.locations, f"{clause.clause_id} has no locations"
        for location in clause.locations:
            assert location.page >= 1
            assert location.text_excerpt.strip()
        # pdf_parser clauses must carry a bbox with all four coordinates.
        bbox = clause.locations[0].bbox
        assert bbox is not None
        assert bbox.x1 >= bbox.x0
        assert bbox.bottom >= bbox.top


# ---------------------------------------------------------------------------
# 5. Evidence index update
# ---------------------------------------------------------------------------
def test_evidence_index_updated(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, "scenario_03_net_90_payment_terms")
    contract = ExtractionAgent().run(run_dir)

    index = json.loads((Path(run_dir) / "evidence_index.json").read_text())
    items = index["evidence_items"]
    assert items

    item_ids = {item["evidence_id"] for item in items}
    all_clause_ids = {
        evidence_id
        for clause in contract.clauses
        for evidence_id in clause.evidence_ids
    }
    assert all_clause_ids, "clauses should reference evidence ids"
    assert all_clause_ids <= item_ids

    first = items[0]
    assert {"evidence_id", "document_id", "filename", "page", "clause_id", "text_excerpt"} <= set(
        first
    )
    assert first["bbox"] is not None


# ---------------------------------------------------------------------------
# 6. Audit log update
# ---------------------------------------------------------------------------
def test_audit_log_has_step_2(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, "scenario_03_net_90_payment_terms")
    ExtractionAgent().run(run_dir)

    audit = (Path(run_dir) / "audit_log.md").read_text()
    assert "## Step 2: Clause Extraction" in audit
    assert "Extraction mode: pdf_parser" in audit
    assert "Clauses extracted:" in audit
    assert "Evidence items created:" in audit
    assert "Overall confidence:" in audit
    assert "Status: extraction_completed" in audit


# ---------------------------------------------------------------------------
# 7. Metrics update
# ---------------------------------------------------------------------------
def test_metrics_updated(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, "scenario_03_net_90_payment_terms")
    ExtractionAgent().run(run_dir)

    metrics = json.loads((Path(run_dir) / "metrics.json").read_text())
    assert "extraction" in metrics["agents_completed"]
    assert "intake" in metrics["agents_completed"]  # Agent A preserved
    assert metrics["extraction_mode"] == "pdf_parser"
    assert metrics["clauses_extracted"] >= 1
    assert metrics["evidence_items_created"] >= 1
    assert metrics["pages_parsed"] >= 1
    assert "overall_extraction_confidence" in metrics
    assert "low_confidence_clauses_count" in metrics
    assert "extraction_completed_at" in metrics


# ---------------------------------------------------------------------------
# 8. Synthetic fallback behavior
# ---------------------------------------------------------------------------
def test_env_mode_forces_synthetic_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _make_run(tmp_path, "scenario_03_net_90_payment_terms")
    monkeypatch.setenv("ENV_MODE", "test")

    contract = ExtractionAgent().run(run_dir, scenario_id="3")
    assert contract.extraction_mode.value == "synthetic_fallback"
    assert (Path(run_dir) / "extracted_contract.json").is_file()

    metrics = json.loads((Path(run_dir) / "metrics.json").read_text())
    assert "extraction" in metrics["agents_completed"]
    assert "## Step 2: Clause Extraction" in (
        Path(run_dir) / "audit_log.md"
    ).read_text()


def test_mock_pipeline_forces_synthetic_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _make_run(tmp_path, "scenario_03_net_90_payment_terms")
    monkeypatch.setenv("MOCK_PIPELINE", "True")

    contract = ExtractionAgent().run(run_dir, scenario_id="3")
    assert contract.extraction_mode.value == "synthetic_fallback"


def test_parser_failure_triggers_synthetic_fallback(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, "scenario_03_net_90_payment_terms")
    # Corrupt the snapshot PDF so live parsing fails in normal mode.
    snapshot_pdf = Path(run_dir) / "input_snapshot" / "contract.pdf"
    snapshot_pdf.write_bytes(b"not a real pdf")

    contract = ExtractionAgent().run(run_dir)
    assert contract.extraction_mode.value == "synthetic_fallback"
    assert contract.overall_confidence_score == PARSER_FAILURE_FALLBACK_CONFIDENCE
    assert (Path(run_dir) / "extracted_contract.json").is_file()

    audit = (Path(run_dir) / "audit_log.md").read_text()
    assert "Synthetic fallback used: yes" in audit
    assert "PDF parsing failed" in audit

    metrics = json.loads((Path(run_dir) / "metrics.json").read_text())
    assert metrics["extraction_mode"] == "synthetic_fallback"
    assert "extraction" in metrics["agents_completed"]


# ---------------------------------------------------------------------------
# 9. Multi-page aggregation
# ---------------------------------------------------------------------------
def test_multi_page_clause_aggregation(tmp_path: Path) -> None:
    # Copy a real bundle, then replace its contract.pdf with a multi-page PDF
    # whose payment clause spans the page boundary.
    bundle = tmp_path / "bundle"
    shutil.copytree(
        BUNDLES_DIR / "scenario_03_net_90_payment_terms", bundle
    )
    filler = [f"Filler line number {i} for pagination." for i in range(44)]
    paragraphs = (
        ["MULTI PAGE SERVICES AGREEMENT"]
        + filler
        + [
            "2. Payment Terms",
            "Payment Terms. The Customer shall pay all undisputed invoices "
            "within ninety (90) days of the date of invoice receipt and shall "
            "continue to honor Net 90 obligations across the remainder of the "
            "billing cycle for every invoice issued under this Agreement.",
        ]
    )
    (bundle / "contract.pdf").write_bytes(
        build_bundles.build_text_pdf(paragraphs)
    )

    intake = IntakeAgent(runs_root=tmp_path / "runs")
    packet = intake.run(str(bundle))
    contract = ExtractionAgent().run(packet.run_directory)

    metrics = json.loads(
        (Path(packet.run_directory) / "metrics.json").read_text()
    )
    assert metrics["pages_parsed"] >= 2

    payment = [
        c for c in contract.clauses if c.clause_type.value == "payment_terms"
    ]
    assert payment, "expected a payment_terms clause"
    clause = payment[0]
    pages = {location.page for location in clause.locations}
    assert len(clause.locations) >= 2
    assert len(pages) >= 2
    # Text is aggregated into a single clause.
    assert "ninety (90) days" in clause.text


# ---------------------------------------------------------------------------
# 10. Schema validation
# ---------------------------------------------------------------------------
def test_output_validates_with_schema(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path, "scenario_03_net_90_payment_terms")
    ExtractionAgent().run(run_dir)
    # Re-loading through the strict schema must succeed.
    reloaded = ExtractedContract.from_json_file(
        Path(run_dir) / "extracted_contract.json"
    )
    assert reloaded.clauses


def test_invalid_contract_raises_validation_error() -> None:
    invalid = {
        "schema_version": "1.0",
        "contract_id": "contract_999",
        "source_file": "contract.pdf",
        "extraction_mode": "pdf_parser",
        "overall_confidence_score": 1.5,  # out of [0, 1]
        "clauses": [],  # must have at least one clause
    }
    with pytest.raises(ValidationError):
        ExtractedContract.model_validate(invalid)


# ---------------------------------------------------------------------------
# 11. Determinism
# ---------------------------------------------------------------------------
def test_extraction_is_deterministic(tmp_path: Path) -> None:
    run_dir_a = _make_run(tmp_path / "a", "scenario_03_net_90_payment_terms")
    run_dir_b = _make_run(tmp_path / "b", "scenario_03_net_90_payment_terms")

    contract_a = ExtractionAgent().run(run_dir_a)
    contract_b = ExtractionAgent().run(run_dir_b)

    summary_a = [
        (c.clause_id, c.clause_type.value, c.confidence_score, c.evidence_ids)
        for c in contract_a.clauses
    ]
    summary_b = [
        (c.clause_id, c.clause_type.value, c.confidence_score, c.evidence_ids)
        for c in contract_b.clauses
    ]
    assert summary_a == summary_b
    assert contract_a.overall_confidence_score == contract_b.overall_confidence_score
