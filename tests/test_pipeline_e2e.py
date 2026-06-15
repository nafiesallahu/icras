"""ICRAS end-to-end pipeline integration test (Agent A -> Agent B).

This deterministic integration test runs the real intake agent (Agent A) on a
born-digital contract bundle and then the extraction agent (Agent B) in live
mode, asserting that the two agents integrate: Agent A produces an isolated run
directory with a context packet, and Agent B parses the snapshot PDF into a
validated ``extracted_contract.json`` plus updated evidence index, audit log,
and metrics.

The suite is hermetic: Agent A writes into a temporary runs root and the test
controls environment routing via monkeypatch, so it never touches the
repository ``runs/`` directory.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("pdfplumber")

from app.agents.extraction_agent import ExtractionAgent
from app.agents.intake_agent import IntakeAgent
from app.schemas.context_packet import ContextPacket
from app.schemas.extracted_contract import ExtractedContract

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_03 = (
    REPO_ROOT / "data" / "bundles" / "scenario_03_net_90_payment_terms"
)


def test_intake_then_live_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Live mode: ensure synthetic routing is disabled.
    monkeypatch.delenv("ENV_MODE", raising=False)
    monkeypatch.delenv("MOCK_PIPELINE", raising=False)

    # PHASE 1: Agent A intake.
    intake_agent = IntakeAgent(runs_root=tmp_path / "runs")
    context_packet: ContextPacket = intake_agent.run(str(SCENARIO_03))

    run_dir = Path(context_packet.run_directory)
    assert run_dir.is_dir()
    assert (run_dir / "context_packet.json").is_file()
    assert (run_dir / "evidence_index.json").is_file()

    # PHASE 2: Agent B live extraction.
    extracted: ExtractedContract = extraction_agent_run(run_dir)

    # PHASE 3: Integrity checks across the shared run directory.
    assert (run_dir / "extracted_contract.json").is_file()
    assert isinstance(extracted, ExtractedContract)
    assert extracted.extraction_mode.value == "pdf_parser"
    assert extracted.contract_id == context_packet.contract_id

    payment = [
        c for c in extracted.clauses if c.clause_type.value == "payment_terms"
    ]
    assert payment
    assert payment[0].structured_fields["payment_terms_days"] == 90

    # Evidence, audit, and metrics were all updated by Agent B.
    index = json.loads((run_dir / "evidence_index.json").read_text())
    assert index["evidence_items"]

    audit = (run_dir / "audit_log.md").read_text()
    assert "## Step 1: Intake" in audit
    assert "## Step 2: Clause Extraction" in audit

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["agents_completed"] == ["intake", "extraction"]


def test_intake_then_env_forced_synthetic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intake_agent = IntakeAgent(runs_root=tmp_path / "runs")
    context_packet = intake_agent.run(str(SCENARIO_03))

    monkeypatch.setenv("ENV_MODE", "test")
    extracted = ExtractionAgent().run(
        context_packet.run_directory, scenario_id="3"
    )

    assert extracted.extraction_mode.value == "synthetic_fallback"
    assert (
        Path(context_packet.run_directory) / "extracted_contract.json"
    ).is_file()


def extraction_agent_run(run_dir: Path) -> ExtractedContract:
    """Run Agent B in live mode against an existing run directory."""
    return ExtractionAgent().run(str(run_dir))
