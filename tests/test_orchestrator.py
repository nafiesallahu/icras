"""Tests for the ICRAS master pipeline orchestrator and mock Agent H boundary."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from app.agents.counterparty_agent import (
    extract_counterparty_name_from_extracted_contract,
    run_counterparty_agent,
)
from app.agents.triage_agent import run_triage_agent
from app.orchestrator import (
    EXPECTED_FINAL_ARTIFACTS,
    PipelineResult,
    run_pipeline,
    run_risk_scoring_agent,
    run_validation_agent,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_03 = REPO_ROOT / "data" / "bundles" / "scenario_03_net_90_payment_terms"
SCENARIO_03_FIXTURE = (
    REPO_ROOT / "data" / "synthetic_fixtures" / "scenario_03_extracted_contract.json"
)

JSON_ARTIFACTS = [name for name in EXPECTED_FINAL_ARTIFACTS if name.endswith(".json")]


@pytest.fixture(autouse=True)
def _force_synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force deterministic synthetic extraction for all orchestrator tests."""
    monkeypatch.setenv("ENV_MODE", "test")
    monkeypatch.delenv("MOCK_PIPELINE", raising=False)


def _run(tmp_path: Path) -> PipelineResult:
    return run_pipeline(str(SCENARIO_03), runs_root=tmp_path / "runs")


# ----------------------------------------------------------------------------
# Orchestrator: run directory + artifacts
# ----------------------------------------------------------------------------
def test_orchestrator_uses_agent_a_run_dir_without_duplicates(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    result = _run(tmp_path)

    assert result.status == "completed"
    run_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    assert Path(result.run_directory) == run_dirs[0]
    assert run_dirs[0].name == result.run_id


def test_orchestrator_produces_expected_artifacts(tmp_path: Path) -> None:
    result = _run(tmp_path)
    run_dir = Path(result.run_directory)

    assert (run_dir / "input_snapshot").is_dir()
    for name in EXPECTED_FINAL_ARTIFACTS:
        assert (run_dir / name).exists(), f"missing artifact: {name}"
        assert (run_dir / name).stat().st_size > 0, f"empty artifact: {name}"


def test_orchestrator_artifacts_are_valid_json(tmp_path: Path) -> None:
    result = _run(tmp_path)
    run_dir = Path(result.run_directory)
    for name in JSON_ARTIFACTS:
        json.loads((run_dir / name).read_text(encoding="utf-8"))


def test_orchestrator_result_reports_decision(tmp_path: Path) -> None:
    result = _run(tmp_path)
    assert result.contract_id == "contract_003"
    assert result.bundle_id == "scenario_03_net_90_payment_terms"
    assert result.final_decision is not None
    assert result.overall_risk is not None
    assert result.agents_completed == [
        "intake",
        "extraction",
        "counterparty",
        "validation",
        "risk_scoring",
        "triage",
    ]


def test_orchestrator_outputs_are_deterministic(tmp_path: Path) -> None:
    first = _run(tmp_path / "a")
    second = _run(tmp_path / "b")

    for name in ("clause_analysis.json", "approval_packet.json", "posting_payload.json"):
        first_data = json.loads((Path(first.run_directory) / name).read_text())
        second_data = json.loads((Path(second.run_directory) / name).read_text())
        assert first_data == second_data

    assert first.final_decision == second.final_decision
    assert first.overall_risk == second.overall_risk


# ----------------------------------------------------------------------------
# Agent C strict-schema counterparty resolution
# ----------------------------------------------------------------------------
def test_extract_counterparty_from_strict_schema_without_parties() -> None:
    extracted = json.loads(SCENARIO_03_FIXTURE.read_text(encoding="utf-8"))
    assert "parties" not in extracted

    name = extract_counterparty_name_from_extracted_contract(extracted)
    assert name == "Acme Services GmbH"


def test_extract_counterparty_prefers_parties_for_backward_compat() -> None:
    extracted = {
        "parties": {"counterparty_name": "Legacy Corp"},
        "clauses": [
            {
                "clause_type": "counterparty",
                "structured_fields": {"counterparty_name": "Strict Corp"},
            }
        ],
    }
    assert extract_counterparty_name_from_extracted_contract(extracted) == "Legacy Corp"


def test_agent_c_reads_counterparty_from_strict_contract(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run_strict"
    (run_dir / "input_snapshot").mkdir(parents=True)

    (run_dir / "context_packet.json").write_text(
        json.dumps({"counterparty_name_from_manifest": "Acme Services GmbH"}),
        encoding="utf-8",
    )
    # Strict ExtractedContract shape (no top-level "parties").
    shutil.copy2(SCENARIO_03_FIXTURE, run_dir / "extracted_contract.json")
    (run_dir / "input_snapshot" / "vendor_master.csv").write_text(
        "vendor_id,vendor_name,country,status,risk_level\n"
        "V003,Acme Services GmbH,Germany,approved,low\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(json.dumps({"agents": {}}), encoding="utf-8")
    (run_dir / "audit_log.md").write_text("# Audit\n", encoding="utf-8")

    result = run_counterparty_agent(run_dir)
    assert result.status == "approved"
    assert result.matched_vendor_name == "Acme Services GmbH"
    assert (run_dir / "normalized_counterparty.json").is_file()


# ----------------------------------------------------------------------------
# Agent D / E wrappers
# ----------------------------------------------------------------------------
def _seed_run_dir_through_extraction(tmp_path: Path) -> Path:
    """Create a run dir populated up to extracted_contract.json (synthetic)."""
    run_dir = tmp_path / "runs" / "run_wrap"
    (run_dir / "input_snapshot").mkdir(parents=True)
    shutil.copy2(SCENARIO_03_FIXTURE, run_dir / "extracted_contract.json")
    shutil.copy2(
        SCENARIO_03 / "playbook.yaml", run_dir / "input_snapshot" / "playbook.yaml"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps({"agents_completed": ["intake", "extraction"]}), encoding="utf-8"
    )
    (run_dir / "audit_log.md").write_text("# Audit\n", encoding="utf-8")
    return run_dir


def test_run_validation_agent_writes_validation_result(tmp_path: Path) -> None:
    run_dir = _seed_run_dir_through_extraction(tmp_path)

    result = run_validation_agent(run_dir)

    out = run_dir / "validation_result.json"
    assert out.is_file()
    assert result.contract_id == "contract_003"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "validation" in metrics["agents_completed"]
    assert "## Step 4: Validation" in (run_dir / "audit_log.md").read_text()


def test_run_risk_scoring_agent_writes_clause_analysis(tmp_path: Path) -> None:
    run_dir = _seed_run_dir_through_extraction(tmp_path)
    run_validation_agent(run_dir)

    clause_analysis = run_risk_scoring_agent(run_dir)

    assert (run_dir / "clause_analysis.json").is_file()
    obligations = run_dir / "obligations.csv"
    assert obligations.is_file()
    lines = obligations.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "obligation_id,contract_id,clause_id,description,owner,due_date"

    assert clause_analysis.contract_id == "contract_003"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "risk_scoring" in metrics["agents_completed"]
    assert metrics["obligations_mode"] == "minimal_orchestrator_fallback"


# ----------------------------------------------------------------------------
# Mock Agent H boundary
# ----------------------------------------------------------------------------
def _seed_run_dir_for_triage(tmp_path: Path, risk_tier: str, category: str) -> Path:
    run_dir = tmp_path / "runs" / "run_triage"
    (run_dir / "input_snapshot").mkdir(parents=True)
    shutil.copy2(
        SCENARIO_03 / "approval_policy.yaml",
        run_dir / "input_snapshot" / "approval_policy.yaml",
    )
    (run_dir / "context_packet.json").write_text(
        json.dumps(
            {
                "contract_id": "contract_003",
                "contract_type": "services_agreement",
                "counterparty_name_from_manifest": "Acme Services GmbH",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "clause_analysis.json").write_text(
        json.dumps(
            {
                "contract_id": "contract_003",
                "total_score": 90,
                "risk_tier": risk_tier,
                "findings": [
                    {
                        "finding_id": "RISK-001",
                        "category": category,
                        "risk_tier": risk_tier,
                        "score": 85,
                        "recommendation": "Review required.",
                        "evidence_ref": "x",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "audit_log.md").write_text("# Audit\n", encoding="utf-8")
    return run_dir


def test_triage_mock_writes_non_empty_artifacts(tmp_path: Path) -> None:
    run_dir = _seed_run_dir_for_triage(tmp_path, "high", "finance")

    outcome = run_triage_agent(run_dir)

    for name in ("exceptions.md", "approval_packet.json", "posting_payload.json"):
        path = run_dir / name
        assert path.is_file()
        assert path.stat().st_size > 0

    assert outcome["final_decision"] == "finance_review_required"
    assert outcome["overall_risk"] == "high"
    packet = json.loads((run_dir / "approval_packet.json").read_text())
    assert packet["contract_id"] == "contract_003"
    for key in (
        "final_decision",
        "overall_risk",
        "secondary_reviews",
        "approvers",
        "key_findings",
        "next_actions",
    ):
        assert key in packet
    payload = json.loads((run_dir / "posting_payload.json").read_text())
    for key in (
        "target_system",
        "contract_id",
        "counterparty",
        "contract_type",
        "risk_level",
        "status",
        "approval_required",
        "approvers",
        "key_findings",
    ):
        assert key in payload


def test_triage_mock_decision_mapping(tmp_path: Path) -> None:
    cases = {
        ("high", "compliance"): "compliance_review_required",
        ("high", "legal"): "legal_review_required",
        ("high", "finance"): "finance_review_required",
        ("medium", "finance"): "manual_review_required",
        ("low", "finance"): "auto_approve",
    }
    for (tier, category), expected in cases.items():
        run_dir = _seed_run_dir_for_triage(tmp_path / f"{tier}_{category}", tier, category)
        outcome = run_triage_agent(run_dir)
        assert outcome["final_decision"] == expected


def test_triage_mock_handles_missing_data(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "empty"
    run_dir.mkdir(parents=True)
    (run_dir / "audit_log.md").write_text("# Audit\n", encoding="utf-8")

    outcome = run_triage_agent(run_dir)

    assert outcome["final_decision"] == "manual_review_required"
    assert outcome["overall_risk"] == "unknown"
    assert (run_dir / "approval_packet.json").is_file()


def test_orchestrator_invokes_agent_h_only_through_run_triage_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The orchestrator must reach Agent H solely via run_triage_agent, so a
    replacement implementation needs no orchestrator changes."""
    calls: list[Path] = []

    def fake_triage(run_directory):  # type: ignore[no-untyped-def]
        run_dir = Path(run_directory)
        calls.append(run_dir)
        (run_dir / "exceptions.md").write_text("# replaced\n", encoding="utf-8")
        (run_dir / "approval_packet.json").write_text(
            json.dumps({"final_decision": "auto_approve"}), encoding="utf-8"
        )
        (run_dir / "posting_payload.json").write_text(
            json.dumps({"status": "ready_for_posting"}), encoding="utf-8"
        )
        return {"overall_risk": "low", "final_decision": "auto_approve"}

    monkeypatch.setattr("app.orchestrator.run_triage_agent", fake_triage)

    result = _run(tmp_path)

    assert result.status == "completed"
    assert len(calls) == 1
    assert calls[0] == Path(result.run_directory)
    assert result.final_decision == "auto_approve"


# ----------------------------------------------------------------------------
# Failure handling
# ----------------------------------------------------------------------------
def test_invalid_bundle_returns_failed_result(tmp_path: Path) -> None:
    empty_bundle = tmp_path / "empty_bundle"
    empty_bundle.mkdir()

    result = run_pipeline(str(empty_bundle), runs_root=tmp_path / "runs")

    assert result.status == "failed_at_validation_input"
    assert result.error_message
    # No run directory must have been created.
    assert not (tmp_path / "runs").exists() or not any(
        (tmp_path / "runs").iterdir()
    )


def test_failure_is_written_to_audit_and_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(run_directory):  # type: ignore[no-untyped-def]
        raise RuntimeError("synthetic triage failure")

    monkeypatch.setattr("app.orchestrator.run_triage_agent", boom)

    result = _run(tmp_path)

    assert result.status == "failed_at_triage"
    assert "synthetic triage failure" in (result.error_message or "")

    run_dir = Path(result.run_directory)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["status"] == "failed_at_triage"
    assert metrics["failed_agent"] == "triage"
    assert "synthetic triage failure" in metrics["error_message"]
    assert "finished_at" in metrics

    audit = (run_dir / "audit_log.md").read_text()
    assert "## Pipeline Failure" in audit
    assert "failed_at_triage" in audit


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def test_cli_main_runs(tmp_path: Path) -> None:
    from app.main import main

    exit_code = main(["--bundle", str(SCENARIO_03), "--runs-root", str(tmp_path / "runs")])
    assert exit_code == 0
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1


def test_cli_main_fails_for_invalid_bundle(tmp_path: Path) -> None:
    from app.main import main

    bad = tmp_path / "missing_bundle"
    exit_code = main(["--bundle", str(bad), "--runs-root", str(tmp_path / "runs")])
    assert exit_code == 1
