"""Per-scenario expected-decision tests for the ICRAS pipeline.

These tests run the full pipeline (Agent A -> B -> C -> D -> E -> H) on each of
the ten curated contract bundles in deterministic synthetic mode
(``ENV_MODE=test``) and assert that every scenario resolves to its expected
final decision and overall risk tier.

They lock in the business outcomes of the decision logic so a regression in
Agent D (validation) or Agent H (triage/routing) is caught immediately:

* clean contracts (NDA + services agreement) auto-approve,
* finance/legal/compliance/manual exceptions route to the correct review queue,
* multiple independent high risks reject/block.

Determinism is guaranteed by synthetic extraction (fixed fixtures), a fixed
runs root per scenario, and the deterministic Agent H artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.orchestrator import EXPECTED_FINAL_ARTIFACTS, run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLES_DIR = REPO_ROOT / "data" / "bundles"

# Expected final decision per scenario, plus the acceptable overall risk
# tier(s). Decisions are exact; risk tiers allow the documented alternatives
# (the requirement is on the decision, with risk "high or critical" for several
# scenarios).
EXPECTED: dict[str, dict[str, object]] = {
    "scenario_01_clean_nda": {
        "decision": "auto_approve",
        "risk": {"low"},
    },
    "scenario_02_missing_liability_cap": {
        "decision": "legal_review_required",
        "risk": {"high", "critical"},
    },
    "scenario_03_net_90_payment_terms": {
        "decision": "finance_review_required",
        "risk": {"high", "critical"},
    },
    "scenario_04_high_risk_jurisdiction": {
        "decision": "compliance_review_required",
        "risk": {"high", "critical"},
    },
    "scenario_05_auto_renewal_no_opt_out": {
        "decision": "legal_review_required",
        "risk": {"high"},
    },
    "scenario_06_conflicting_governing_law": {
        "decision": "legal_review_required",
        "risk": {"high", "critical"},
    },
    "scenario_07_low_confidence_signature": {
        "decision": "manual_review_required",
        "risk": {"medium", "high"},
    },
    "scenario_08_missing_gdpr_clause": {
        "decision": "compliance_review_required",
        "risk": {"high", "critical"},
    },
    "scenario_09_clean_services_agreement": {
        "decision": "auto_approve",
        "risk": {"low"},
    },
    "scenario_10_multiple_high_risks": {
        "decision": "reject_or_block",
        "risk": {"critical"},
    },
}


@pytest.fixture(autouse=True)
def _force_synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force deterministic synthetic extraction for every scenario run."""
    monkeypatch.setenv("ENV_MODE", "test")
    monkeypatch.delenv("MOCK_PIPELINE", raising=False)


def _run(bundle_id: str, tmp_path: Path):
    bundle_path = BUNDLES_DIR / bundle_id
    return run_pipeline(str(bundle_path), runs_root=tmp_path / "runs" / bundle_id)


@pytest.mark.parametrize("bundle_id", sorted(EXPECTED))
def test_scenario_pipeline_completes_with_all_artifacts(
    bundle_id: str, tmp_path: Path
) -> None:
    result = _run(bundle_id, tmp_path)

    assert result.status == "completed", (
        f"{bundle_id} did not complete: {result.status} ({result.error_message})"
    )

    run_dir = Path(result.run_directory)
    for name in EXPECTED_FINAL_ARTIFACTS:
        artifact = run_dir / name
        assert artifact.exists(), f"{bundle_id}: missing artifact {name}"
        assert artifact.stat().st_size > 0, f"{bundle_id}: empty artifact {name}"


@pytest.mark.parametrize("bundle_id", sorted(EXPECTED))
def test_scenario_final_decision_matches_expected(
    bundle_id: str, tmp_path: Path
) -> None:
    result = _run(bundle_id, tmp_path)
    expected = EXPECTED[bundle_id]

    run_dir = Path(result.run_directory)
    packet = json.loads((run_dir / "approval_packet.json").read_text("utf-8"))
    payload = json.loads((run_dir / "posting_payload.json").read_text("utf-8"))

    # The pipeline result, the approval packet, and the posting payload must all
    # agree on the final decision.
    assert result.final_decision == expected["decision"], (
        f"{bundle_id}: expected {expected['decision']!r}, "
        f"got {result.final_decision!r}"
    )
    assert packet["final_decision"] == expected["decision"]
    assert payload["final_decision"] == expected["decision"]

    assert result.overall_risk in expected["risk"], (
        f"{bundle_id}: overall risk {result.overall_risk!r} "
        f"not in {sorted(expected['risk'])}"
    )
    assert packet["overall_risk"] == result.overall_risk


def test_clean_scenarios_auto_approve(tmp_path: Path) -> None:
    for bundle_id in (
        "scenario_01_clean_nda",
        "scenario_09_clean_services_agreement",
    ):
        result = _run(bundle_id, tmp_path)
        assert result.final_decision == "auto_approve", (
            f"{bundle_id}: clean contract must auto-approve, "
            f"got {result.final_decision!r}"
        )
        assert result.overall_risk == "low"
        payload = json.loads(
            (Path(result.run_directory) / "posting_payload.json").read_text("utf-8")
        )
        assert payload["status"] == "ready_for_posting"


def test_clean_nda_never_escalates(tmp_path: Path) -> None:
    # Regression guard: the clean NDA must not become manual_review_required or
    # critical due to contract-type-blind mandatory-field validation.
    result = _run("scenario_01_clean_nda", tmp_path)
    assert result.final_decision == "auto_approve"
    assert result.final_decision != "manual_review_required"
    assert result.overall_risk != "critical"


def test_multiple_high_risks_reject_or_block(tmp_path: Path) -> None:
    result = _run("scenario_10_multiple_high_risks", tmp_path)
    assert result.final_decision == "reject_or_block"
    assert result.overall_risk == "critical"
    payload = json.loads(
        (Path(result.run_directory) / "posting_payload.json").read_text("utf-8")
    )
    assert payload["status"] == "blocked"


def test_scenario_decisions_are_deterministic(tmp_path: Path) -> None:
    # Running a scenario twice produces byte-identical final artifacts (run_id
    # is the only run-specific field and is excluded from the comparison).
    bundle_id = "scenario_10_multiple_high_risks"
    first = _run(bundle_id, tmp_path / "a")
    second = _run(bundle_id, tmp_path / "b")

    assert first.final_decision == second.final_decision
    assert first.overall_risk == second.overall_risk

    for name in ("approval_packet.json", "posting_payload.json"):
        first_data = json.loads((Path(first.run_directory) / name).read_text("utf-8"))
        second_data = json.loads((Path(second.run_directory) / name).read_text("utf-8"))
        first_data.pop("run_id", None)
        second_data.pop("run_id", None)
        assert first_data == second_data, f"non-deterministic artifact: {name}"
