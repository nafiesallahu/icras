"""Tests for the real Agent H — Exception Triage & Lead Orchestrator.

These tests exercise Agent H directly (without the orchestrator) by seeding a
run directory with the upstream artifacts of Agents C, D and E, then asserting
on the merge/dedupe/prioritize/decision/routing behavior and the determinism
and audit/metrics guarantees.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.agents.triage_agent import AgentHError, run_triage_agent

REPO_ROOT = Path(__file__).resolve().parents[1]
APPROVAL_POLICY = (
    REPO_ROOT
    / "data"
    / "bundles"
    / "scenario_03_net_90_payment_terms"
    / "approval_policy.yaml"
)


# ----------------------------------------------------------------------------
# Seeding helpers
# ----------------------------------------------------------------------------
def _seed(
    run_dir: Path,
    *,
    clause_analysis: dict[str, Any] | None = None,
    validation_result: dict[str, Any] | None = None,
    normalized_counterparty: dict[str, Any] | None = None,
    evidence_index: dict[str, Any] | None = None,
    context_packet: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    with_policy: bool = True,
) -> Path:
    """Create a run directory populated with the requested upstream artifacts."""
    (run_dir / "input_snapshot").mkdir(parents=True, exist_ok=True)
    if with_policy:
        (run_dir / "input_snapshot" / "approval_policy.yaml").write_text(
            APPROVAL_POLICY.read_text(encoding="utf-8"), encoding="utf-8"
        )

    if context_packet is None:
        context_packet = {
            "contract_id": "contract_003",
            "contract_type": "services_agreement",
            "counterparty_name_from_manifest": "Acme Services GmbH",
        }
    (run_dir / "context_packet.json").write_text(
        json.dumps(context_packet), encoding="utf-8"
    )

    if clause_analysis is not None:
        (run_dir / "clause_analysis.json").write_text(
            json.dumps(clause_analysis), encoding="utf-8"
        )
    if validation_result is not None:
        (run_dir / "validation_result.json").write_text(
            json.dumps(validation_result), encoding="utf-8"
        )
    if normalized_counterparty is not None:
        (run_dir / "normalized_counterparty.json").write_text(
            json.dumps(normalized_counterparty), encoding="utf-8"
        )
    if evidence_index is not None:
        (run_dir / "evidence_index.json").write_text(
            json.dumps(evidence_index), encoding="utf-8"
        )
    if metrics is not None:
        (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    (run_dir / "audit_log.md").write_text("# Audit\n", encoding="utf-8")
    return run_dir


def _scored_finding(
    finding_id: str,
    *,
    category: str,
    risk_tier: str,
    score: int,
    source_finding_id: str | None = None,
    policy_rule: str | None = None,
    field: str | None = None,
    clause_id: str | None = None,
    evidence_ref: str | None = None,
    recommendation: str = "Review required.",
) -> dict[str, Any]:
    return {
        "finding_id": finding_id,
        "source_finding_id": source_finding_id,
        "category": category,
        "risk_tier": risk_tier,
        "score": score,
        "policy_rule": policy_rule,
        "field": field,
        "clause_id": clause_id,
        "evidence_ref": evidence_ref,
        "recommendation": recommendation,
    }


def _clause_analysis(findings: list[dict[str, Any]], risk_tier: str) -> dict[str, Any]:
    return {
        "contract_id": "contract_003",
        "total_score": 100,
        "risk_tier": risk_tier,
        "findings": findings,
    }


# ----------------------------------------------------------------------------
# Artifact generation
# ----------------------------------------------------------------------------
def test_writes_three_final_artifacts(tmp_path: Path) -> None:
    run_dir = _seed(
        tmp_path / "run",
        clause_analysis=_clause_analysis(
            [_scored_finding("RISK-001", category="finance", risk_tier="high", score=85)],
            "high",
        ),
    )

    result = run_triage_agent(run_dir)

    for name in ("exceptions.md", "approval_packet.json", "posting_payload.json"):
        path = run_dir / name
        assert path.is_file(), f"missing {name}"
        assert path.stat().st_size > 0, f"empty {name}"

    assert result["final_decision"] == "finance_review_required"
    assert result["overall_risk"] == "high"

    packet = json.loads((run_dir / "approval_packet.json").read_text())
    for key in (
        "schema_version",
        "deterministic",
        "run_id",
        "bundle_id",
        "final_decision",
        "overall_risk",
        "primary_approval_path",
        "secondary_reviews",
        "approvers",
        "prioritized_findings",
        "key_findings",
        "exception_categories",
        "evidence_summary",
        "dedupe_summary",
        "next_actions",
    ):
        assert key in packet, f"missing packet key: {key}"
    assert packet["schema_version"] == "1.0"
    assert packet["deterministic"] is True
    assert "reason" in packet["primary_approval_path"]

    payload = json.loads((run_dir / "posting_payload.json").read_text())
    for key in (
        "schema_version",
        "run_id",
        "bundle_id",
        "target_system",
        "contract_id",
        "counterparty",
        "contract_type",
        "risk_level",
        "status",
        "approval_required",
        "approvers",
        "summary",
        "key_findings",
    ):
        assert key in payload, f"missing payload key: {key}"
    assert payload["schema_version"] == "1.0"
    for key in (
        "finding_count",
        "critical_count",
        "high_count",
        "medium_count",
        "low_count",
    ):
        assert key in payload["summary"], f"missing summary key: {key}"


# ----------------------------------------------------------------------------
# Extended output fields (schema_version / run_id / bundle_id / reasons / summary)
# ----------------------------------------------------------------------------
def test_extended_output_fields_from_context_packet(tmp_path: Path) -> None:
    run_dir = _seed(
        tmp_path / "run",
        context_packet={
            "run_id": "run_test_001",
            "bundle_id": "scenario_03_net_90_payment_terms",
            "contract_id": "contract_003",
            "contract_type": "services_agreement",
            "counterparty_name_from_manifest": "Acme Services GmbH",
            "jurisdiction": "Germany",
            "contains_personal_data": True,
        },
        # finance (70) + compliance (90) high -> compliance primary, finance secondary.
        clause_analysis=_clause_analysis(
            [
                _scored_finding("RISK-F", category="finance", risk_tier="high", score=85, policy_rule="f"),
                _scored_finding("RISK-C", category="compliance", risk_tier="high", score=80, policy_rule="c"),
            ],
            "high",
        ),
    )

    run_triage_agent(run_dir)

    packet = json.loads((run_dir / "approval_packet.json").read_text())
    assert packet["schema_version"] == "1.0"
    assert packet["deterministic"] is True
    assert packet["run_id"] == "run_test_001"
    assert packet["bundle_id"] == "scenario_03_net_90_payment_terms"
    assert packet["primary_approval_path"]["reason"]
    assert packet["secondary_reviews"], "expected a secondary review"
    secondary = packet["secondary_reviews"][0]
    assert secondary["decision"] == "finance_review_required"
    assert secondary["reason"]
    assert secondary["related_finding_ids"] == ["RISK-F"]

    payload = json.loads((run_dir / "posting_payload.json").read_text())
    assert payload["schema_version"] == "1.0"
    assert payload["run_id"] == "run_test_001"
    assert payload["bundle_id"] == "scenario_03_net_90_payment_terms"
    assert payload["summary"] == {
        "finding_count": 2,
        "critical_count": 0,
        "high_count": 2,
        "medium_count": 0,
        "low_count": 0,
    }

    exceptions = (run_dir / "exceptions.md").read_text()
    assert "- Run ID: run_test_001" in exceptions
    assert "- Bundle ID: scenario_03_net_90_payment_terms" in exceptions
    assert "- Jurisdiction: Germany" in exceptions
    assert "- Contains personal data: True" in exceptions

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["overall_risk"] == "high"
    assert metrics["final_decision"] == "compliance_review_required"
    assert metrics["approval_required"] is True


def test_run_and_bundle_id_default_to_unknown(tmp_path: Path) -> None:
    run_dir = _seed(
        tmp_path / "run",
        context_packet={"contract_id": "contract_003"},
        clause_analysis=_clause_analysis(
            [_scored_finding("RISK-001", category="finance", risk_tier="high", score=85, policy_rule="x")],
            "high",
        ),
    )
    run_triage_agent(run_dir)
    packet = json.loads((run_dir / "approval_packet.json").read_text())
    assert packet["run_id"] == "unknown"
    assert packet["bundle_id"] == "unknown"


# ----------------------------------------------------------------------------
# Merge across agents C, D, E
# ----------------------------------------------------------------------------
def test_merges_findings_from_agents_c_d_e(tmp_path: Path) -> None:
    run_dir = _seed(
        tmp_path / "run",
        # Agent E: one finance finding (distinct from the others).
        clause_analysis=_clause_analysis(
            [
                _scored_finding(
                    "RISK-100",
                    category="finance",
                    risk_tier="high",
                    score=85,
                    policy_rule="payment_terms.max_allowed_days",
                    field="payment_terms_days",
                )
            ],
            "high",
        ),
        # Agent D: a distinct legal validation finding (no Agent E equivalent).
        validation_result={
            "contract_id": "contract_003",
            "validation_status": "completed_with_findings",
            "findings": [
                {
                    "finding_id": "VAL-050",
                    "field": "liability_cap_present",
                    "severity": "high",
                    "message": "Missing liability cap clause.",
                    "evidence_ref": "missing:liability_cap",
                    "policy_rule": "liability_cap.required",
                }
            ],
        },
        # Agent C: a high-risk counterparty finding.
        normalized_counterparty={
            "status": "high_risk",
            "risk_level": "high",
            "flags": ["high_risk_counterparty"],
            "findings": [
                {
                    "finding_id": "CP-010",
                    "source_agent": "agent_c",
                    "category": "counterparty",
                    "severity": "high",
                    "risk_level": "high",
                    "score": 70,
                    "finding_type": "high_risk_counterparty",
                    "recommendation": "Review counterparty.",
                }
            ],
        },
    )

    result = run_triage_agent(run_dir)

    sources = {f["source"] for f in result["prioritized_findings"]}
    assert sources == {"agent_e", "agent_d", "agent_c"}
    ids = {f["finding_id"] for f in result["prioritized_findings"]}
    assert ids == {"RISK-100", "VAL-050", "CP-010"}
    assert result["dedupe_summary"]["by_source"] == {
        "agent_c": 1,
        "agent_d": 1,
        "agent_e": 1,
    }


# ----------------------------------------------------------------------------
# Deduplication
# ----------------------------------------------------------------------------
def test_deduplicates_agent_e_supersedes_agent_d(tmp_path: Path) -> None:
    run_dir = _seed(
        tmp_path / "run",
        clause_analysis=_clause_analysis(
            [
                _scored_finding(
                    "RISK-001",
                    category="finance",
                    risk_tier="high",
                    score=85,
                    source_finding_id="VAL-001",
                    policy_rule="payment_terms.max_allowed_days",
                    field="payment_terms_days",
                    clause_id="payment_terms_001",
                    evidence_ref="contract.pdf#clause=payment_terms_001",
                )
            ],
            "high",
        ),
        validation_result={
            "contract_id": "contract_003",
            "validation_status": "completed_with_findings",
            "findings": [
                {
                    "finding_id": "VAL-001",
                    "field": "payment_terms_days",
                    "severity": "high",
                    "message": "Payment terms exceed limit.",
                    "evidence_ref": "contract.pdf#clause=payment_terms_001",
                    "policy_rule": "payment_terms.max_allowed_days",
                    "clause_id": "payment_terms_001",
                }
            ],
        },
    )

    result = run_triage_agent(run_dir)

    assert result["dedupe_summary"]["input_count"] == 2
    assert result["dedupe_summary"]["output_count"] == 1
    assert result["dedupe_summary"]["duplicates_removed"] == 1
    kept = result["prioritized_findings"]
    assert len(kept) == 1
    assert kept[0]["finding_id"] == "RISK-001"
    assert kept[0]["source"] == "agent_e"


def test_dedupe_is_deterministic_regardless_of_repeat(tmp_path: Path) -> None:
    seed_kwargs = dict(
        clause_analysis=_clause_analysis(
            [
                _scored_finding(
                    "RISK-001", category="finance", risk_tier="high", score=85,
                    source_finding_id="VAL-001", policy_rule="r1", field="f1",
                ),
                _scored_finding(
                    "RISK-002", category="legal", risk_tier="high", score=80,
                    source_finding_id="VAL-002", policy_rule="r2", field="f2",
                ),
            ],
            "high",
        ),
    )
    first = run_triage_agent(_seed(tmp_path / "a", **seed_kwargs))
    second = run_triage_agent(_seed(tmp_path / "b", **seed_kwargs))
    assert first["dedupe_summary"] == second["dedupe_summary"]
    assert first["prioritized_findings"] == second["prioritized_findings"]


# ----------------------------------------------------------------------------
# Prioritization
# ----------------------------------------------------------------------------
def test_prioritizes_findings_by_score(tmp_path: Path) -> None:
    run_dir = _seed(
        tmp_path / "run",
        clause_analysis=_clause_analysis(
            [
                _scored_finding("RISK-LOW", category="finance", risk_tier="high", score=70, policy_rule="a"),
                _scored_finding("RISK-HIGH", category="finance", risk_tier="high", score=95, policy_rule="b"),
                _scored_finding("RISK-MID", category="finance", risk_tier="high", score=80, policy_rule="c"),
            ],
            "high",
        ),
    )

    result = run_triage_agent(run_dir)
    ordered = [f["finding_id"] for f in result["prioritized_findings"]]
    assert ordered == ["RISK-HIGH", "RISK-MID", "RISK-LOW"]
    assert [f["priority_rank"] for f in result["prioritized_findings"]] == [1, 2, 3]


def test_blocking_finding_sorts_first(tmp_path: Path) -> None:
    run_dir = _seed(
        tmp_path / "run",
        clause_analysis=_clause_analysis(
            [
                _scored_finding("RISK-A", category="finance", risk_tier="high", score=99, policy_rule="a"),
                _scored_finding("RISK-B", category="legal", risk_tier="critical", score=50, policy_rule="b"),
            ],
            "critical",
        ),
    )
    result = run_triage_agent(run_dir)
    assert result["prioritized_findings"][0]["finding_id"] == "RISK-B"
    assert result["prioritized_findings"][0]["blocking"] is True


# ----------------------------------------------------------------------------
# Decision priority + routing
# ----------------------------------------------------------------------------
def test_applies_decision_priority_highest_wins(tmp_path: Path) -> None:
    # finance (70) + compliance (90) both high -> compliance wins by priority.
    run_dir = _seed(
        tmp_path / "run",
        clause_analysis=_clause_analysis(
            [
                _scored_finding("RISK-F", category="finance", risk_tier="high", score=85, policy_rule="f"),
                _scored_finding("RISK-C", category="compliance", risk_tier="high", score=85, policy_rule="c"),
            ],
            "high",
        ),
    )
    result = run_triage_agent(run_dir)
    assert result["final_decision"] == "compliance_review_required"


def test_preserves_secondary_reviews(tmp_path: Path) -> None:
    run_dir = _seed(
        tmp_path / "run",
        clause_analysis=_clause_analysis(
            [
                _scored_finding("RISK-F", category="finance", risk_tier="high", score=85, policy_rule="f"),
                _scored_finding("RISK-C", category="compliance", risk_tier="high", score=85, policy_rule="c"),
                _scored_finding("RISK-L", category="legal", risk_tier="high", score=85, policy_rule="l"),
            ],
            "high",
        ),
    )
    result = run_triage_agent(run_dir)
    assert result["final_decision"] == "compliance_review_required"
    secondary = [r["decision"] for r in result["secondary_reviews"]]
    # legal (80) then finance (70), ordered by descending priority.
    assert secondary == ["legal_review_required", "finance_review_required"]
    # Approvers carry the primary first, then secondary, unique.
    assert result["approvers"][0] == "compliance_team"
    assert set(result["approvers"]) >= {"compliance_team", "legal_team", "finance_team"}


def test_decision_priority_from_policy_overrides_default(tmp_path: Path) -> None:
    # Custom policy makes finance the highest priority decision.
    policy_dir = tmp_path / "run" / "input_snapshot"
    run_dir = _seed(
        tmp_path / "run",
        clause_analysis=_clause_analysis(
            [
                _scored_finding("RISK-F", category="finance", risk_tier="high", score=85, policy_rule="f"),
                _scored_finding("RISK-C", category="compliance", risk_tier="high", score=85, policy_rule="c"),
            ],
            "high",
        ),
        with_policy=False,
    )
    (policy_dir).mkdir(parents=True, exist_ok=True)
    (policy_dir / "approval_policy.yaml").write_text(
        "decision_priority:\n"
        "  finance_review_required: 99\n"
        "  compliance_review_required: 10\n",
        encoding="utf-8",
    )
    result = run_triage_agent(run_dir)
    assert result["final_decision"] == "finance_review_required"


# ----------------------------------------------------------------------------
# Decision outcomes: auto_approve / reject_or_block
# ----------------------------------------------------------------------------
def test_no_findings_routes_to_auto_approve(tmp_path: Path) -> None:
    run_dir = _seed(
        tmp_path / "run",
        clause_analysis=_clause_analysis([], "low"),
        normalized_counterparty={"status": "approved", "risk_level": "low", "flags": [], "findings": []},
    )
    result = run_triage_agent(run_dir)
    assert result["final_decision"] == "auto_approve"
    assert result["overall_risk"] == "low"
    assert result["approvers"] == []
    payload = json.loads((run_dir / "posting_payload.json").read_text())
    assert payload["status"] == "ready_for_posting"
    assert payload["approval_required"] is False


def test_low_risk_findings_still_auto_approve(tmp_path: Path) -> None:
    run_dir = _seed(
        tmp_path / "run",
        clause_analysis=_clause_analysis(
            [_scored_finding("RISK-1", category="finance", risk_tier="low", score=10, policy_rule="x")],
            "low",
        ),
    )
    result = run_triage_agent(run_dir)
    assert result["final_decision"] == "auto_approve"


def test_critical_finding_routes_to_reject_or_block(tmp_path: Path) -> None:
    run_dir = _seed(
        tmp_path / "run",
        clause_analysis=_clause_analysis(
            [_scored_finding("RISK-1", category="legal", risk_tier="critical", score=100, policy_rule="x")],
            "critical",
        ),
    )
    result = run_triage_agent(run_dir)
    assert result["final_decision"] == "reject_or_block"
    assert "executive_committee" in result["approvers"]
    payload = json.loads((run_dir / "posting_payload.json").read_text())
    assert payload["status"] == "blocked"


def test_medium_risk_routes_to_manual_review(tmp_path: Path) -> None:
    run_dir = _seed(
        tmp_path / "run",
        clause_analysis=_clause_analysis(
            [_scored_finding("RISK-1", category="finance", risk_tier="medium", score=40, policy_rule="x")],
            "medium",
        ),
    )
    result = run_triage_agent(run_dir)
    assert result["final_decision"] == "manual_review_required"


# ----------------------------------------------------------------------------
# Evidence
# ----------------------------------------------------------------------------
def test_preserves_evidence_refs_when_available(tmp_path: Path) -> None:
    run_dir = _seed(
        tmp_path / "run",
        clause_analysis=_clause_analysis(
            [
                _scored_finding(
                    "RISK-001",
                    category="finance",
                    risk_tier="high",
                    score=85,
                    policy_rule="payment_terms.max_allowed_days",
                    clause_id="payment_terms_001",
                    evidence_ref="contract.pdf#clause=payment_terms_001",
                )
            ],
            "high",
        ),
        evidence_index={
            "documents": [],
            "evidence_items": [
                {
                    "evidence_id": "EV-002",
                    "document_id": "doc_001",
                    "filename": "contract.pdf",
                    "page": 1,
                    "clause_id": "payment_terms_001",
                    "text_excerpt": "Payment Terms ...",
                }
            ],
        },
    )

    result = run_triage_agent(run_dir)
    summary = result["evidence_summary"]
    assert "EV-002" in summary["evidence_ids"]
    assert "contract.pdf#clause=payment_terms_001" in summary["evidence_refs"]
    assert any(item["evidence_id"] == "EV-002" for item in summary["items"])


def test_does_not_invent_evidence(tmp_path: Path) -> None:
    run_dir = _seed(
        tmp_path / "run",
        clause_analysis=_clause_analysis(
            [_scored_finding("RISK-001", category="finance", risk_tier="high", score=85, policy_rule="x")],
            "high",
        ),
        evidence_index={"documents": [], "evidence_items": []},
    )
    result = run_triage_agent(run_dir)
    # No evidence items in the index -> no resolved evidence IDs are fabricated.
    assert result["evidence_summary"]["evidence_ids"] == []
    assert result["evidence_summary"]["items"] == []


# ----------------------------------------------------------------------------
# Audit log + metrics
# ----------------------------------------------------------------------------
def test_appends_to_audit_log_without_clobbering(tmp_path: Path) -> None:
    run_dir = _seed(
        tmp_path / "run",
        clause_analysis=_clause_analysis(
            [_scored_finding("RISK-001", category="finance", risk_tier="high", score=85, policy_rule="x")],
            "high",
        ),
    )
    (run_dir / "audit_log.md").write_text("# Audit\n\n## Step 5: Risk Scoring\n", encoding="utf-8")

    run_triage_agent(run_dir)

    audit = (run_dir / "audit_log.md").read_text()
    assert "## Step 5: Risk Scoring" in audit  # earlier content preserved
    assert "## Step 6: Exception Triage / Final Decision" in audit
    assert "triage_completed" in audit


def test_updates_metrics_without_deleting_earlier_agents(tmp_path: Path) -> None:
    run_dir = _seed(
        tmp_path / "run",
        clause_analysis=_clause_analysis(
            [_scored_finding("RISK-001", category="finance", risk_tier="high", score=85, policy_rule="x")],
            "high",
        ),
        metrics={
            "agents": {"agent_c": {"status": "completed", "risk_level": "low"}},
            "agents_completed": ["intake", "extraction", "counterparty", "validation", "risk_scoring"],
            "contract_id": "contract_003",
        },
    )

    run_triage_agent(run_dir)

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["agents"]["agent_c"] == {"status": "completed", "risk_level": "low"}
    assert metrics["agents"]["agent_h"]["status"] == "completed"
    assert metrics["agents"]["agent_h"]["final_decision"] == "finance_review_required"
    # Pre-existing top-level keys are not destroyed.
    assert metrics["contract_id"] == "contract_003"
    assert "risk_scoring" in metrics["agents_completed"]


def test_creates_metrics_when_absent(tmp_path: Path) -> None:
    run_dir = _seed(
        tmp_path / "run",
        clause_analysis=_clause_analysis(
            [_scored_finding("RISK-001", category="finance", risk_tier="high", score=85, policy_rule="x")],
            "high",
        ),
    )
    run_triage_agent(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["agents"]["agent_h"]["status"] == "completed"


# ----------------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------------
def test_repeated_runs_produce_identical_outputs(tmp_path: Path) -> None:
    seed_kwargs = dict(
        clause_analysis=_clause_analysis(
            [
                _scored_finding("RISK-001", category="finance", risk_tier="high", score=85, policy_rule="f", clause_id="c1", evidence_ref="e1"),
                _scored_finding("RISK-002", category="legal", risk_tier="high", score=80, policy_rule="l", clause_id="c2", evidence_ref="e2"),
            ],
            "critical",
        ),
        normalized_counterparty={"status": "approved", "risk_level": "low", "flags": [], "findings": []},
        evidence_index={"documents": [], "evidence_items": []},
    )
    run_a = _seed(tmp_path / "a", **seed_kwargs)
    run_b = _seed(tmp_path / "b", **seed_kwargs)
    run_triage_agent(run_a)
    run_triage_agent(run_b)

    for name in ("approval_packet.json", "posting_payload.json", "exceptions.md"):
        assert (run_a / name).read_text() == (run_b / name).read_text(), f"non-deterministic: {name}"


def test_rerun_on_same_dir_is_idempotent(tmp_path: Path) -> None:
    run_dir = _seed(
        tmp_path / "run",
        clause_analysis=_clause_analysis(
            [_scored_finding("RISK-001", category="finance", risk_tier="high", score=85, policy_rule="x")],
            "high",
        ),
    )
    run_triage_agent(run_dir)
    first = {
        name: (run_dir / name).read_text()
        for name in ("approval_packet.json", "posting_payload.json", "exceptions.md")
    }
    run_triage_agent(run_dir)
    for name, content in first.items():
        assert (run_dir / name).read_text() == content, f"not idempotent: {name}"


# ----------------------------------------------------------------------------
# Error handling
# ----------------------------------------------------------------------------
def test_invalid_required_input_raises_agent_h_error(tmp_path: Path) -> None:
    run_dir = _seed(
        tmp_path / "run",
        clause_analysis=_clause_analysis(
            [_scored_finding("RISK-001", category="finance", risk_tier="high", score=85, policy_rule="x")],
            "high",
        ),
    )
    # Corrupt an existing required artifact.
    (run_dir / "clause_analysis.json").write_text("{ not valid json", encoding="utf-8")

    with pytest.raises(AgentHError) as exc:
        run_triage_agent(run_dir)
    assert "clause_analysis.json" in str(exc.value)


def test_wrong_shape_required_input_raises(tmp_path: Path) -> None:
    run_dir = _seed(
        tmp_path / "run",
        clause_analysis=_clause_analysis([], "low"),
    )
    # A JSON array where an object is required.
    (run_dir / "validation_result.json").write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(AgentHError):
        run_triage_agent(run_dir)


def test_missing_inputs_degrade_gracefully(tmp_path: Path) -> None:
    # Only an audit log present: missing (not corrupt) inputs must not raise.
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "audit_log.md").write_text("# Audit\n", encoding="utf-8")

    result = run_triage_agent(run_dir)
    assert result["final_decision"] == "manual_review_required"
    assert result["overall_risk"] == "unknown"
    assert (run_dir / "approval_packet.json").is_file()
