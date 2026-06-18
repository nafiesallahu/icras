from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pydantic import ValidationError

from app.agents.triage_agent import (
    ApprovalPolicyValidationError,
    MalformedTriageJsonError,
    MissingTriageInputError,
    TriageConsistencyError,
    TriageEvidenceError,
    TriageSchemaValidationError,
    deduplicate_findings,
    prepare_triage_run,
)
from app.schemas.approval_policy import ApprovalPolicy
from app.schemas.triage_finding import TriageFinding

APPROVAL_POLICY_YAML = """\
version: "1.0"
owner: "ICRAS Team"
description: "Approval routing policy based on contract risk level and risk category."

approval_rules:
  - risk_level: "low"
    action: "auto_approve"
    approver: "none"

  - risk_level: "medium"
    action: "manager_review"
    approver: "procurement_manager"

  - risk_level: "high"
    category: "legal"
    action: "legal_review"
    approver: "legal_team"

  - risk_level: "high"
    category: "finance"
    action: "finance_review"
    approver: "finance_team"

  - risk_level: "high"
    category: "compliance"
    action: "compliance_review"
    approver: "compliance_team"

  - risk_level: "critical"
    action: "executive_review"
    approver: "executive_committee"

department_routing:
  finance:
    triggers:
      - "payment_terms_exceeded"

  legal:
    triggers:
      - "missing_liability_cap"
      - "missing_required_clause"
      - "auto_renewal_without_opt_out"
      - "jurisdiction_conflict"

  compliance:
    triggers:
      - "high_risk_jurisdiction"
      - "missing_gdpr_clause"

  procurement:
    triggers:
      - "unknown_vendor"
      - "medium_risk_contract"

  decision_priority:
    reject_or_block: 100
    compliance_review_required: 90
    legal_review_required: 80
    finance_review_required: 70
    manual_review_required: 60
    manager_review: 50
    auto_approve: 10
"""


def _write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    """Write a formatted JSON fixture."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(
    path: Path,
) -> dict[str, Any]:
    """Read one JSON fixture."""

    return json.loads(path.read_text(encoding="utf-8"))


def _build_complete_run(
    tmp_path: Path,
    *,
    run_name: str = "run_complete",
    contract_id: str = "contract_001",
) -> Path:
    """
    Create a complete valid Agent H run directory.

    The Agent D and Agent E findings intentionally describe the same
    payment-terms problem so the complete workflow tests cross-agent
    deduplication.
    """

    run_dir = tmp_path / run_name
    input_snapshot_dir = run_dir / "input_snapshot"

    input_snapshot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    context_packet = {
        "run_id": run_name,
        "bundle_id": "bundle_001",
        "contract_id": contract_id,
        "contract_type": "services_agreement",
        "input_file": "contract.pdf",
        "bundle_path": str(run_dir / "bundle"),
        "run_directory": str(run_dir),
        "input_snapshot_directory": str(input_snapshot_dir),
        "contract_sha256": "a" * 64,
        "received_timestamp": ("2026-06-18T10:00:00+00:00"),
        "document_type": "SERVICES_AGREEMENT",
        "jurisdiction": "Malta",
        "counterparty_name_from_manifest": ("Acme Services GmbH"),
        "contains_personal_data": False,
        "policy_files": {
            "approval_policy": str(input_snapshot_dir / "approval_policy.yaml")
        },
        "documents": [
            {
                "document_id": "doc-001",
                "filename": "contract.pdf",
            }
        ],
        "ignored_files": [],
        "initial_risk_indicators": [],
        "status": "intake_completed",
    }

    normalized_counterparty = {
        "input_counterparty_name": ("Acme Service GmbH"),
        "manifest_counterparty_name": ("Acme Services GmbH"),
        "extracted_counterparty_name": ("Acme Service GmbH"),
        "matched_vendor_id": "VENDOR-001",
        "matched_vendor_name": ("Acme Services GmbH"),
        "match_score": 92,
        "status": "high_risk",
        "risk_level": "high",
        "flags": [
            "high_risk_vendor",
        ],
        "findings": [
            {
                "finding_id": "C-001",
                "source_agent": "agent_c",
                "category": "counterparty",
                "severity": "high",
                "risk_level": "high",
                "score": 80,
                "confidence": 0.82,
                "finding_type": ("counterparty_mismatch"),
                "clause_type": "party_identification",
                "expected": ("Acme Services GmbH"),
                "actual": ("Acme Service GmbH"),
                "recommendation": (
                    "Confirm the legal counterparty " "before approval."
                ),
                "open_questions": [
                    "Confirm the counterparty's registered legal entity.",
                ],
                "evidence_ids": [],
                "requires_human_review": True,
            }
        ],
    }

    validation_result = {
        "schema_version": "1.0",
        "contract_id": contract_id,
        "validation_status": ("completed_with_findings"),
        "normalized_fields": {
            "payment_terms_days": 90,
            "governing_law": "Malta",
            "effective_date": "2026-06-01",
            "liability_cap_present": True,
            "gdpr_clause_present": True,
            "auto_renewal_present": False,
            "opt_out_window_days": None,
            "counterparty_name": ("Acme Service GmbH"),
            "has_low_confidence_extraction": (False),
        },
        "findings": [
            {
                "finding_id": "VAL-001",
                "field": "payment_terms_days",
                "severity": "high",
                "message": ("Payment terms exceed company " "policy."),
                "evidence_ref": ("contract.pdf#page=3&" "clause=payment_terms_001"),
                "clause_id": ("payment_terms_001"),
                "page_number": 3,
                "bbox": None,
                "policy_rule": ("payment_terms_exceeded"),
                "expected_value": "30",
                "actual_value": "90",
                "recommendation": ("Review the payment terms."),
            }
        ],
    }

    clause_analysis = {
        "schema_version": "1.0",
        "contract_id": contract_id,
        "total_score": 95,
        "risk_tier": "critical",
        "jurisdictions_evaluated": [
            "Malta",
        ],
        "findings": [
            {
                "finding_id": "RISK-001",
                "source_finding_id": "VAL-001",
                "field": "payment_terms_days",
                "category": "finance",
                "risk_tier": "critical",
                "score": 95,
                "policy_rule": ("payment_terms_exceeded"),
                "expected_value": "30",
                "actual_value": "90",
                "recommendation": (
                    "Reduce payment terms to 30 days " "before approval."
                ),
                "evidence_ref": ("contract.pdf#page=3&" "clause=payment_terms_001"),
                "clause_id": ("payment_terms_001"),
                "page_number": 3,
                "bbox": None,
                "tolerance_band": ("outside_policy"),
            }
        ],
    }

    evidence_index = {
        "documents": [
            {
                "document_id": "doc-001",
                "filename": "contract.pdf",
            }
        ],
        "evidence_items": [
            {
                "evidence_id": "EV-001",
                "document_id": "doc-001",
                "filename": "contract.pdf",
                "page": 3,
                "clause_id": ("payment_terms_001"),
                "bbox": None,
                "text_excerpt": ("Payment shall be made within " "ninety days."),
            }
        ],
    }

    _write_json(
        run_dir / "context_packet.json",
        context_packet,
    )

    _write_json(
        run_dir / "normalized_counterparty.json",
        normalized_counterparty,
    )

    _write_json(
        run_dir / "validation_result.json",
        validation_result,
    )

    _write_json(
        run_dir / "clause_analysis.json",
        clause_analysis,
    )

    _write_json(
        run_dir / "evidence_index.json",
        evidence_index,
    )

    (input_snapshot_dir / "approval_policy.yaml").write_text(
        APPROVAL_POLICY_YAML,
        encoding="utf-8",
    )

    return run_dir


def test_policy_loader_supports_nested_priority(
    tmp_path: Path,
) -> None:
    run_dir = _build_complete_run(tmp_path)

    policy = ApprovalPolicy.from_yaml_file(
        run_dir / "input_snapshot" / "approval_policy.yaml"
    )

    assert list(policy.department_routing) == [
        "finance",
        "legal",
        "compliance",
        "procurement",
    ]

    assert policy.decision_priority["legal_review_required"] == 80

    assert "decision_priority" not in (policy.department_routing)

    assert policy.priority_for_action("auto_approve") == 10

    assert policy.priority_for_action("legal_review") == 80

    assert policy.priority_for_action("finance_review") == 70

    assert policy.priority_for_action("compliance_review") == 90

    assert policy.priority_for_action("executive_review") == 100


def test_policy_action_without_priority_is_rejected() -> None:
    with pytest.raises(
        ValidationError,
        match=("Approval actions without " "decision priorities"),
    ):
        ApprovalPolicy.model_validate(
            {
                "version": "1.0",
                "owner": "ICRAS Team",
                "description": ("Policy containing an " "unranked action."),
                "approval_rules": [
                    {
                        "risk_level": "high",
                        "action": "unknown_review",
                        "approver": "unknown_team",
                    }
                ],
                "department_routing": {},
                "decision_priority": {
                    "auto_approve": 10,
                },
            }
        )


def test_agent_c_flag_without_finding_is_collected(
    tmp_path: Path,
) -> None:
    run_dir = _build_complete_run(tmp_path)

    counterparty_path = run_dir / "normalized_counterparty.json"

    counterparty = _read_json(counterparty_path)

    counterparty["matched_vendor_id"] = None
    counterparty["matched_vendor_name"] = None
    counterparty["match_score"] = None
    counterparty["status"] = "unknown"
    counterparty["risk_level"] = "medium"
    counterparty["flags"] = [
        "unknown_vendor",
    ]
    counterparty["findings"] = []

    _write_json(
        counterparty_path,
        counterparty,
    )

    result = prepare_triage_run(run_dir)

    # One Agent C flag plus the Agent D/E findings.
    assert result.findings_received == 3

    # Agent D and E still merge into one finding.
    assert result.duplicates_removed == 1
    assert len(result.consolidated_findings) == 2

    assert result.findings_by_source == {
        "agent_c": 1,
        "agent_d": 1,
        "agent_e": 1,
    }

    flag_finding = next(
        finding
        for finding in result.consolidated_findings
        if finding.finding_type == "unknown_vendor"
    )

    assert flag_finding.category.value == ("counterparty")

    assert flag_finding.severity.value == "medium"
    assert flag_finding.risk_level.value == "medium"
    assert flag_finding.score == 50

    assert flag_finding.source_agents == ["agent_c"]

    assert flag_finding.requires_human_review is True


def test_overlapping_evidence_merges_and_preserves_location() -> None:
    first = TriageFinding(
        finding_id="TRIAGE-EVIDENCE-A",
        source_finding_ids=["VAL-200"],
        source_agents=["agent_d"],
        finding_type="payment_terms_exceeded",
        title="Payment terms exceed policy",
        category="finance",
        severity="high",
        risk_level="high",
        score=75,
        clause_id="payment_terms_001",
        field_reference="payment_terms_days",
        policy_rule="payment_terms_exceeded",
        expected_value="30",
        actual_value="90",
        evidence_ids=["EV-SHARED"],
        document_name="contract.pdf",
        page_number=3,
        bbox={
            "x0": 72.1,
            "top": 412.5,
            "x1": 510.8,
            "bottom": 455.2,
        },
    )

    second = TriageFinding(
        finding_id="TRIAGE-EVIDENCE-B",
        source_finding_ids=["RISK-200"],
        source_agents=["agent_e"],
        finding_type="excessive_payment_terms",
        title="Excessive payment period",
        category="finance",
        severity="critical",
        risk_level="critical",
        score=95,
        clause_id="payment_terms_001",
        field_reference="payment_terms_days",
        policy_rule="payment_terms_exceeded",
        expected_value="30",
        actual_value="90",
        evidence_ids=[
            "EV-SHARED",
            "EV-EXTRA",
        ],
        text_excerpt=(
            "Payment shall be made within " "ninety days of invoice receipt."
        ),
    )

    results, removed = deduplicate_findings([first, second])

    assert removed == 1
    assert len(results) == 1

    merged = results[0]

    assert merged.source_agents == [
        "agent_d",
        "agent_e",
    ]

    assert merged.evidence_ids == [
        "EV-EXTRA",
        "EV-SHARED",
    ]

    assert merged.severity.value == "critical"
    assert merged.risk_level.value == "critical"
    assert merged.score == 95

    # Location data came from both findings.
    assert merged.document_name == "contract.pdf"
    assert merged.page_number == 3

    assert merged.bbox is not None
    assert merged.bbox.model_dump(mode="json") == {
        "x0": 72.1,
        "top": 412.5,
        "x1": 510.8,
        "bottom": 455.2,
    }

    assert merged.text_excerpt == (
        "Payment shall be made within " "ninety days of invoice receipt."
    )


def test_shared_evidence_does_not_merge_unique_findings() -> None:
    legal_finding = TriageFinding(
        finding_id="LEGAL-001",
        source_finding_ids=["VAL-L"],
        source_agents=["agent_d"],
        finding_type="missing_liability_cap",
        title="Missing liability cap",
        category="legal",
        severity="high",
        risk_level="high",
        score=75,
        policy_rule="missing_liability_cap",
        expected_value="present",
        actual_value="missing",
        evidence_ids=["EV-SHARED"],
    )

    finance_finding = TriageFinding(
        finding_id="FINANCE-001",
        source_finding_ids=["VAL-F"],
        source_agents=["agent_d"],
        finding_type="payment_terms_exceeded",
        title="Payment terms exceeded",
        category="finance",
        severity="high",
        risk_level="high",
        score=75,
        policy_rule="payment_terms_exceeded",
        expected_value="30",
        actual_value="90",
        evidence_ids=["EV-SHARED"],
    )

    results, removed = deduplicate_findings(
        [
            legal_finding,
            finance_finding,
        ]
    )

    assert removed == 0
    assert len(results) == 2

    assert {finding.finding_type for finding in results} == {
        "missing_liability_cap",
        "payment_terms_exceeded",
    }


def test_prepare_triage_run_normalizes_and_deduplicates(
    tmp_path: Path,
) -> None:
    run_dir = _build_complete_run(tmp_path)

    result = prepare_triage_run(run_dir)

    assert result.run_id == "run_complete"
    assert result.contract_id == "contract_001"

    assert result.findings_received == 4
    assert result.duplicates_removed == 1

    assert result.findings_by_source == {
        "agent_c": 2,
        "agent_d": 1,
        "agent_e": 1,
    }

    assert len(result.consolidated_findings) == 3

    payment_finding = next(
        finding
        for finding in result.consolidated_findings
        if finding.finding_type == "payment_terms_exceeded"
    )

    assert payment_finding.category.value == "finance"
    assert payment_finding.severity.value == "critical"
    assert payment_finding.risk_level.value == "critical"
    assert payment_finding.score == 95

    assert payment_finding.source_agents == [
        "agent_d",
        "agent_e",
    ]

    assert payment_finding.source_finding_ids == [
        "RISK-001",
        "VAL-001",
    ]

    assert payment_finding.evidence_ids == [
        "EV-001",
    ]

    assert payment_finding.document_name == ("contract.pdf")

    assert payment_finding.page_number == 3

    assert payment_finding.text_excerpt == ("Payment shall be made within ninety days.")

    assert payment_finding.recommendation == (
        "Reduce payment terms to 30 days " "before approval."
    )

    counterparty_finding = next(
        finding
        for finding in result.consolidated_findings
        if finding.finding_type == "counterparty_mismatch"
    )

    assert counterparty_finding.confidence == 0.82

    assert counterparty_finding.clause_type == ("party_identification")

    assert counterparty_finding.open_questions == [
        "Confirm the counterparty's registered legal entity.",
    ]

    assert counterparty_finding.category.value == ("counterparty")

    assert counterparty_finding.source_agents == ["agent_c"]

    flag_finding = next(
        finding
        for finding in result.consolidated_findings
        if finding.finding_type == "high_risk_vendor"
    )

    assert flag_finding.category.value == "counterparty"
    assert flag_finding.severity.value == "high"
    assert flag_finding.risk_level.value == "high"
    assert flag_finding.score == 75
    assert flag_finding.source_agents == ["agent_c"]
    assert flag_finding.requires_human_review is True


def test_findings_are_sorted_deterministically(
    tmp_path: Path,
) -> None:
    run_dir = _build_complete_run(tmp_path)

    first_result = prepare_triage_run(run_dir)
    second_result = prepare_triage_run(run_dir)

    first_dump = first_result.model_dump(mode="json")

    second_dump = second_result.model_dump(mode="json")

    assert first_dump == second_dump

    # The critical finance finding should appear before
    # the high counterparty finding.
    assert (
        first_result.consolidated_findings[0].finding_type == "payment_terms_exceeded"
    )


def test_missing_required_file_is_rejected(
    tmp_path: Path,
) -> None:
    run_dir = _build_complete_run(tmp_path)

    (run_dir / "validation_result.json").unlink()

    with pytest.raises(
        MissingTriageInputError,
        match="validation_result.json",
    ):
        prepare_triage_run(run_dir)


def test_malformed_json_is_rejected(
    tmp_path: Path,
) -> None:
    run_dir = _build_complete_run(tmp_path)

    (run_dir / "validation_result.json").write_text(
        '{"contract_id": ',
        encoding="utf-8",
    )

    with pytest.raises(
        MalformedTriageJsonError,
        match="Malformed JSON",
    ):
        prepare_triage_run(run_dir)


def test_invalid_json_schema_is_rejected(
    tmp_path: Path,
) -> None:
    run_dir = _build_complete_run(tmp_path)

    validation_path = run_dir / "validation_result.json"

    validation_result = _read_json(validation_path)

    del validation_result["contract_id"]

    _write_json(
        validation_path,
        validation_result,
    )

    with pytest.raises(
        TriageSchemaValidationError,
        match="schema validation",
    ):
        prepare_triage_run(run_dir)


def test_malformed_approval_policy_is_rejected(
    tmp_path: Path,
) -> None:
    run_dir = _build_complete_run(tmp_path)

    policy_path = run_dir / "input_snapshot" / "approval_policy.yaml"

    policy_path.write_text(
        "approval_rules: [\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ApprovalPolicyValidationError,
        match="approval policy failed validation",
    ):
        prepare_triage_run(run_dir)


def test_run_id_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    run_dir = _build_complete_run(tmp_path)

    context_path = run_dir / "context_packet.json"

    context_packet = _read_json(context_path)
    context_packet["run_id"] = "different_run"

    _write_json(
        context_path,
        context_packet,
    )

    with pytest.raises(
        TriageConsistencyError,
        match="Run ID mismatch",
    ):
        prepare_triage_run(run_dir)


def test_contract_id_mismatch_is_rejected(
    tmp_path: Path,
) -> None:
    run_dir = _build_complete_run(tmp_path)

    clause_path = run_dir / "clause_analysis.json"

    clause_analysis = _read_json(clause_path)

    clause_analysis["contract_id"] = "different_contract"

    _write_json(
        clause_path,
        clause_analysis,
    )

    with pytest.raises(
        TriageConsistencyError,
        match="Contract ID mismatch",
    ):
        prepare_triage_run(run_dir)


def test_failed_validation_status_is_rejected(
    tmp_path: Path,
) -> None:
    run_dir = _build_complete_run(tmp_path)

    validation_path = run_dir / "validation_result.json"

    validation_result = _read_json(validation_path)

    validation_result["validation_status"] = "failed"

    _write_json(
        validation_path,
        validation_result,
    )

    with pytest.raises(
        TriageConsistencyError,
        match="reports status 'failed'",
    ):
        prepare_triage_run(run_dir)


def test_unknown_evidence_reference_is_rejected(
    tmp_path: Path,
) -> None:
    run_dir = _build_complete_run(tmp_path)

    validation_path = run_dir / "validation_result.json"

    validation_result = _read_json(validation_path)

    validation_result["findings"][0]["evidence_ref"] = "EV-999"

    _write_json(
        validation_path,
        validation_result,
    )

    with pytest.raises(
        TriageEvidenceError,
        match="could not be resolved",
    ):
        prepare_triage_run(run_dir)


def test_duplicate_evidence_ids_are_rejected(
    tmp_path: Path,
) -> None:
    run_dir = _build_complete_run(tmp_path)

    evidence_path = run_dir / "evidence_index.json"

    evidence_index = _read_json(evidence_path)

    duplicate_item = dict(evidence_index["evidence_items"][0])

    duplicate_item["document_id"] = "doc-002"

    evidence_index["evidence_items"].append(duplicate_item)

    _write_json(
        evidence_path,
        evidence_index,
    )

    with pytest.raises(
        TriageEvidenceError,
        match="Duplicate evidence ID",
    ):
        prepare_triage_run(run_dir)


def test_exact_duplicates_merge_deterministically() -> None:
    first = TriageFinding(
        finding_id="TRIAGE-FIRST",
        source_finding_ids=["VAL-100"],
        source_agents=["agent_d"],
        finding_type="missing_liability_cap",
        title="Liability cap is missing",
        category="legal",
        severity="high",
        risk_level="high",
        score=75,
        policy_rule="missing_liability_cap",
        expected_value="present",
        actual_value="missing",
        recommendation=("Review the liability provision."),
        requires_human_review=True,
        evidence_ids=["EV-100"],
    )

    second = TriageFinding(
        finding_id="TRIAGE-SECOND",
        source_finding_ids=["RISK-100"],
        source_agents=["agent_e"],
        finding_type="missing_liability_cap",
        title=("Required liability cap is missing"),
        category="legal",
        severity="critical",
        risk_level="critical",
        score=98,
        policy_rule="missing_liability_cap",
        expected_value="present",
        actual_value="missing",
        recommendation=(
            "Add an approved liability cap before " "the contract is signed."
        ),
        requires_human_review=True,
        evidence_ids=["EV-101"],
    )

    forward_results, forward_removed = deduplicate_findings([first, second])

    reversed_results, reversed_removed = deduplicate_findings([second, first])

    assert forward_removed == 1
    assert reversed_removed == 1

    assert len(forward_results) == 1
    assert len(reversed_results) == 1

    assert forward_results[0].model_dump(mode="json") == reversed_results[0].model_dump(
        mode="json"
    )

    merged = forward_results[0]

    assert merged.source_agents == [
        "agent_d",
        "agent_e",
    ]

    assert merged.source_finding_ids == [
        "RISK-100",
        "VAL-100",
    ]

    assert merged.severity.value == "critical"
    assert merged.risk_level.value == "critical"
    assert merged.score == 98

    assert merged.evidence_ids == [
        "EV-100",
        "EV-101",
    ]
