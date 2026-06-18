"""Agent H — Exception Triage & Final Decision (deterministic MVP / mock).

This module implements a **deterministic mock** of Agent H so the ICRAS pipeline
can run end-to-end and produce valid final artifacts. The real Agent H is being
built separately; the orchestrator only ever calls :func:`run_triage_agent`, so
the real implementation can replace this module later without any orchestrator
changes.

The mock reads the upstream run artifacts, derives an ``overall_risk`` and a
``final_decision`` deterministically from ``clause_analysis.json`` (best-effort
informed by ``approval_policy.yaml``), and writes the three final artifacts:

* ``exceptions.md``
* ``approval_packet.json``
* ``posting_payload.json``

It uses no LLM and never raises on missing/partial data: absent inputs degrade
gracefully to a ``manual_review_required`` / ``unknown`` outcome.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

CONTEXT_PACKET_FILENAME = "context_packet.json"
NORMALIZED_COUNTERPARTY_FILENAME = "normalized_counterparty.json"
VALIDATION_RESULT_FILENAME = "validation_result.json"
CLAUSE_ANALYSIS_FILENAME = "clause_analysis.json"
EVIDENCE_INDEX_FILENAME = "evidence_index.json"
APPROVAL_POLICY_SNAPSHOT = "input_snapshot/approval_policy.yaml"

EXCEPTIONS_FILENAME = "exceptions.md"
APPROVAL_PACKET_FILENAME = "approval_packet.json"
POSTING_PAYLOAD_FILENAME = "posting_payload.json"
AUDIT_LOG_FILENAME = "audit_log.md"

TARGET_SYSTEM = "ICRAS_CLM_MOCK"

# Final decision identifiers produced by the mock triage boundary.
DECISION_REJECT_OR_BLOCK = "reject_or_block"
DECISION_COMPLIANCE = "compliance_review_required"
DECISION_LEGAL = "legal_review_required"
DECISION_FINANCE = "finance_review_required"
DECISION_MANUAL = "manual_review_required"
DECISION_MANAGER = "manager_review"
DECISION_AUTO_APPROVE = "auto_approve"

# Deterministic fallback priority (higher wins) used when approval_policy.yaml
# does not provide a usable decision_priority mapping.
DEFAULT_DECISION_PRIORITY: dict[str, int] = {
    DECISION_REJECT_OR_BLOCK: 100,
    DECISION_COMPLIANCE: 90,
    DECISION_LEGAL: 80,
    DECISION_FINANCE: 70,
    DECISION_MANUAL: 60,
    DECISION_MANAGER: 50,
    DECISION_AUTO_APPROVE: 10,
}

# Maps a risk category to the review decision it triggers when the risk tier is
# high (or critical).
CATEGORY_TO_DECISION: dict[str, str] = {
    "compliance": DECISION_COMPLIANCE,
    "legal": DECISION_LEGAL,
    "finance": DECISION_FINANCE,
    "counterparty": DECISION_MANUAL,
    "manual_review": DECISION_MANUAL,
}

# Default approver routing per decision; refined from approval_policy.yaml when
# possible.
DECISION_TO_APPROVER: dict[str, str] = {
    DECISION_REJECT_OR_BLOCK: "executive_committee",
    DECISION_COMPLIANCE: "compliance_team",
    DECISION_LEGAL: "legal_team",
    DECISION_FINANCE: "finance_team",
    DECISION_MANUAL: "procurement_manager",
    DECISION_MANAGER: "procurement_manager",
    DECISION_AUTO_APPROVE: "none",
}


# ----------------------------------------------------------------------------
# Safe readers
# ----------------------------------------------------------------------------
def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object, returning ``None`` on any problem."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _read_yaml(path: Path) -> dict[str, Any] | None:
    """Read a YAML mapping, returning ``None`` on any problem."""
    if not path.is_file():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


# ----------------------------------------------------------------------------
# Decision logic
# ----------------------------------------------------------------------------
def _decision_priority(approval_policy: dict[str, Any] | None) -> dict[str, int]:
    """Resolve the decision priority map from policy, else the default."""
    if isinstance(approval_policy, dict):
        priority = approval_policy.get("decision_priority")
        if isinstance(priority, dict):
            cleaned = {
                str(key): int(value)
                for key, value in priority.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
            if cleaned:
                return cleaned
    return dict(DEFAULT_DECISION_PRIORITY)


def _approver_for_decision(
    decision: str,
    approval_policy: dict[str, Any] | None,
) -> str:
    """Best-effort approver lookup from approval_policy.yaml, else default."""
    default = DECISION_TO_APPROVER.get(decision, "manual_reviewer")
    if not isinstance(approval_policy, dict):
        return default

    # approval_rules entries look like {action, approver, risk_level, category}.
    rules = approval_policy.get("approval_rules")
    if isinstance(rules, list):
        action_aliases = {
            decision,
            decision.replace("_required", ""),
        }
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            action = str(rule.get("action", "")).strip()
            if action and action in action_aliases:
                approver = rule.get("approver")
                if isinstance(approver, str) and approver.strip():
                    return approver.strip()
    return default


def _candidate_decisions_from_findings(
    findings: list[dict[str, Any]],
) -> list[str]:
    """Map each high-severity finding category to a review decision."""
    decisions: list[str] = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        category = str(finding.get("category", "")).strip().lower()
        decision = CATEGORY_TO_DECISION.get(category)
        if decision and decision not in decisions:
            decisions.append(decision)
    return decisions


def derive_decision(
    clause_analysis: dict[str, Any] | None,
    approval_policy: dict[str, Any] | None,
) -> tuple[str, str, list[str]]:
    """Derive ``(overall_risk, final_decision, secondary_reviews)``.

    Deterministic and crash-free. Missing data degrades to a manual review with
    an ``unknown`` overall risk.
    """
    if not isinstance(clause_analysis, dict):
        return "unknown", DECISION_MANUAL, []

    risk_tier = str(clause_analysis.get("risk_tier", "")).strip().lower()
    findings_raw = clause_analysis.get("findings")
    findings = findings_raw if isinstance(findings_raw, list) else []

    if risk_tier == "low":
        return "low", DECISION_AUTO_APPROVE, []

    if risk_tier == "medium":
        return "medium", DECISION_MANUAL, []

    if risk_tier in {"high", "critical"}:
        candidates = _candidate_decisions_from_findings(findings)
        if not candidates:
            return risk_tier, DECISION_MANUAL, []

        priority = _decision_priority(approval_policy)
        # Pick the highest-priority candidate; ties broken by first occurrence.
        primary = max(
            candidates,
            key=lambda decision: priority.get(decision, 0),
        )
        secondary = [
            decision for decision in candidates if decision != primary
        ]
        return risk_tier, primary, secondary

    # Unknown / unexpected tier.
    return "unknown", DECISION_MANUAL, []


# ----------------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------------
def _counterparty_name(
    context_packet: dict[str, Any] | None,
    normalized_counterparty: dict[str, Any] | None,
) -> str | None:
    """Resolve a display counterparty name from available artifacts."""
    if isinstance(normalized_counterparty, dict):
        for key in (
            "matched_vendor_name",
            "extracted_counterparty_name",
            "input_counterparty_name",
            "manifest_counterparty_name",
        ):
            value = normalized_counterparty.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(context_packet, dict):
        value = context_packet.get("counterparty_name_from_manifest")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _key_findings(
    clause_analysis: dict[str, Any] | None,
    normalized_counterparty: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Build a compact, deterministic list of key findings for the packet."""
    key_findings: list[dict[str, Any]] = []

    if isinstance(clause_analysis, dict):
        for finding in clause_analysis.get("findings", []) or []:
            if not isinstance(finding, dict):
                continue
            key_findings.append(
                {
                    "finding_id": finding.get("finding_id"),
                    "source": "agent_e",
                    "category": finding.get("category"),
                    "risk_tier": finding.get("risk_tier"),
                    "score": finding.get("score"),
                    "policy_rule": finding.get("policy_rule"),
                    "recommendation": finding.get("recommendation"),
                    "evidence_ref": finding.get("evidence_ref"),
                }
            )

    if isinstance(normalized_counterparty, dict):
        for finding in normalized_counterparty.get("findings", []) or []:
            if not isinstance(finding, dict):
                continue
            key_findings.append(
                {
                    "finding_id": finding.get("finding_id"),
                    "source": finding.get("source_agent", "agent_c"),
                    "category": finding.get("category"),
                    "risk_tier": finding.get("risk_level"),
                    "score": finding.get("score"),
                    "policy_rule": finding.get("policy_rule"),
                    "recommendation": finding.get("recommendation"),
                    "evidence_ref": finding.get("evidence_ref"),
                }
            )

    return key_findings


def _evidence_ids(
    clause_analysis: dict[str, Any] | None,
    normalized_counterparty: dict[str, Any] | None,
    evidence_index: dict[str, Any] | None,
) -> list[str]:
    """Collect distinct evidence identifiers referenced by the findings."""
    ids: list[str] = []

    def _add(value: Any) -> None:
        if isinstance(value, str) and value.strip() and value not in ids:
            ids.append(value)

    if isinstance(clause_analysis, dict):
        for finding in clause_analysis.get("findings", []) or []:
            if isinstance(finding, dict):
                _add(finding.get("clause_id"))

    if isinstance(normalized_counterparty, dict):
        for finding in normalized_counterparty.get("findings", []) or []:
            if isinstance(finding, dict):
                for evidence_id in finding.get("evidence_ids", []) or []:
                    _add(evidence_id)

    if isinstance(evidence_index, dict) and not ids:
        for item in evidence_index.get("evidence_items", []) or []:
            if isinstance(item, dict):
                _add(item.get("evidence_id"))

    return ids


def _next_actions(final_decision: str, approver: str) -> list[str]:
    """Deterministic next-action checklist for the chosen decision."""
    if final_decision == DECISION_AUTO_APPROVE:
        return ["No exceptions found. Auto-approve and post the contract."]
    if final_decision == DECISION_REJECT_OR_BLOCK:
        return ["Block the contract and escalate to the executive committee."]
    return [
        f"Route the contract to {approver} for {final_decision}.",
        "Resolve the listed exceptions before final approval.",
    ]


# ----------------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------------
def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_exceptions_md(
    path: Path,
    *,
    contract_id: str,
    contract_type: str | None,
    counterparty: str | None,
    overall_risk: str,
    final_decision: str,
    key_findings: list[dict[str, Any]],
    evidence_ids: list[str],
    next_actions: list[str],
) -> None:
    lines: list[str] = [
        f"# Exception Report — {contract_id}",
        "",
        f"- Contract ID: {contract_id}",
        f"- Contract type: {contract_type or 'unknown'}",
        f"- Counterparty: {counterparty or 'unknown'}",
        f"- Overall risk: {overall_risk}",
        f"- Final decision: {final_decision}",
        "",
        "## Key Findings",
    ]
    if key_findings:
        for finding in key_findings:
            finding_id = finding.get("finding_id") or "UNKNOWN"
            category = finding.get("category") or "n/a"
            tier = finding.get("risk_tier") or "n/a"
            score = finding.get("score")
            recommendation = finding.get("recommendation") or "Review required."
            lines.append(
                f"- {finding_id} [{category}/{tier}, score={score}]: "
                f"{recommendation}"
            )
    else:
        lines.append("- No risk findings were raised.")

    lines.extend(["", "## Evidence"])
    if evidence_ids:
        lines.append(f"- Evidence IDs: {', '.join(evidence_ids)}")
    else:
        lines.append("- No evidence references available.")

    lines.extend(["", "## Next Actions"])
    for action in next_actions:
        lines.append(f"- {action}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _append_audit_log(
    run_directory: Path,
    *,
    overall_risk: str,
    final_decision: str,
    approvers: list[str],
    key_findings: list[dict[str, Any]],
) -> None:
    audit_path = run_directory / AUDIT_LOG_FILENAME
    existing = (
        audit_path.read_text(encoding="utf-8") if audit_path.exists() else ""
    )
    approvers_text = ", ".join(approvers) if approvers else "none"
    lines = [
        "",
        "## Step 6: Exception Triage / Final Decision",
        "- Agent: H (deterministic mock)",
        f"- Overall risk: {overall_risk}",
        f"- Final decision: {final_decision}",
        f"- Approvers: {approvers_text}",
        f"- Key findings: {len(key_findings)}",
        "- Generated artifacts: exceptions.md, approval_packet.json, "
        "posting_payload.json",
        "- Status: triage_completed",
        "",
    ]
    audit_path.write_text(existing + "\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------
# Public stable interface
# ----------------------------------------------------------------------------
def run_triage_agent(run_directory: str | Path) -> dict[str, Any]:
    """Run the deterministic mock Agent H triage boundary for one run.

    This is the stable interface the orchestrator always calls. Replacing the
    body with the real Agent H must keep this signature and the three written
    artifacts so the orchestrator flow remains unchanged.

    Args:
        run_directory: The run directory created by Agent A.

    Returns:
        A dict with at least ``overall_risk``, ``final_decision``,
        ``approvers``, and ``key_findings``.
    """
    run_dir = Path(run_directory)

    context_packet = _read_json(run_dir / CONTEXT_PACKET_FILENAME)
    normalized_counterparty = _read_json(run_dir / NORMALIZED_COUNTERPARTY_FILENAME)
    validation_result = _read_json(run_dir / VALIDATION_RESULT_FILENAME)
    clause_analysis = _read_json(run_dir / CLAUSE_ANALYSIS_FILENAME)
    evidence_index = _read_json(run_dir / EVIDENCE_INDEX_FILENAME)
    approval_policy = _read_yaml(run_dir / APPROVAL_POLICY_SNAPSHOT)

    overall_risk, final_decision, secondary_decisions = derive_decision(
        clause_analysis, approval_policy
    )

    # Resolve identifiers with graceful fallbacks.
    contract_id = "unknown"
    for source in (clause_analysis, validation_result, context_packet):
        if isinstance(source, dict):
            value = source.get("contract_id")
            if isinstance(value, str) and value.strip():
                contract_id = value.strip()
                break

    contract_type = (
        context_packet.get("contract_type")
        if isinstance(context_packet, dict)
        else None
    )
    counterparty = _counterparty_name(context_packet, normalized_counterparty)
    key_findings = _key_findings(clause_analysis, normalized_counterparty)
    evidence_ids = _evidence_ids(
        clause_analysis, normalized_counterparty, evidence_index
    )

    primary_approver = _approver_for_decision(final_decision, approval_policy)
    secondary_reviews = [
        {
            "decision": decision,
            "approver": _approver_for_decision(decision, approval_policy),
        }
        for decision in secondary_decisions
    ]
    approvers: list[str] = []
    if primary_approver and primary_approver != "none":
        approvers.append(primary_approver)
    for review in secondary_reviews:
        approver = review["approver"]
        if approver and approver != "none" and approver not in approvers:
            approvers.append(approver)

    approval_required = final_decision != DECISION_AUTO_APPROVE
    next_actions = _next_actions(final_decision, primary_approver)

    # approval_packet.json
    approval_packet = {
        "contract_id": contract_id,
        "final_decision": final_decision,
        "overall_risk": overall_risk,
        "secondary_reviews": secondary_reviews,
        "approvers": approvers,
        "key_findings": key_findings,
        "next_actions": next_actions,
    }
    _write_json(run_dir / APPROVAL_PACKET_FILENAME, approval_packet)

    # posting_payload.json
    posting_payload = {
        "target_system": TARGET_SYSTEM,
        "contract_id": contract_id,
        "counterparty": counterparty,
        "contract_type": contract_type,
        "risk_level": overall_risk,
        "status": (
            "ready_for_posting"
            if final_decision == DECISION_AUTO_APPROVE
            else "pending_review"
        ),
        "approval_required": approval_required,
        "approvers": approvers,
        "key_findings": key_findings,
    }
    _write_json(run_dir / POSTING_PAYLOAD_FILENAME, posting_payload)

    # exceptions.md
    _write_exceptions_md(
        run_dir / EXCEPTIONS_FILENAME,
        contract_id=contract_id,
        contract_type=contract_type,
        counterparty=counterparty,
        overall_risk=overall_risk,
        final_decision=final_decision,
        key_findings=key_findings,
        evidence_ids=evidence_ids,
        next_actions=next_actions,
    )

    _append_audit_log(
        run_dir,
        overall_risk=overall_risk,
        final_decision=final_decision,
        approvers=approvers,
        key_findings=key_findings,
    )

    return {
        "overall_risk": overall_risk,
        "final_decision": final_decision,
        "approvers": approvers,
        "secondary_reviews": secondary_reviews,
        "key_findings": key_findings,
        "approval_required": approval_required,
        "contract_id": contract_id,
        "counterparty": counterparty,
    }
