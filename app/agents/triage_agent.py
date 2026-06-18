"""Agent H — Exception Triage & Lead Orchestrator (deterministic, file-based).

Agent H is the final decision-making and triage stage of the ICRAS pipeline. It
consolidates every upstream finding (Agents C, D, E), removes duplicates,
prioritizes the most important risks, applies rule-based decision and approval
routing logic, determines the final contract action, and emits the final
human-readable and machine-readable artifacts.

It reads from the run directory created by Agent A:

* ``context_packet.json``            (Agent A — run metadata)
* ``normalized_counterparty.json``   (Agent C — counterparty resolution)
* ``validation_result.json``         (Agent D — field/rule validation)
* ``clause_analysis.json``           (Agent E — scored risk findings)
* ``evidence_index.json``            (evidence references; never invented)
* ``input_snapshot/approval_policy.yaml`` (decision priority + routing)
* ``obligations.csv``                (optional, best-effort)

It writes:

* ``exceptions.md``        — human-readable exception report
* ``approval_packet.json`` — machine-readable approval/routing packet
* ``posting_payload.json`` — downstream posting payload

And updates (without destroying earlier agents' data):

* ``audit_log.md``
* ``metrics.json``

Determinism guarantees: no timestamps/random IDs in the three final artifacts,
sorted JSON keys, stable list ordering, explicit tie-breakers, and no external
service / LLM calls. Running Agent H twice on identical run artifacts produces
byte-identical ``exceptions.md``, ``approval_packet.json`` and
``posting_payload.json``.

The orchestrator only ever calls :func:`run_triage_agent`; this module preserves
that stable interface and the return keys ``overall_risk`` and ``final_decision``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

# ----------------------------------------------------------------------------
# Input / output filenames
# ----------------------------------------------------------------------------
CONTEXT_PACKET_FILENAME = "context_packet.json"
NORMALIZED_COUNTERPARTY_FILENAME = "normalized_counterparty.json"
VALIDATION_RESULT_FILENAME = "validation_result.json"
CLAUSE_ANALYSIS_FILENAME = "clause_analysis.json"
EVIDENCE_INDEX_FILENAME = "evidence_index.json"
OBLIGATIONS_FILENAME = "obligations.csv"
APPROVAL_POLICY_SNAPSHOT = "input_snapshot/approval_policy.yaml"

EXCEPTIONS_FILENAME = "exceptions.md"
APPROVAL_PACKET_FILENAME = "approval_packet.json"
POSTING_PAYLOAD_FILENAME = "posting_payload.json"
AUDIT_LOG_FILENAME = "audit_log.md"
METRICS_FILENAME = "metrics.json"

TARGET_SYSTEM = "ICRAS_CLM_MOCK"

# Required upstream JSON artifacts. When any of these files *exists* but cannot
# be parsed as a JSON object, Agent H raises a clear error. A *missing* file
# degrades gracefully (the run may legitimately lack an upstream artifact).
REQUIRED_JSON_INPUTS = (
    CONTEXT_PACKET_FILENAME,
    NORMALIZED_COUNTERPARTY_FILENAME,
    VALIDATION_RESULT_FILENAME,
    CLAUSE_ANALYSIS_FILENAME,
    EVIDENCE_INDEX_FILENAME,
)

# ----------------------------------------------------------------------------
# Final decisions
# ----------------------------------------------------------------------------
DECISION_REJECT_OR_BLOCK = "reject_or_block"
DECISION_COMPLIANCE = "compliance_review_required"
DECISION_LEGAL = "legal_review_required"
DECISION_FINANCE = "finance_review_required"
DECISION_MANUAL = "manual_review_required"
DECISION_MANAGER = "manager_review"
DECISION_AUTO_APPROVE = "auto_approve"

ALL_DECISIONS = (
    DECISION_REJECT_OR_BLOCK,
    DECISION_COMPLIANCE,
    DECISION_LEGAL,
    DECISION_FINANCE,
    DECISION_MANUAL,
    DECISION_MANAGER,
    DECISION_AUTO_APPROVE,
)

# Deterministic fallback priority (higher wins) used when approval_policy.yaml
# does not provide a usable ``decision_priority`` mapping.
DEFAULT_DECISION_PRIORITY: dict[str, int] = {
    DECISION_REJECT_OR_BLOCK: 100,
    DECISION_COMPLIANCE: 90,
    DECISION_LEGAL: 80,
    DECISION_FINANCE: 70,
    DECISION_MANUAL: 60,
    DECISION_MANAGER: 50,
    DECISION_AUTO_APPROVE: 10,
}

# Maps a risk/business category to the review decision it triggers when the
# finding is high (or critical) but not outright blocking.
CATEGORY_TO_DECISION: dict[str, str] = {
    "compliance": DECISION_COMPLIANCE,
    "legal": DECISION_LEGAL,
    "finance": DECISION_FINANCE,
    "counterparty": DECISION_MANUAL,
    "manual_review": DECISION_MANUAL,
}

# Fallback approver routing per decision, used when approval_policy.yaml cannot
# resolve an approver.
DECISION_TO_APPROVER: dict[str, str] = {
    DECISION_REJECT_OR_BLOCK: "executive_committee",
    DECISION_COMPLIANCE: "compliance_team",
    DECISION_LEGAL: "legal_team",
    DECISION_FINANCE: "finance_team",
    DECISION_MANUAL: "procurement_manager",
    DECISION_MANAGER: "procurement_manager",
    DECISION_AUTO_APPROVE: "none",
}

# Policy ``action`` aliases (approval_policy.yaml uses bare actions such as
# ``finance_review`` / ``executive_review``).
DECISION_ACTION_ALIASES: dict[str, set[str]] = {
    DECISION_REJECT_OR_BLOCK: {
        "reject_or_block", "reject", "block", "executive_review",
        "executive_committee_review",
    },
    DECISION_COMPLIANCE: {"compliance_review_required", "compliance_review"},
    DECISION_LEGAL: {"legal_review_required", "legal_review"},
    DECISION_FINANCE: {"finance_review_required", "finance_review"},
    DECISION_MANUAL: {
        "manual_review_required", "manual_review", "procurement_review",
    },
    DECISION_MANAGER: {"manager_review", "manager_review_required"},
    DECISION_AUTO_APPROVE: {"auto_approve"},
}

# Ordering helpers (higher number = more severe / more authoritative).
TIER_RANK: dict[str, int] = {"low": 1, "medium": 2, "high": 3, "critical": 4}
SOURCE_RANK: dict[str, int] = {"agent_e": 3, "agent_d": 2, "agent_c": 1}

# Finding types / flags that force a blocking (reject_or_block) decision.
BLOCKING_SIGNALS = {"reject", "block", "blocking", "prohibited", "reject_or_block"}


class AgentHError(Exception):
    """Raised when Agent H cannot run because a required input is invalid.

    A *missing* upstream artifact degrades gracefully; this error is reserved
    for inputs that are present but corrupt (unparseable or the wrong shape).
    """


# ----------------------------------------------------------------------------
# Safe readers
# ----------------------------------------------------------------------------
def _read_required_json(path: Path) -> dict[str, Any] | None:
    """Read a required JSON object.

    Returns ``None`` when the file is absent (graceful degradation). Raises
    :class:`AgentHError` when the file exists but is not a valid JSON object.
    """
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - filesystem failure
        raise AgentHError(f"Agent H could not read required input '{path.name}': {exc}") from exc
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise AgentHError(
            f"Agent H received invalid JSON in required input '{path.name}': {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise AgentHError(
            f"Agent H expected a JSON object in '{path.name}', got {type(data).__name__}."
        )
    return data


def _read_yaml(path: Path) -> dict[str, Any] | None:
    """Read a YAML mapping, returning ``None`` on any problem (graceful)."""
    if not path.is_file():
        return None
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return None
    return data if isinstance(data, dict) else None


# ----------------------------------------------------------------------------
# Small deterministic utilities
# ----------------------------------------------------------------------------
def _clean_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _as_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return 0
    return 0


def _tier_from(value: Any) -> str:
    tier = (str(value).strip().lower() if value is not None else "")
    return tier if tier in TIER_RANK else ""


def _max_tier(tiers: list[str]) -> str:
    ranked = [(TIER_RANK[t], t) for t in tiers if t in TIER_RANK]
    if not ranked:
        return ""
    return max(ranked)[1]


def _infer_category(text_parts: list[str | None]) -> str:
    """Best-effort category inference for findings lacking an explicit one."""
    text = " ".join(part for part in text_parts if part).lower()
    if not text:
        return "manual_review"
    if any(k in text for k in ("payment", "fee", "price", "invoice", "finance")):
        return "finance"
    if "gdpr" in text or "personal data" in text or "privacy" in text or "data_protection" in text:
        return "compliance"
    if "jurisdiction" in text and ("high_risk" in text or "high risk" in text):
        return "compliance"
    if "jurisdiction" in text or "governing" in text:
        return "legal"
    if any(k in text for k in ("liability", "clause", "renewal", "indemn", "law", "legal")):
        return "legal"
    if "counterparty" in text or "vendor" in text:
        return "counterparty"
    return "manual_review"


# ----------------------------------------------------------------------------
# Finding normalization (merge upstream findings into one shared structure)
# ----------------------------------------------------------------------------
def _normalize_evidence_refs(*values: Any) -> list[str]:
    refs: list[str] = []
    for value in values:
        if value is None:
            continue
        items = value if isinstance(value, list) else [value]
        for item in items:
            ref = _clean_str(item)
            if ref and ref not in refs:
                refs.append(ref)
    return refs


def _is_blocking(severity: str, risk_tier: str, finding_type: str | None) -> bool:
    if severity == "critical" or risk_tier == "critical":
        return True
    ftype = (finding_type or "").strip().lower()
    return any(signal in ftype for signal in BLOCKING_SIGNALS)


def _normalize_agent_e(clause_analysis: dict[str, Any] | None) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if not isinstance(clause_analysis, dict):
        return findings
    for raw in clause_analysis.get("findings") or []:
        if not isinstance(raw, dict):
            continue
        category = (_clean_str(raw.get("category")) or "manual_review").lower()
        risk_tier = _tier_from(raw.get("risk_tier")) or "medium"
        finding_type = _clean_str(raw.get("finding_type"))
        findings.append(
            {
                "finding_id": _clean_str(raw.get("finding_id")) or "RISK-UNKNOWN",
                "source": "agent_e",
                "source_finding_id": _clean_str(raw.get("source_finding_id")),
                "category": category,
                "severity": risk_tier,
                "risk_tier": risk_tier,
                "score": _as_int(raw.get("score")),
                "finding_type": finding_type,
                "policy_rule": _clean_str(raw.get("policy_rule")),
                "field": _clean_str(raw.get("field")),
                "clause_id": _clean_str(raw.get("clause_id")),
                "evidence_refs": _normalize_evidence_refs(
                    raw.get("evidence_ref"), raw.get("clause_id")
                ),
                "recommendation": _clean_str(raw.get("recommendation")),
                "message": _clean_str(raw.get("recommendation")),
                "requires_human_review": True,
                "blocking": _is_blocking(risk_tier, risk_tier, finding_type),
            }
        )
    return findings


def _normalize_agent_d(validation_result: dict[str, Any] | None) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if not isinstance(validation_result, dict):
        return findings
    for raw in validation_result.get("findings") or []:
        if not isinstance(raw, dict):
            continue
        severity = (_clean_str(raw.get("severity")) or "medium").lower()
        field = _clean_str(raw.get("field"))
        policy_rule = _clean_str(raw.get("policy_rule"))
        message = _clean_str(raw.get("message"))
        category = _infer_category([field, policy_rule, message])
        findings.append(
            {
                "finding_id": _clean_str(raw.get("finding_id")) or "VAL-UNKNOWN",
                "source": "agent_d",
                "source_finding_id": None,
                "category": category,
                "severity": severity,
                "risk_tier": severity if severity in TIER_RANK else "medium",
                "score": _as_int(raw.get("score")),
                "finding_type": None,
                "policy_rule": policy_rule,
                "field": field,
                "clause_id": _clean_str(raw.get("clause_id")),
                "evidence_refs": _normalize_evidence_refs(
                    raw.get("evidence_ref"), raw.get("clause_id")
                ),
                "recommendation": _clean_str(raw.get("recommendation")),
                "message": message,
                "requires_human_review": severity in ("high", "critical"),
                "blocking": _is_blocking(severity, severity, None),
            }
        )
    return findings


def _normalize_agent_c(
    normalized_counterparty: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    if not isinstance(normalized_counterparty, dict):
        return findings

    for raw in normalized_counterparty.get("findings") or []:
        if not isinstance(raw, dict):
            continue
        severity = (_clean_str(raw.get("severity")) or "medium").lower()
        risk_level = (_clean_str(raw.get("risk_level")) or severity).lower()
        risk_tier = risk_level if risk_level in TIER_RANK else severity
        if risk_tier not in TIER_RANK:
            risk_tier = "medium"
        category = (_clean_str(raw.get("category")) or "counterparty").lower()
        finding_type = _clean_str(raw.get("finding_type"))
        findings.append(
            {
                "finding_id": _clean_str(raw.get("finding_id")) or "CP-UNKNOWN",
                "source": _clean_str(raw.get("source_agent")) or "agent_c",
                "source_finding_id": None,
                "category": category,
                "severity": severity,
                "risk_tier": risk_tier,
                "score": _as_int(raw.get("score")),
                "finding_type": finding_type,
                "policy_rule": _clean_str(raw.get("policy_rule")),
                "field": _clean_str(raw.get("field")),
                "clause_id": _clean_str(raw.get("clause_id")),
                "evidence_refs": _normalize_evidence_refs(
                    raw.get("evidence_ids"),
                    raw.get("evidence_ref"),
                    raw.get("clause_id"),
                ),
                "recommendation": _clean_str(raw.get("recommendation")),
                "message": _clean_str(raw.get("message")) or _clean_str(raw.get("recommendation")),
                "requires_human_review": bool(raw.get("requires_human_review", False)),
                "blocking": _is_blocking(severity, risk_tier, finding_type),
            }
        )

    # Derive a counterparty finding from the resolution status when Agent C did
    # not emit explicit findings but the counterparty is risky/unresolved. This
    # is derived from Agent C's own status only; it invents no evidence.
    if not findings:
        status = (_clean_str(normalized_counterparty.get("status")) or "").lower()
        risk_level = (_clean_str(normalized_counterparty.get("risk_level")) or "").lower()
        if status in ("unknown", "high_risk") or risk_level == "high":
            finding_type = {
                "unknown": "unknown_counterparty",
                "high_risk": "high_risk_counterparty",
            }.get(status, "counterparty_risk")
            findings.append(
                {
                    "finding_id": "CP-STATUS-001",
                    "source": "agent_c",
                    "source_finding_id": None,
                    "category": "counterparty",
                    "severity": "high",
                    "risk_tier": "high",
                    "score": 70,
                    "finding_type": finding_type,
                    "policy_rule": "counterparty.resolution",
                    "field": "counterparty_name",
                    "clause_id": None,
                    "evidence_refs": [],
                    "recommendation": (
                        "Manually review the counterparty before approval "
                        f"(status: {status or 'unresolved'})."
                    ),
                    "message": f"Counterparty resolution status: {status or 'unresolved'}.",
                    "requires_human_review": True,
                    "blocking": False,
                }
            )
    return findings


# ----------------------------------------------------------------------------
# Deduplication
# ----------------------------------------------------------------------------
def _dedupe_key(finding: dict[str, Any]) -> tuple[str, str, str, str]:
    """Stable cross-source identity for a finding (ignores the source agent)."""
    policy_rule = (finding.get("policy_rule") or "").lower()
    field = (finding.get("field") or "").lower()
    clause_id = (finding.get("clause_id") or "").lower()
    evidence = (finding.get("evidence_refs") or [""])
    evidence_first = (evidence[0] if evidence else "").lower()
    # Fall back to finding_type/message so findings without policy_rule still
    # collapse when they describe the same underlying issue.
    if not policy_rule and not field and not clause_id:
        policy_rule = (finding.get("finding_type") or finding.get("message") or "").lower()
    return (policy_rule, field, clause_id, evidence_first)


def _dedupe_findings(
    findings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deduplicate merged findings deterministically.

    Two passes:

    1. Drop Agent C/D findings whose ``finding_id`` is referenced by an Agent E
       finding's ``source_finding_id`` (Agent E re-scored the same issue).
    2. Collapse findings that share a cross-source identity key, keeping the
       most authoritative source (E > D > C), then the higher score, then the
       lexicographically smallest ``finding_id``.
    """
    input_count = len(findings)

    superseded_ids = {
        f["source_finding_id"]
        for f in findings
        if f.get("source") == "agent_e" and f.get("source_finding_id")
    }
    stage_one = [
        f
        for f in findings
        if not (f.get("source") != "agent_e" and f.get("finding_id") in superseded_ids)
    ]

    def _better(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
        c_rank = (
            SOURCE_RANK.get(candidate.get("source", ""), 0),
            candidate.get("score", 0),
        )
        cur_rank = (
            SOURCE_RANK.get(current.get("source", ""), 0),
            current.get("score", 0),
        )
        if c_rank != cur_rank:
            return c_rank > cur_rank
        # Final tie-breaker: smallest finding_id wins (stable, deterministic).
        return str(candidate.get("finding_id", "")) < str(current.get("finding_id", ""))

    kept: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for finding in stage_one:
        key = _dedupe_key(finding)
        existing = kept.get(key)
        if existing is None or _better(finding, existing):
            kept[key] = finding

    output = list(kept.values())
    by_source = {"agent_c": 0, "agent_d": 0, "agent_e": 0}
    for finding in output:
        src = finding.get("source", "")
        if src in by_source:
            by_source[src] += 1
    summary = {
        "input_count": input_count,
        "duplicates_removed": input_count - len(output),
        "output_count": len(output),
        "by_source": by_source,
    }
    return output, summary


# ----------------------------------------------------------------------------
# Prioritization
# ----------------------------------------------------------------------------
def _prioritize(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return findings sorted by importance (most important first)."""
    return sorted(
        findings,
        key=lambda f: (
            0 if f.get("blocking") else 1,
            -int(f.get("score", 0)),
            -TIER_RANK.get(f.get("risk_tier", ""), 0),
            -SOURCE_RANK.get(f.get("source", ""), 0),
            str(f.get("finding_id", "")),
        ),
    )


# ----------------------------------------------------------------------------
# Decision logic
# ----------------------------------------------------------------------------
def _decision_priority(approval_policy: dict[str, Any] | None) -> dict[str, int]:
    """Resolve the decision-priority map from policy, else the default."""
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


def _decision_for_finding(finding: dict[str, Any]) -> str | None:
    """Map a single high/critical finding to the review decision it triggers."""
    if finding.get("blocking"):
        return DECISION_REJECT_OR_BLOCK
    if finding.get("risk_tier") not in ("high", "critical"):
        return None
    category = (finding.get("category") or "").lower()
    return CATEGORY_TO_DECISION.get(category, DECISION_MANUAL)


def _select_decision(
    findings: list[dict[str, Any]],
    overall_tier: str,
    priority: dict[str, int],
) -> tuple[str, list[str], dict[str, list[str]]]:
    """Return ``(primary_decision, secondary_decisions, decision_to_finding_ids)``.

    Deterministic. ``decision_to_finding_ids`` records which findings drove each
    candidate decision (sorted) so the exception report can explain routing.
    """
    decision_to_ids: dict[str, list[str]] = {}
    for finding in findings:
        decision = _decision_for_finding(finding)
        if decision is None:
            continue
        decision_to_ids.setdefault(decision, []).append(
            str(finding.get("finding_id", ""))
        )
    for ids in decision_to_ids.values():
        ids.sort()

    candidates = list(decision_to_ids.keys())
    if candidates:
        # Highest decision_priority wins; ties broken by fixed decision order.
        order = {name: index for index, name in enumerate(ALL_DECISIONS)}
        primary = max(
            candidates,
            key=lambda d: (priority.get(d, 0), -order.get(d, len(order))),
        )
        secondary = sorted(
            (d for d in candidates if d != primary),
            key=lambda d: (-priority.get(d, 0), order.get(d, len(order))),
        )
        return primary, secondary, decision_to_ids

    # No high/critical findings drove a decision: fall back to the tier.
    if overall_tier == "low":
        return DECISION_AUTO_APPROVE, [], decision_to_ids
    if overall_tier == "medium":
        return DECISION_MANUAL, [], decision_to_ids
    if overall_tier in ("high", "critical"):
        return DECISION_MANUAL, [], decision_to_ids
    # Unknown tier with no findings.
    return DECISION_MANUAL, [], decision_to_ids


def _approver_for_decision(
    decision: str,
    approval_policy: dict[str, Any] | None,
) -> str:
    """Resolve the approver for a decision from approval_policy.yaml, else default."""
    default = DECISION_TO_APPROVER.get(decision, "procurement_manager")
    if not isinstance(approval_policy, dict):
        return default
    rules = approval_policy.get("approval_rules")
    if not isinstance(rules, list):
        return default
    aliases = DECISION_ACTION_ALIASES.get(decision, {decision})
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        action = str(rule.get("action", "")).strip().lower()
        if action and action in aliases:
            approver = _clean_str(rule.get("approver"))
            if approver:
                return approver
    return default


# ----------------------------------------------------------------------------
# Overall risk
# ----------------------------------------------------------------------------
def _overall_risk(
    clause_analysis: dict[str, Any] | None,
    normalized_counterparty: dict[str, Any] | None,
    findings: list[dict[str, Any]],
) -> str:
    """Derive the overall risk tier (escalated by counterparty / findings)."""
    if not isinstance(clause_analysis, dict):
        return "unknown"
    tiers: list[str] = []
    base = _tier_from(clause_analysis.get("risk_tier"))
    if base:
        tiers.append(base)
    if isinstance(normalized_counterparty, dict):
        cp_tier = _tier_from(normalized_counterparty.get("risk_level"))
        if cp_tier:
            tiers.append(cp_tier)
    tiers.extend(f.get("risk_tier", "") for f in findings)
    return _max_tier(tiers) or (base or "low")


# ----------------------------------------------------------------------------
# Evidence (preserve references; never invent)
# ----------------------------------------------------------------------------
def _evidence_summary(
    findings: list[dict[str, Any]],
    evidence_index: dict[str, Any] | None,
) -> dict[str, Any]:
    clause_to_evidence: dict[str, list[str]] = {}
    items_by_id: dict[str, dict[str, Any]] = {}
    if isinstance(evidence_index, dict):
        for item in evidence_index.get("evidence_items") or []:
            if not isinstance(item, dict):
                continue
            evidence_id = _clean_str(item.get("evidence_id"))
            clause_id = _clean_str(item.get("clause_id"))
            if evidence_id:
                items_by_id[evidence_id] = item
                if clause_id:
                    clause_to_evidence.setdefault(clause_id, []).append(evidence_id)

    refs: list[str] = []
    evidence_ids: list[str] = []
    for finding in findings:
        for ref in finding.get("evidence_refs") or []:
            if ref not in refs:
                refs.append(ref)
            if ref in items_by_id and ref not in evidence_ids:
                evidence_ids.append(ref)
        clause_id = finding.get("clause_id")
        if clause_id:
            for evidence_id in clause_to_evidence.get(clause_id, []):
                if evidence_id not in evidence_ids:
                    evidence_ids.append(evidence_id)

    items: list[dict[str, Any]] = []
    for evidence_id in sorted(evidence_ids):
        item = items_by_id.get(evidence_id)
        if not item:
            continue
        items.append(
            {
                "evidence_id": evidence_id,
                "clause_id": _clean_str(item.get("clause_id")),
                "filename": _clean_str(item.get("filename")),
                "page": item.get("page"),
            }
        )

    return {
        "evidence_ids": sorted(evidence_ids),
        "evidence_refs": sorted(refs),
        "items": items,
    }


# ----------------------------------------------------------------------------
# Exception categories
# ----------------------------------------------------------------------------
def _exception_categories(
    findings: list[dict[str, Any]],
    decision_to_ids: dict[str, list[str]],
    approval_policy: dict[str, Any] | None,
    priority: dict[str, int],
) -> list[dict[str, Any]]:
    """Group prioritized findings into actionable exception categories."""
    by_category: dict[str, list[dict[str, Any]]] = {}
    for finding in findings:
        decision = _decision_for_finding(finding)
        if decision is None:
            continue
        category = (finding.get("category") or "manual_review").lower()
        by_category.setdefault(category, []).append(finding)

    categories: list[dict[str, Any]] = []
    for category, group in by_category.items():
        # Decision for the category = highest-priority decision among its members.
        decisions = {
            _decision_for_finding(f) for f in group if _decision_for_finding(f)
        }
        order = {name: index for index, name in enumerate(ALL_DECISIONS)}
        decision = max(
            decisions,
            key=lambda d: (priority.get(d, 0), -order.get(d, len(order))),
        )
        max_severity = _max_tier([f.get("risk_tier", "") for f in group]) or "medium"
        recommendations: list[str] = []
        for finding in group:
            rec = finding.get("recommendation")
            if rec and rec not in recommendations:
                recommendations.append(rec)
        categories.append(
            {
                "category": category,
                "decision": decision,
                "approver": _approver_for_decision(decision, approval_policy),
                "max_severity": max_severity,
                "finding_count": len(group),
                "finding_ids": sorted(str(f.get("finding_id", "")) for f in group),
                "recommendations": recommendations,
            }
        )

    order = {name: index for index, name in enumerate(ALL_DECISIONS)}
    categories.sort(
        key=lambda c: (
            -priority.get(c["decision"], 0),
            order.get(c["decision"], len(order)),
            c["category"],
        )
    )
    return categories


# ----------------------------------------------------------------------------
# Identifiers & next actions
# ----------------------------------------------------------------------------
def _contract_id(
    clause_analysis: dict[str, Any] | None,
    validation_result: dict[str, Any] | None,
    context_packet: dict[str, Any] | None,
) -> str:
    for source in (clause_analysis, validation_result, context_packet):
        if isinstance(source, dict):
            value = _clean_str(source.get("contract_id"))
            if value:
                return value
    return "unknown"


def _counterparty_name(
    context_packet: dict[str, Any] | None,
    normalized_counterparty: dict[str, Any] | None,
) -> str | None:
    if isinstance(normalized_counterparty, dict):
        for key in (
            "matched_vendor_name",
            "extracted_counterparty_name",
            "input_counterparty_name",
            "manifest_counterparty_name",
        ):
            value = _clean_str(normalized_counterparty.get(key))
            if value:
                return value
    if isinstance(context_packet, dict):
        value = _clean_str(context_packet.get("counterparty_name_from_manifest"))
        if value:
            return value
    return None


def _next_actions(
    final_decision: str,
    primary_approver: str,
    categories: list[dict[str, Any]],
) -> list[str]:
    if final_decision == DECISION_AUTO_APPROVE:
        return ["No exceptions found. Auto-approve and post the contract."]
    if final_decision == DECISION_REJECT_OR_BLOCK:
        return [
            "Block the contract; do not post to the CLM.",
            f"Escalate to {primary_approver} for final disposition.",
        ]
    actions = [f"Route the contract to {primary_approver} for {final_decision}."]
    for category in categories:
        actions.append(
            f"Resolve {category['category']} exceptions "
            f"({', '.join(category['finding_ids'])}) via {category['approver']}."
        )
    actions.append("Resolve all listed exceptions before final approval.")
    return actions


# ----------------------------------------------------------------------------
# Output shaping
# ----------------------------------------------------------------------------
def _prioritized_finding_view(finding: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "priority_rank": rank,
        "finding_id": finding.get("finding_id"),
        "source": finding.get("source"),
        "category": finding.get("category"),
        "severity": finding.get("severity"),
        "risk_tier": finding.get("risk_tier"),
        "score": finding.get("score"),
        "finding_type": finding.get("finding_type"),
        "policy_rule": finding.get("policy_rule"),
        "field": finding.get("field"),
        "clause_id": finding.get("clause_id"),
        "evidence_refs": list(finding.get("evidence_refs") or []),
        "recommendation": finding.get("recommendation"),
        "requires_human_review": bool(finding.get("requires_human_review", False)),
        "blocking": bool(finding.get("blocking", False)),
        "mapped_decision": _decision_for_finding(finding),
    }


def _key_finding_view(finding: dict[str, Any]) -> dict[str, Any]:
    evidence_refs = finding.get("evidence_refs") or []
    return {
        "finding_id": finding.get("finding_id"),
        "source": finding.get("source"),
        "category": finding.get("category"),
        "risk_tier": finding.get("risk_tier"),
        "score": finding.get("score"),
        "policy_rule": finding.get("policy_rule"),
        "recommendation": finding.get("recommendation"),
        "evidence_ref": evidence_refs[0] if evidence_refs else None,
    }


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
    primary_approver: str,
    secondary_reviews: list[dict[str, str]],
    prioritized: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    evidence_summary: dict[str, Any],
    next_actions: list[str],
    dedupe_summary: dict[str, Any],
) -> None:
    lines: list[str] = [
        f"# Exception Report — {contract_id}",
        "",
        "## Summary",
        f"- Contract ID: {contract_id}",
        f"- Contract type: {contract_type or 'unknown'}",
        f"- Counterparty: {counterparty or 'unknown'}",
        f"- Overall risk: {overall_risk}",
        f"- Final decision: {final_decision}",
        f"- Primary approval path: {primary_approver} ({final_decision})",
    ]
    if secondary_reviews:
        secondary_text = ", ".join(
            f"{review['decision']} -> {review['approver']}"
            for review in secondary_reviews
        )
        lines.append(f"- Secondary reviews: {secondary_text}")
    else:
        lines.append("- Secondary reviews: none")

    lines.extend(["", "## Exception Categories"])
    if categories:
        for category in categories:
            lines.append(
                f"- {category['category']} "
                f"[{category['max_severity']}] -> {category['decision']} "
                f"({category['approver']}); findings: "
                f"{', '.join(category['finding_ids'])}"
            )
    else:
        lines.append("- No exception categories were raised.")

    lines.extend(["", "## Prioritized Findings"])
    if prioritized:
        for finding in prioritized:
            evidence_refs = finding.get("evidence_refs") or []
            evidence = evidence_refs[0] if evidence_refs else "n/a"
            lines.append(
                f"- {finding.get('finding_id')} "
                f"[{finding.get('source')}/{finding.get('category')}/"
                f"{finding.get('risk_tier')}, score={finding.get('score')}]: "
                f"{finding.get('recommendation') or 'Review required.'} "
                f"(evidence: {evidence})"
            )
    else:
        lines.append("- No risk findings were raised.")

    lines.extend(["", "## Evidence"])
    evidence_ids = evidence_summary.get("evidence_ids") or []
    if evidence_ids:
        lines.append(f"- Evidence IDs: {', '.join(evidence_ids)}")
    else:
        lines.append("- No evidence references available.")
    evidence_refs = evidence_summary.get("evidence_refs") or []
    if evidence_refs:
        lines.append(f"- Evidence references: {', '.join(evidence_refs)}")

    lines.extend(["", "## Deduplication"])
    lines.append(
        f"- Findings merged: {dedupe_summary.get('input_count', 0)}; "
        f"duplicates removed: {dedupe_summary.get('duplicates_removed', 0)}; "
        f"retained: {dedupe_summary.get('output_count', 0)}"
    )

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
    prioritized_count: int,
    dedupe_summary: dict[str, Any],
    category_count: int,
) -> None:
    audit_path = run_directory / AUDIT_LOG_FILENAME
    existing = audit_path.read_text(encoding="utf-8") if audit_path.exists() else ""
    approvers_text = ", ".join(approvers) if approvers else "none"
    lines = [
        "",
        "## Step 6: Exception Triage / Final Decision",
        "- Agent: H (exception triage & lead orchestrator)",
        f"- Overall risk: {overall_risk}",
        f"- Final decision: {final_decision}",
        f"- Approvers: {approvers_text}",
        f"- Prioritized findings: {prioritized_count}",
        f"- Exception categories: {category_count}",
        f"- Findings merged: {dedupe_summary.get('input_count', 0)}, "
        f"duplicates removed: {dedupe_summary.get('duplicates_removed', 0)}",
        "- Generated artifacts: exceptions.md, approval_packet.json, "
        "posting_payload.json",
        "- Status: triage_completed",
        "",
    ]
    audit_path.write_text(existing + "\n".join(lines), encoding="utf-8")


def _update_metrics(
    run_directory: Path,
    *,
    overall_risk: str,
    final_decision: str,
    primary_approver: str,
    approvers: list[str],
    prioritized_count: int,
    dedupe_summary: dict[str, Any],
    category_count: int,
    secondary_count: int,
) -> None:
    """Add Agent H metrics without deleting any earlier agent's metrics."""
    metrics_path = run_directory / METRICS_FILENAME
    metrics: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metrics = loaded
        except (OSError, ValueError):
            metrics = {}

    agents = metrics.get("agents")
    if not isinstance(agents, dict):
        agents = {}
    agents["agent_h"] = {
        "status": "completed",
        "final_decision": final_decision,
        "overall_risk": overall_risk,
        "primary_approver": primary_approver,
        "approvers": approvers,
        "prioritized_findings": prioritized_count,
        "exception_categories": category_count,
        "secondary_reviews": secondary_count,
        "findings_merged": dedupe_summary.get("input_count", 0),
        "duplicates_removed": dedupe_summary.get("duplicates_removed", 0),
    }
    metrics["agents"] = agents

    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# ----------------------------------------------------------------------------
# Public stable interface
# ----------------------------------------------------------------------------
def run_triage_agent(run_directory: str | Path) -> dict[str, Any]:
    """Run Agent H (Exception Triage & Lead Orchestrator) for one run directory.

    This is the stable interface the orchestrator always calls. It reads the
    upstream artifacts, consolidates/deduplicates/prioritizes findings, applies
    deterministic decision and approval-routing logic, and writes the final
    artifacts (``exceptions.md``, ``approval_packet.json``,
    ``posting_payload.json``) plus updates ``audit_log.md`` and ``metrics.json``.

    Args:
        run_directory: The run directory created by Agent A.

    Returns:
        A dict containing at least ``overall_risk`` and ``final_decision`` (the
        keys the orchestrator reads), plus the full triage result.

    Raises:
        AgentHError: When a required upstream artifact exists but is invalid
            (corrupt JSON or wrong shape). Missing artifacts degrade gracefully.
    """
    run_dir = Path(run_directory)

    context_packet = _read_required_json(run_dir / CONTEXT_PACKET_FILENAME)
    normalized_counterparty = _read_required_json(
        run_dir / NORMALIZED_COUNTERPARTY_FILENAME
    )
    validation_result = _read_required_json(run_dir / VALIDATION_RESULT_FILENAME)
    clause_analysis = _read_required_json(run_dir / CLAUSE_ANALYSIS_FILENAME)
    evidence_index = _read_required_json(run_dir / EVIDENCE_INDEX_FILENAME)
    approval_policy = _read_yaml(run_dir / APPROVAL_POLICY_SNAPSHOT)

    priority = _decision_priority(approval_policy)

    # 1. Merge upstream findings (Agent E, then D, then C).
    merged = (
        _normalize_agent_e(clause_analysis)
        + _normalize_agent_d(validation_result)
        + _normalize_agent_c(normalized_counterparty)
    )

    # 2. Deduplicate, then 3. prioritize.
    deduped, dedupe_summary = _dedupe_findings(merged)
    prioritized = _prioritize(deduped)

    # 4. Overall risk + 5. decision + secondary reviews.
    overall_risk = _overall_risk(clause_analysis, normalized_counterparty, prioritized)

    if not isinstance(clause_analysis, dict):
        # No risk scoring available: degrade gracefully to manual review.
        final_decision = DECISION_MANUAL
        secondary_decisions: list[str] = []
        decision_to_ids: dict[str, list[str]] = {}
    else:
        final_decision, secondary_decisions, decision_to_ids = _select_decision(
            prioritized, overall_risk, priority
        )

    # 6. Approval routing.
    primary_approver = _approver_for_decision(final_decision, approval_policy)
    secondary_reviews = [
        {"decision": decision, "approver": _approver_for_decision(decision, approval_policy)}
        for decision in secondary_decisions
    ]
    approvers: list[str] = []
    if primary_approver and primary_approver != "none":
        approvers.append(primary_approver)
    for review in secondary_reviews:
        approver = review["approver"]
        if approver and approver != "none" and approver not in approvers:
            approvers.append(approver)

    # Derived/contextual values.
    contract_id = _contract_id(clause_analysis, validation_result, context_packet)
    contract_type = (
        _clean_str(context_packet.get("contract_type"))
        if isinstance(context_packet, dict)
        else None
    )
    counterparty = _counterparty_name(context_packet, normalized_counterparty)

    categories = _exception_categories(
        prioritized, decision_to_ids, approval_policy, priority
    )
    evidence_summary = _evidence_summary(prioritized, evidence_index)
    next_actions = _next_actions(final_decision, primary_approver, categories)

    prioritized_view = [
        _prioritized_finding_view(finding, index)
        for index, finding in enumerate(prioritized, start=1)
    ]
    key_findings = [_key_finding_view(finding) for finding in prioritized]

    approval_required = final_decision != DECISION_AUTO_APPROVE
    primary_approval_path = {"decision": final_decision, "approver": primary_approver}

    # approval_packet.json
    approval_packet = {
        "contract_id": contract_id,
        "contract_type": contract_type,
        "counterparty": counterparty,
        "overall_risk": overall_risk,
        "final_decision": final_decision,
        "primary_approval_path": primary_approval_path,
        "secondary_reviews": secondary_reviews,
        "approvers": approvers,
        "approval_required": approval_required,
        "prioritized_findings": prioritized_view,
        "key_findings": key_findings,
        "exception_categories": categories,
        "evidence_summary": evidence_summary,
        "dedupe_summary": dedupe_summary,
        "next_actions": next_actions,
    }
    _write_json(run_dir / APPROVAL_PACKET_FILENAME, approval_packet)

    # posting_payload.json
    if final_decision == DECISION_AUTO_APPROVE:
        posting_status = "ready_for_posting"
    elif final_decision == DECISION_REJECT_OR_BLOCK:
        posting_status = "blocked"
    else:
        posting_status = "pending_review"
    posting_payload = {
        "target_system": TARGET_SYSTEM,
        "contract_id": contract_id,
        "counterparty": counterparty,
        "contract_type": contract_type,
        "risk_level": overall_risk,
        "final_decision": final_decision,
        "status": posting_status,
        "approval_required": approval_required,
        "primary_approver": primary_approver,
        "approvers": approvers,
        "exception_categories": [category["category"] for category in categories],
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
        primary_approver=primary_approver,
        secondary_reviews=secondary_reviews,
        prioritized=prioritized,
        categories=categories,
        evidence_summary=evidence_summary,
        next_actions=next_actions,
        dedupe_summary=dedupe_summary,
    )

    _append_audit_log(
        run_dir,
        overall_risk=overall_risk,
        final_decision=final_decision,
        approvers=approvers,
        prioritized_count=len(prioritized_view),
        dedupe_summary=dedupe_summary,
        category_count=len(categories),
    )

    _update_metrics(
        run_dir,
        overall_risk=overall_risk,
        final_decision=final_decision,
        primary_approver=primary_approver,
        approvers=approvers,
        prioritized_count=len(prioritized_view),
        dedupe_summary=dedupe_summary,
        category_count=len(categories),
        secondary_count=len(secondary_reviews),
    )

    return {
        "overall_risk": overall_risk,
        "final_decision": final_decision,
        "primary_approval_path": primary_approval_path,
        "secondary_reviews": secondary_reviews,
        "approvers": approvers,
        "approval_required": approval_required,
        "prioritized_findings": prioritized_view,
        "key_findings": key_findings,
        "exception_categories": categories,
        "evidence_summary": evidence_summary,
        "dedupe_summary": dedupe_summary,
        "next_actions": next_actions,
        "contract_id": contract_id,
        "counterparty": counterparty,
    }
