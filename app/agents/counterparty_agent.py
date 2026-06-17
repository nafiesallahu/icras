from __future__ import annotations

import json
import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz import fuzz

from app.schemas.finding import (
    FindingRiskLevel,
    Severity,
    UnifiedFinding,
)

# NormalizedCounterparty is the final structured Agent C result.
from app.schemas.normalized_counterparty import NormalizedCounterparty

# FuzzyMatchResult contains the vendor match and score produced
# by the RapidFuzz service.
from app.services.fuzzy_matcher import (
    DEFAULT_MATCH_THRESHOLD,
    FuzzyMatchResult,
    find_best_vendor_match,
    load_vendor_master_for_run,
    normalize_company_name,
)

# The names are considered consistent when their similarity score
# is equal to or above this value.
#
# A separate threshold is used for manifest-versus-contract comparison
# because this is a different decision from vendor-master matching.
DEFAULT_COUNTERPARTY_MISMATCH_THRESHOLD = 85

# Machine-readable flags used by downstream Agents D, E, and H.
NEW_COUNTERPARTY_FLAG = "new_counterparty"
COUNTERPARTY_NOT_FOUND_FLAG = "counterparty_not_found"
HIGH_RISK_COUNTERPARTY_FLAG = "high_risk_counterparty"
MANUAL_REVIEW_REQUIRED_FLAG = "manual_review_required"
COUNTERPARTY_MISMATCH_FLAG = "counterparty_mismatch"


# Vendor risk levels supported by NormalizedCounterparty.
SUPPORTED_VENDOR_RISK_LEVELS = frozenset(
    {
        "low",
        "medium",
        "high",
    }
)


# These vendor-master statuses represent an additional high-risk
# condition even when risk_level is not explicitly "high".
#
# This supports the requirement:
# "high risk_level or another high-risk vendor condition."
HIGH_RISK_VENDOR_STATUSES = frozenset(
    {
        "high_risk",
        "blocked",
        "suspended",
        "sanctioned",
    }
)

# Every Agent C finding must identify its source and category.
AGENT_C_SOURCE = "agent_c"
COUNTERPARTY_CATEGORY = "counterparty"


# Deterministic scores assigned to Agent C findings.
#
# These values are fixed rather than calculated at runtime,
# which keeps repeated executions deterministic.
UNKNOWN_COUNTERPARTY_SCORE = 80
HIGH_RISK_COUNTERPARTY_SCORE = 90

HIGH_COUNTERPARTY_MISMATCH_SCORE = 85
MEDIUM_COUNTERPARTY_MISMATCH_SCORE = 60


# These are the exact JSON locations Agent C must read.
CONTEXT_PACKET_FILENAME = "context_packet.json"
EXTRACTED_CONTRACT_FILENAME = "extracted_contract.json"
EVIDENCE_INDEX_FILENAME = "evidence_index.json"
NORMALIZED_COUNTERPARTY_FILENAME = "normalized_counterparty.json"
AUDIT_LOG_FILENAME = "audit_log.md"
METRICS_FILENAME = "metrics.json"

MANIFEST_COUNTERPARTY_FIELD = "counterparty_name_from_manifest"
EXTRACTED_PARTIES_FIELD = "parties"
EXTRACTED_COUNTERPARTY_FIELD = "counterparty_name"


class CounterpartyInputError(Exception):
    """
    Raised when Agent C cannot read the required counterparty inputs.

    The orchestrator can catch this exception and:
    - fail the Agent C stage;
    - write the error to audit_log.md;
    - update metrics.json.
    """


class CounterpartyClassificationError(Exception):
    """
    Raised when matched vendor metadata cannot be classified safely.

    Examples:
    - unsupported vendor status;
    - unsupported risk level;
    - inconsistent fuzzy-match result.

    The orchestrator can catch this exception and fail Agent C cleanly.
    """


class CounterpartyComparisonResult(BaseModel):
    """
    Structured result of comparing the manifest counterparty
    with the counterparty extracted by Agent B.
    """

    model_config = ConfigDict(
        # Reject undeclared output fields.
        extra="forbid",
        # Prevent a completed comparison from being reassigned.
        frozen=True,
        # Require correct primitive types.
        strict=True,
        # Remove unnecessary surrounding whitespace.
        str_strip_whitespace=True,
    )

    # Counterparty declared in manifest.yaml and later stored
    # in context_packet.json by Agent A.
    manifest_counterparty_name: str = Field(
        ...,
        min_length=1,
    )

    # Counterparty extracted from the contract by Agent B.
    extracted_counterparty_name: str = Field(
        ...,
        min_length=1,
    )

    # RapidFuzz similarity score between the two normalized names.
    similarity_score: int = Field(
        ...,
        ge=0,
        le=100,
    )

    # Threshold used to decide whether the difference is material.
    mismatch_threshold: int = Field(
        ...,
        ge=0,
        le=100,
    )

    # True means the two names materially differ.
    is_mismatch: bool

    # A mismatch creates a standardized finding.
    # Matching or near-matching names return None.
    finding: UnifiedFinding | None = None


def _load_json_object(
    json_path: Path,
    *,
    file_description: str,
) -> dict[str, Any]:
    """
    Load a JSON file and ensure its top-level value is an object.

    A clear Agent C exception is raised for missing, unreadable,
    malformed, or incorrectly structured JSON.
    """

    # Required input files must exist before Agent C can continue.
    if not json_path.is_file():
        raise CounterpartyInputError(
            f"{file_description} was not found at: {json_path}"
        )

    try:
        # Read the JSON file using deterministic UTF-8 decoding.
        raw_content = json_path.read_text(encoding="utf-8")

        # Convert the JSON text into Python values.
        data = json.loads(raw_content)

    except UnicodeDecodeError as exc:
        raise CounterpartyInputError(
            f"{file_description} must use valid UTF-8 encoding."
        ) from exc

    except json.JSONDecodeError as exc:
        raise CounterpartyInputError(
            f"{file_description} contains invalid JSON: {exc}"
        ) from exc

    except OSError as exc:
        raise CounterpartyInputError(
            f"{file_description} could not be read: {exc}"
        ) from exc

    # Both required artifacts must contain JSON objects,
    # not lists, strings, numbers, or null.
    if not isinstance(data, dict):
        raise CounterpartyInputError(f"{file_description} must contain a JSON object.")

    return data


def _read_optional_evidence_index(
    run_dir: Path,
) -> dict[str, Any] | None:
    """
    Read evidence_index.json when available.

    Missing evidence_index.json is allowed.
    An existing but invalid file causes a clear failure.
    """

    evidence_path = run_dir / EVIDENCE_INDEX_FILENAME

    if not evidence_path.exists():
        return None

    evidence_index = _load_json_object(
        evidence_path,
        file_description=EVIDENCE_INDEX_FILENAME,
    )

    evidence_items = evidence_index.get(
        "evidence_items",
        [],
    )

    if not isinstance(evidence_items, list):
        raise CounterpartyInputError(
            "evidence_index.json field 'evidence_items' " "must contain a JSON array."
        )

    return evidence_index


def _require_non_empty_name(
    value: Any,
    *,
    field_path: str,
    file_description: str,
) -> str:
    """
    Validate that a counterparty field contains a usable string.

    The original string is returned so the final finding preserves
    the exact manifest and extracted values for auditability.
    """

    if not isinstance(value, str):
        raise CounterpartyInputError(
            f"{file_description} field '{field_path}' " "must contain a string."
        )

    if not value.strip():
        raise CounterpartyInputError(
            f"{file_description} field '{field_path}' " "must not be empty."
        )

    return value


def read_counterparty_names(
    run_dir: str | Path,
) -> tuple[str, str]:
    """
    Read the two counterparty names required by Agent C.

    Values are loaded from:

    context_packet.json:
        counterparty_name_from_manifest

    extracted_contract.json:
        parties.counterparty_name
    """

    resolved_run_dir = Path(run_dir)

    context_packet_path = resolved_run_dir / CONTEXT_PACKET_FILENAME

    extracted_contract_path = resolved_run_dir / EXTRACTED_CONTRACT_FILENAME

    # Load the output created by Agent A.
    context_packet = _load_json_object(
        context_packet_path,
        file_description="context_packet.json",
    )

    # Load the output created by Agent B.
    extracted_contract = _load_json_object(
        extracted_contract_path,
        file_description="extracted_contract.json",
    )

    # Read counterparty_name_from_manifest directly from
    # the top level of context_packet.json.
    manifest_counterparty_name = _require_non_empty_name(
        context_packet.get(MANIFEST_COUNTERPARTY_FIELD),
        field_path=MANIFEST_COUNTERPARTY_FIELD,
        file_description="context_packet.json",
    )

    # The extracted counterparty is nested under "parties".
    parties = extracted_contract.get(EXTRACTED_PARTIES_FIELD)

    if not isinstance(parties, dict):
        raise CounterpartyInputError(
            "extracted_contract.json field 'parties' " "must contain a JSON object."
        )

    extracted_counterparty_name = _require_non_empty_name(
        parties.get(EXTRACTED_COUNTERPARTY_FIELD),
        field_path="parties.counterparty_name",
        file_description="extracted_contract.json",
    )

    return (
        manifest_counterparty_name,
        extracted_counterparty_name,
    )


def compare_counterparty_names(
    manifest_counterparty_name: str,
    extracted_counterparty_name: str,
    *,
    mismatch_threshold: int = (DEFAULT_COUNTERPARTY_MISMATCH_THRESHOLD),
    mismatch_severity: Severity = Severity.HIGH,
) -> CounterpartyComparisonResult:
    """
    Compare the manifest and extracted counterparty names.

    The function:

    1. validates the configurable threshold;
    2. normalizes both company names;
    3. calculates a RapidFuzz similarity score;
    4. treats scores below the threshold as material mismatches;
    5. creates a UnifiedFinding when a mismatch exists.

    Minor formatting differences are handled by the deterministic
    normalization implemented in DZ-01.3.
    """

    # bool is a subclass of int in Python, so it must be rejected
    # explicitly to prevent True or False being used as a threshold.
    if isinstance(mismatch_threshold, bool) or not isinstance(mismatch_threshold, int):
        raise TypeError(
            "Counterparty mismatch threshold must be an integer " "from 0 to 100."
        )

    if mismatch_threshold < 0 or mismatch_threshold > 100:
        raise ValueError(
            "Counterparty mismatch threshold must be " "between 0 and 100."
        )

    # The task permits medium or high mismatch risk depending
    # on configuration.
    if not isinstance(mismatch_severity, Severity):
        raise TypeError("Counterparty mismatch severity must be a Severity value.")

    if mismatch_severity not in {
        Severity.MEDIUM,
        Severity.HIGH,
    }:
        raise ValueError(
            "Counterparty mismatch severity must be "
            "Severity.MEDIUM or Severity.HIGH."
        )

    # Apply exactly the same normalization rules to both names.
    normalized_manifest_name = normalize_company_name(manifest_counterparty_name)

    normalized_extracted_name = normalize_company_name(extracted_counterparty_name)

    # Compare the two normalized names using RapidFuzz.
    raw_score = fuzz.ratio(
        normalized_manifest_name,
        normalized_extracted_name,
    )

    # Store a deterministic integer score from 0 to 100.
    similarity_score = int(round(raw_score))

    # A score equal to the threshold is accepted.
    # Only a score below the threshold is a mismatch.
    is_mismatch = similarity_score < mismatch_threshold

    finding: UnifiedFinding | None = None

    if is_mismatch:
        # A configured medium mismatch uses a medium risk level and score.
        if mismatch_severity == Severity.MEDIUM:
            mismatch_risk_level = FindingRiskLevel.MEDIUM
            mismatch_score = MEDIUM_COUNTERPARTY_MISMATCH_SCORE
        else:
            mismatch_risk_level = FindingRiskLevel.HIGH
            mismatch_score = HIGH_COUNTERPARTY_MISMATCH_SCORE

        mismatch_message = (
            "counterparty_mismatch: The counterparty declared "
            "in the manifest materially differs from the "
            "counterparty extracted from the contract. "
            f"Similarity score {similarity_score} is below "
            f"the threshold {mismatch_threshold}."
        )

        finding = _create_agent_c_finding(
            finding_id="CP-MISMATCH-001",
            finding_type="counterparty_mismatch",
            severity=mismatch_severity,
            risk_level=mismatch_risk_level,
            score=mismatch_score,
            policy_rule="counterparty.names_must_match",
            expected=manifest_counterparty_name,
            actual=extracted_counterparty_name,
            recommendation=(
                "Manually review the contract and manifest "
                "counterparty names before approval."
            ),
            message=mismatch_message,
            evidence_ref=(
                "context_packet.json:"
                "counterparty_name_from_manifest; "
                "extracted_contract.json:"
                "parties.counterparty_name"
            ),
        )

    return CounterpartyComparisonResult(
        manifest_counterparty_name=manifest_counterparty_name,
        extracted_counterparty_name=extracted_counterparty_name,
        similarity_score=similarity_score,
        mismatch_threshold=mismatch_threshold,
        is_mismatch=is_mismatch,
        finding=finding,
    )


def _create_agent_c_finding(
    *,
    finding_id: str,
    finding_type: str,
    severity: Severity,
    risk_level: FindingRiskLevel,
    score: int,
    policy_rule: str,
    expected: str,
    actual: str,
    recommendation: str,
    message: str,
    evidence_ref: str,
) -> UnifiedFinding:
    """
    Create one standardized Agent C finding.

    Using one builder guarantees that every Agent C finding has
    the same required source, category, evidence, and review fields.
    """

    return UnifiedFinding(
        # Required unified fields.
        finding_id=finding_id,
        source_agent=AGENT_C_SOURCE,
        category=COUNTERPARTY_CATEGORY,
        severity=severity,
        risk_level=risk_level,
        score=score,
        finding_type=finding_type,
        clause_id=None,
        policy_rule=policy_rule,
        expected=expected,
        actual=actual,
        recommendation=recommendation,
        evidence_ids=[],
        requires_human_review=True,
        # Legacy compatibility fields.
        field="counterparty",
        message=message,
        evidence_ref=evidence_ref,
        page_number=None,
        bbox=None,
        expected_value=expected,
        actual_value=actual,
    )


def detect_counterparty_mismatch_from_run(
    run_dir: str | Path,
    *,
    mismatch_threshold: int = (DEFAULT_COUNTERPARTY_MISMATCH_THRESHOLD),
    mismatch_severity: Severity = Severity.HIGH,
) -> CounterpartyComparisonResult:
    """
    Read Agent A and Agent B outputs from the run directory,
    then perform the counterparty mismatch check.
    """

    (
        manifest_counterparty_name,
        extracted_counterparty_name,
    ) = read_counterparty_names(run_dir)

    return compare_counterparty_names(
        manifest_counterparty_name,
        extracted_counterparty_name,
        mismatch_threshold=mismatch_threshold,
        mismatch_severity=mismatch_severity,
    )


def _append_unique_flag(
    flags: list[str],
    flag: str,
) -> None:
    """
    Add a flag only when it is not already present.

    This preserves deterministic ordering and prevents duplicates.
    """

    if flag not in flags:
        flags.append(flag)


def _has_finding_type(
    findings: list[UnifiedFinding],
    finding_type: str,
) -> bool:
    """
    Check whether a finding type already exists.

    This prevents duplicate findings if classification is called
    more than once with the same existing findings.
    """

    return any(finding.finding_type == finding_type for finding in findings)


def classify_counterparty_status(
    match_result: FuzzyMatchResult,
    *,
    manifest_counterparty_name: str | None = None,
    extracted_counterparty_name: str | None = None,
    findings: Sequence[UnifiedFinding] = (),
) -> NormalizedCounterparty:
    """
    Convert a fuzzy vendor match into Agent C's normalized output.

    This function classifies the counterparty as:

    - approved
    - new
    - unknown
    - high_risk

    It also creates the required flags and findings.
    """

    if not isinstance(match_result, FuzzyMatchResult):
        raise TypeError("match_result must be a FuzzyMatchResult instance.")

    # Copy existing findings, such as a mismatch finding,
    # so the original list is not modified.
    normalized_findings = list(findings)

    for finding_index, finding in enumerate(normalized_findings):
        if not isinstance(finding, UnifiedFinding):
            raise TypeError(
                "Every finding must be a UnifiedFinding instance. "
                f"Invalid finding at position {finding_index}."
            )

    # Start with an empty flag list.
    #
    # Flags are added depending on the classification.
    output_flags: list[str] = []

    # If DZ-01.5 already created a counterparty mismatch finding,
    # also add the required mismatch and manual-review flags.
    if _has_finding_type(
        normalized_findings,
        "counterparty_mismatch",
    ):
        _append_unique_flag(
            output_flags,
            COUNTERPARTY_MISMATCH_FLAG,
        )

        _append_unique_flag(
            output_flags,
            MANUAL_REVIEW_REQUIRED_FLAG,
        )

    # ==========================================================
    # UNKNOWN COUNTERPARTY
    # ==========================================================
    #
    # The fuzzy matcher returned unknown because no vendor
    # reached the configured matching threshold.
    if match_result.match_status == "unknown":
        if match_result.matched_vendor is not None:
            raise CounterpartyClassificationError(
                "An unknown fuzzy-match result must not contain " "a matched_vendor."
            )

        # Create the unknown_counterparty finding only once.
        if not _has_finding_type(
            normalized_findings,
            "unknown_counterparty",
        ):
            normalized_findings.append(
                _create_agent_c_finding(
                    finding_id="CP-UNKNOWN-001",
                    finding_type="unknown_counterparty",
                    severity=Severity.HIGH,
                    risk_level=FindingRiskLevel.HIGH,
                    score=UNKNOWN_COUNTERPARTY_SCORE,
                    policy_rule="vendor_master.match_required",
                    expected=(
                        "Counterparty should match an approved "
                        "vendor in vendor_master.csv"
                    ),
                    actual=(
                        "No vendor matched above the configured "
                        f"threshold. Best score: "
                        f"{match_result.match_score}"
                    ),
                    recommendation=(
                        "Manually verify the counterparty and "
                        "vendor master record before approval."
                    ),
                    message=(
                        "unknown_counterparty: No acceptable "
                        "vendor master match was found."
                    ),
                    evidence_ref="vendor_master.csv",
                )
            )

        # Required flags for an unknown counterparty.
        _append_unique_flag(
            output_flags,
            COUNTERPARTY_NOT_FOUND_FLAG,
        )

        _append_unique_flag(
            output_flags,
            MANUAL_REVIEW_REQUIRED_FLAG,
        )

        # Unknown counterparties must not expose the closest
        # low-scoring candidate as an accepted vendor.
        return NormalizedCounterparty(
            input_counterparty_name=(match_result.input_counterparty_name),
            manifest_counterparty_name=(manifest_counterparty_name),
            extracted_counterparty_name=(extracted_counterparty_name),
            matched_vendor_id=None,
            matched_vendor_name=None,
            match_score=match_result.match_score,
            status="unknown",
            risk_level="high",
            flags=output_flags,
            findings=normalized_findings,
        )

    # ==========================================================
    # MATCHED COUNTERPARTY
    # ==========================================================

    matched_vendor = match_result.matched_vendor

    if matched_vendor is None:
        raise CounterpartyClassificationError(
            "A matched fuzzy-match result must contain " "matched_vendor."
        )

    vendor_status = matched_vendor.status.strip().casefold()

    vendor_risk_level = matched_vendor.risk_level.strip().casefold()

    if vendor_risk_level not in (SUPPORTED_VENDOR_RISK_LEVELS):
        raise CounterpartyClassificationError(
            "Unsupported vendor risk_level "
            f"'{matched_vendor.risk_level}' for vendor "
            f"'{matched_vendor.vendor_id}'."
        )

    output_status: str
    output_risk_level: str

    # ==========================================================
    # HIGH-RISK COUNTERPARTY
    # ==========================================================
    #
    # High risk has priority over approved or new.
    #
    # Examples:
    #
    # status = approved, risk_level = high
    # → high_risk
    #
    # status = blocked, risk_level = medium
    # → high_risk
    if vendor_risk_level == "high" or vendor_status in HIGH_RISK_VENDOR_STATUSES:
        output_status = "high_risk"
        output_risk_level = "high"

        # Required high-risk flag.
        _append_unique_flag(
            output_flags,
            HIGH_RISK_COUNTERPARTY_FLAG,
        )

        # A high-risk vendor cannot be automatically approved.
        _append_unique_flag(
            output_flags,
            MANUAL_REVIEW_REQUIRED_FLAG,
        )

        # Create the high_risk_counterparty finding only once.
        if not _has_finding_type(
            normalized_findings,
            "high_risk_counterparty",
        ):
            normalized_findings.append(
                _create_agent_c_finding(
                    finding_id="CP-HIGH-RISK-001",
                    finding_type=("high_risk_counterparty"),
                    severity=Severity.HIGH,
                    risk_level=FindingRiskLevel.HIGH,
                    score=HIGH_RISK_COUNTERPARTY_SCORE,
                    policy_rule=("vendor_master.high_risk_review"),
                    expected=(
                        "Counterparty should not be " "classified as a high-risk vendor"
                    ),
                    actual=(
                        f"Matched vendor "
                        f"'{matched_vendor.vendor_name}' has "
                        f"status '{matched_vendor.status}' and "
                        f"risk level "
                        f"'{matched_vendor.risk_level}'"
                    ),
                    recommendation=(
                        "Require manual procurement and "
                        "compliance review before approval."
                    ),
                    message=(
                        "high_risk_counterparty: The matched "
                        "vendor is classified as high risk."
                    ),
                    evidence_ref="vendor_master.csv",
                )
            )

    # ==========================================================
    # NEW COUNTERPARTY
    # ==========================================================
    elif vendor_status == "new":
        output_status = "new"
        output_risk_level = vendor_risk_level

        _append_unique_flag(
            output_flags,
            NEW_COUNTERPARTY_FLAG,
        )

        # A new vendor is not yet safely approved.
        _append_unique_flag(
            output_flags,
            MANUAL_REVIEW_REQUIRED_FLAG,
        )

    # ==========================================================
    # APPROVED COUNTERPARTY
    # ==========================================================
    elif vendor_status == "approved":
        output_status = "approved"
        output_risk_level = vendor_risk_level

        # No vendor-status flag is needed for an approved vendor.
        #
        # A mismatch flag may still already exist in output_flags.

    else:
        raise CounterpartyClassificationError(
            "Unsupported vendor status "
            f"'{matched_vendor.status}' for vendor "
            f"'{matched_vendor.vendor_id}'."
        )

    return NormalizedCounterparty(
        input_counterparty_name=(match_result.input_counterparty_name),
        manifest_counterparty_name=(manifest_counterparty_name),
        extracted_counterparty_name=(extracted_counterparty_name),
        matched_vendor_id=matched_vendor.vendor_id,
        matched_vendor_name=matched_vendor.vendor_name,
        match_score=match_result.match_score,
        status=output_status,
        risk_level=output_risk_level,
        flags=output_flags,
        findings=normalized_findings,
    )


def write_normalized_counterparty(
    run_dir: str | Path,
    result: NormalizedCounterparty,
) -> Path:
    """
    Serialize Agent C's final result to normalized_counterparty.json.

    The returned path allows the orchestrator and tests to verify
    exactly where the artifact was written.
    """

    if not isinstance(result, NormalizedCounterparty):
        raise TypeError("result must be a NormalizedCounterparty instance.")

    output_path = Path(run_dir) / NORMALIZED_COUNTERPARTY_FILENAME

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            result.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return output_path


def _write_json_object(
    output_path: Path,
    payload: dict[str, Any],
) -> None:
    """
    Write a deterministic JSON object.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _append_audit_log(
    run_dir: Path,
    lines: Sequence[str],
) -> Path:
    """
    Append Agent C details without removing earlier audit entries.
    """

    audit_path = run_dir / AUDIT_LOG_FILENAME

    audit_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    separator = ""

    if audit_path.exists() and audit_path.stat().st_size > 0:
        separator = "\n"

    with audit_path.open(
        "a",
        encoding="utf-8",
    ) as audit_file:
        audit_file.write(separator + "\n".join(lines) + "\n")

    return audit_path


def _update_agent_c_metrics(
    run_dir: Path,
    agent_metrics: dict[str, Any],
) -> Path:
    """
    Update only Agent C's metrics while preserving other agents.
    """

    metrics_path = run_dir / METRICS_FILENAME

    if metrics_path.exists():
        metrics = _load_json_object(
            metrics_path,
            file_description=METRICS_FILENAME,
        )
    else:
        metrics = {}

    existing_agents = metrics.get("agents", {})

    if not isinstance(existing_agents, dict):
        raise CounterpartyInputError(
            "metrics.json field 'agents' " "must contain a JSON object."
        )

    agents = dict(existing_agents)
    agents["agent_c"] = agent_metrics
    metrics["agents"] = agents

    if agent_metrics.get("status") == "failed":
        metrics["status"] = "failed_at_agent_c"

    _write_json_object(
        metrics_path,
        metrics,
    )

    return metrics_path


def _record_agent_c_failure(
    run_dir: Path,
    error: Exception,
) -> None:
    """
    Record an Agent C failure while preserving the original error.
    """

    try:
        _append_audit_log(
            run_dir,
            [
                "## Agent C — Counterparty Resolution",
                "- status: failed",
                f"- error_type: {type(error).__name__}",
                f"- error: {error}",
            ],
        )
    except Exception:
        pass

    try:
        _update_agent_c_metrics(
            run_dir,
            {
                "status": "failed",
                "error_status": type(error).__name__,
                "error_message": str(error),
            },
        )
    except Exception:
        pass


def run_counterparty_agent(
    run_dir: str | Path,
    *,
    vendor_policy_path: str | Path | None = None,
    match_threshold: int = DEFAULT_MATCH_THRESHOLD,
    mismatch_threshold: int = (DEFAULT_COUNTERPARTY_MISMATCH_THRESHOLD),
    mismatch_severity: Severity = Severity.HIGH,
) -> NormalizedCounterparty:
    """
    Execute the complete Agent C flow for one run directory.
    """

    resolved_run_dir = Path(run_dir)

    if not resolved_run_dir.is_dir():
        raise CounterpartyInputError(
            "Agent C run directory was not found at: " f"{resolved_run_dir}"
        )

    try:
        # 1–2. Read both counterparty names.
        (
            manifest_counterparty_name,
            extracted_counterparty_name,
        ) = read_counterparty_names(resolved_run_dir)

        # 3. Load the run snapshot vendor master, with optional fallback.
        vendors = load_vendor_master_for_run(
            resolved_run_dir,
            policy_path=vendor_policy_path,
        )

        # 4. Read optional evidence.
        evidence_index = _read_optional_evidence_index(resolved_run_dir)

        evidence_item_count = 0

        if evidence_index is not None:
            evidence_item_count = len(evidence_index.get("evidence_items", []))

        # 5. Compare manifest and extracted names.
        comparison = compare_counterparty_names(
            manifest_counterparty_name,
            extracted_counterparty_name,
            mismatch_threshold=mismatch_threshold,
            mismatch_severity=mismatch_severity,
        )

        existing_findings: list[UnifiedFinding] = []

        if comparison.finding is not None:
            existing_findings.append(comparison.finding)

        # 6. Fuzzy match the extracted contract name.
        match_result = find_best_vendor_match(
            extracted_counterparty_name,
            vendors,
            threshold=match_threshold,
        )

        # 7–8. Classify and generate flags/findings.
        normalized_counterparty = classify_counterparty_status(
            match_result,
            manifest_counterparty_name=(manifest_counterparty_name),
            extracted_counterparty_name=(extracted_counterparty_name),
            findings=existing_findings,
        )

        # 9. Write normalized_counterparty.json.
        output_path = write_normalized_counterparty(
            resolved_run_dir,
            normalized_counterparty,
        )

        # Validate the exact JSON written to disk.
        validated_output = NormalizedCounterparty.model_validate_json(
            output_path.read_text(encoding="utf-8")
        )

        finding_types = [finding.finding_type for finding in validated_output.findings]

        # 10. Update audit_log.md.
        _append_audit_log(
            resolved_run_dir,
            [
                "## Agent C — Counterparty Resolution",
                "- status: completed",
                ("- manifest_counterparty: " f"{manifest_counterparty_name}"),
                ("- extracted_counterparty: " f"{extracted_counterparty_name}"),
                ("- manifest_extracted_similarity: " f"{comparison.similarity_score}"),
                ("- material_mismatch: " f"{comparison.is_mismatch}"),
                ("- matched_vendor_id: " f"{validated_output.matched_vendor_id}"),
                ("- matched_vendor_name: " f"{validated_output.matched_vendor_name}"),
                ("- vendor_match_score: " f"{validated_output.match_score}"),
                ("- counterparty_status: " f"{validated_output.status}"),
                ("- risk_level: " f"{validated_output.risk_level}"),
                f"- flags: {validated_output.flags}",
                f"- findings: {finding_types}",
                ("- evidence_index_available: " f"{evidence_index is not None}"),
                ("- output: " f"{NORMALIZED_COUNTERPARTY_FILENAME}"),
            ],
        )

        # 11. Update metrics.json.
        _update_agent_c_metrics(
            resolved_run_dir,
            {
                "status": "completed",
                "counterparty_status": (validated_output.status),
                "risk_level": (validated_output.risk_level),
                "match_score": (validated_output.match_score),
                "flag_count": len(validated_output.flags),
                "finding_count": len(validated_output.findings),
                "material_mismatch": (comparison.is_mismatch),
                "evidence_index_available": (evidence_index is not None),
                "evidence_item_count": (evidence_item_count),
                "output_file": (NORMALIZED_COUNTERPARTY_FILENAME),
            },
        )

        return validated_output

    except Exception as exc:
        _record_agent_c_failure(
            resolved_run_dir,
            exc,
        )
        raise


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """
    Run Agent C independently from a terminal.
    """

    parser = argparse.ArgumentParser(
        description=("Run Agent C counterparty resolution.")
    )

    parser.add_argument(
        "run_dir",
        help="Path to the ICRAS run directory.",
    )

    parser.add_argument(
        "--vendor-policy-path",
        default=None,
        help="Optional fallback vendor_master.csv path.",
    )

    parser.add_argument(
        "--match-threshold",
        type=int,
        default=DEFAULT_MATCH_THRESHOLD,
    )

    parser.add_argument(
        "--mismatch-threshold",
        type=int,
        default=(DEFAULT_COUNTERPARTY_MISMATCH_THRESHOLD),
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        result = run_counterparty_agent(
            args.run_dir,
            vendor_policy_path=args.vendor_policy_path,
            match_threshold=args.match_threshold,
            mismatch_threshold=args.mismatch_threshold,
        )
    except Exception as exc:
        print(
            f"Agent C failed: {exc}",
            file=sys.stderr,
        )
        return 1

    print(
        "Agent C completed: "
        f"status={result.status}, "
        f"match_score={result.match_score}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
