from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from rapidfuzz import fuzz
from collections.abc import Sequence

from app.schemas.finding import Severity, UnifiedFinding

# NormalizedCounterparty is the final structured Agent C result.
from app.schemas.normalized_counterparty import NormalizedCounterparty

# FuzzyMatchResult contains the vendor match and score produced
# by the RapidFuzz service.
from app.services.fuzzy_matcher import (
    FuzzyMatchResult,
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


# These are the exact JSON locations Agent C must read.
CONTEXT_PACKET_FILENAME = "context_packet.json"
EXTRACTED_CONTRACT_FILENAME = "extracted_contract.json"

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
        # Create the standardized finding used by the current project.
        finding = UnifiedFinding(
            # One manifest-versus-contract mismatch can exist per run,
            # so this fixed ID remains deterministic.
            finding_id="CP-MISMATCH-001",
            # UnifiedFinding currently uses "field" rather than
            # a separate category property.
            # "counterparty" therefore represents the category.
            field="counterparty",
            # Configurable medium or high mismatch risk.
            severity=mismatch_severity,
            # The message identifies this as a
            # counterparty_mismatch finding.
            message=(
                "counterparty_mismatch: The counterparty declared "
                "in the manifest materially differs from the "
                "counterparty extracted from the contract. "
                f"Similarity score {similarity_score} is below "
                f"the threshold {mismatch_threshold}."
            ),
            # Both JSON pointers are recorded so reviewers know
            # exactly which inputs were compared.
            evidence_ref=(
                "context_packet.json:"
                "counterparty_name_from_manifest; "
                "extracted_contract.json:"
                "parties.counterparty_name"
            ),
            # This mismatch concerns party identity rather than
            # one particular contract clause.
            clause_id=None,
            page_number=None,
            bbox=None,
            # Deterministic rule that triggered the finding.
            policy_rule="counterparty.names_must_match",
            # The manifest is the declared expected party.
            expected_value=manifest_counterparty_name,
            # Agent B's extracted party is the actual value.
            actual_value=extracted_counterparty_name,
            # Required downstream action.
            recommendation=(
                "Manually review the contract and manifest "
                "counterparty names before approval."
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


def classify_counterparty_status(
    match_result: FuzzyMatchResult,
    *,
    manifest_counterparty_name: str | None = None,
    extracted_counterparty_name: str | None = None,
    findings: Sequence[UnifiedFinding] = (),
) -> NormalizedCounterparty:
    """
    Convert a fuzzy vendor match into Agent C's normalized output.

    Classification rules:

    1. No accepted vendor match:
    status = unknown
    risk_level = high
    matched vendor fields = None

    2. Matched vendor with high risk_level or a high-risk status:
    status = high_risk
    risk_level = high

    3. Matched vendor with status new:
    status = new

    4. Matched vendor with status approved:
    status = approved

    Existing findings, such as a counterparty_mismatch finding from
    DZ-01.5, may be passed into this function and preserved.
    """

    # Require the structured matching result created by
    # find_best_vendor_match().
    if not isinstance(match_result, FuzzyMatchResult):
        raise TypeError("match_result must be a FuzzyMatchResult instance.")

    # Create a separate list so this function never modifies
    # the caller's original collection of findings.
    normalized_findings = list(findings)

    # Validate that every supplied finding follows the unified model.
    for finding_index, finding in enumerate(normalized_findings):
        if not isinstance(finding, UnifiedFinding):
            raise TypeError(
                "Every finding must be a UnifiedFinding instance. "
                f"Invalid finding at position {finding_index}."
            )

    # An unknown result must not expose an accepted vendor.
    if match_result.match_status == "unknown":
        if match_result.matched_vendor is not None:
            raise CounterpartyClassificationError(
                "An unknown fuzzy-match result must not contain " "a matched_vendor."
            )

        return NormalizedCounterparty(
            # Preserve the original name passed into the matcher.
            input_counterparty_name=(match_result.input_counterparty_name),
            # Preserve the manifest and extraction values for audit.
            manifest_counterparty_name=manifest_counterparty_name,
            extracted_counterparty_name=extracted_counterparty_name,
            # Unknown counterparties must not be connected to
            # the closest low-scoring candidate.
            matched_vendor_id=None,
            matched_vendor_name=None,
            # Preserve the best similarity score for auditability.
            match_score=match_result.match_score,
            # Required unknown classification.
            status="unknown",
            # Unknown counterparties require high-risk review.
            risk_level="high",
            # Downstream workflow flags.
            flags=[
                COUNTERPARTY_NOT_FOUND_FLAG,
                MANUAL_REVIEW_REQUIRED_FLAG,
            ],
            findings=normalized_findings,
        )

    # A result classified as matched must contain the accepted vendor.
    matched_vendor = match_result.matched_vendor

    if matched_vendor is None:
        raise CounterpartyClassificationError(
            "A matched fuzzy-match result must contain matched_vendor."
        )

    # Normalize metadata so differences such as "Approved",
    # "APPROVED", and "approved" behave identically.
    vendor_status = matched_vendor.status.strip().casefold()
    vendor_risk_level = matched_vendor.risk_level.strip().casefold()

    # NormalizedCounterparty only permits low, medium, or high.
    if vendor_risk_level not in SUPPORTED_VENDOR_RISK_LEVELS:
        raise CounterpartyClassificationError(
            "Unsupported vendor risk_level "
            f"'{matched_vendor.risk_level}' for vendor "
            f"'{matched_vendor.vendor_id}'."
        )

    output_status: str
    output_risk_level: str
    output_flags: list[str]

    # High risk takes priority over approved or new.
    #
    # For example, a vendor may have:
    # status = new
    # risk_level = high
    #
    # The final Agent C status must be high_risk.
    if vendor_risk_level == "high" or vendor_status in HIGH_RISK_VENDOR_STATUSES:
        output_status = "high_risk"
        output_risk_level = "high"
        output_flags = [
            HIGH_RISK_COUNTERPARTY_FLAG,
            MANUAL_REVIEW_REQUIRED_FLAG,
        ]

    elif vendor_status == "new":
        output_status = "new"
        output_risk_level = vendor_risk_level
        output_flags = [
            NEW_COUNTERPARTY_FLAG,
        ]

    elif vendor_status == "approved":
        output_status = "approved"
        output_risk_level = vendor_risk_level
        output_flags = []

    else:
        # Do not silently classify unsupported vendor statuses.
        # A clear exception allows the orchestrator to fail safely.
        raise CounterpartyClassificationError(
            "Unsupported vendor status "
            f"'{matched_vendor.status}' for vendor "
            f"'{matched_vendor.vendor_id}'."
        )

    return NormalizedCounterparty(
        input_counterparty_name=(match_result.input_counterparty_name),
        manifest_counterparty_name=manifest_counterparty_name,
        extracted_counterparty_name=extracted_counterparty_name,
        # Accepted matches retain the official vendor details.
        matched_vendor_id=matched_vendor.vendor_id,
        matched_vendor_name=matched_vendor.vendor_name,
        # Preserve the RapidFuzz score.
        match_score=match_result.match_score,
        status=output_status,
        risk_level=output_risk_level,
        flags=output_flags,
        findings=normalized_findings,
    )
