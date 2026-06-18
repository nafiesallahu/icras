from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.schemas.approval_policy import ApprovalPolicy
from app.schemas.clause_analysis import (
    ClauseAnalysis,
    ScoredRiskFinding,
)
from app.schemas.finding import UnifiedFinding
from app.schemas.context_packet import ContextPacket
from app.schemas.evidence import EvidenceIndex, EvidenceItem
from app.schemas.normalized_counterparty import NormalizedCounterparty
from app.schemas.triage_finding import (
    TriageCategory,
    TriageFinding,
    TriagePreparationResult,
    TriageRiskLevel,
    TriageSeverity,
)
from app.schemas.validation_result import (
    ValidationFinding,
    ValidationResult,
)

ModelType = TypeVar(
    "ModelType",
    bound=BaseModel,
)


class TriageInputError(RuntimeError):
    """
    Base error raised when Agent H cannot safely load its inputs.
    """


class TriageEvidenceError(TriageInputError):
    """
    Raised when a finding references evidence that does not exist
    in evidence_index.json or cannot be resolved safely.
    """


class MissingTriageInputError(TriageInputError):
    """
    Raised when one or more required Agent H files are missing.
    """


class MalformedTriageJsonError(TriageInputError):
    """
    Raised when a required JSON file cannot be parsed.
    """


class TriageSchemaValidationError(TriageInputError):
    """
    Raised when parsed input does not satisfy its Pydantic schema.
    """


class ApprovalPolicyValidationError(TriageInputError):
    """
    Raised when approval_policy.yaml is missing or invalid.
    """


class TriageConsistencyError(TriageInputError):
    """
    Raised when validated Agent H inputs contradict one another.

    Examples include mismatched run IDs, mismatched contract IDs,
    or an upstream artifact reporting a failed state.
    """


@dataclass(frozen=True)
class TriageInputs:
    """
    Validated inputs consumed by Agent H.

    This container stores the already validated upstream artifacts.
    Agent H must not modify these objects.
    """

    context_packet: ContextPacket
    normalized_counterparty: NormalizedCounterparty
    validation_result: ValidationResult
    clause_analysis: ClauseAnalysis
    evidence_index: EvidenceIndex
    approval_policy: ApprovalPolicy


REQUIRED_INPUT_PATHS: dict[str, Path] = {
    "context_packet": Path("context_packet.json"),
    "normalized_counterparty": Path("normalized_counterparty.json"),
    "validation_result": Path("validation_result.json"),
    "clause_analysis": Path("clause_analysis.json"),
    "evidence_index": Path("evidence_index.json"),
    "approval_policy": Path("input_snapshot") / "approval_policy.yaml",
}

CATEGORY_ALIASES: dict[str, TriageCategory] = {
    "finance": TriageCategory.FINANCE,
    "financial": TriageCategory.FINANCE,
    "legal": TriageCategory.LEGAL,
    "compliance": TriageCategory.COMPLIANCE,
    "counterparty": TriageCategory.COUNTERPARTY,
    "vendor": TriageCategory.COUNTERPARTY,
    "procurement": TriageCategory.PROCUREMENT,
    "operational": TriageCategory.OPERATIONAL,
    "operations": TriageCategory.OPERATIONAL,
    "manual_review": TriageCategory.MANUAL_REVIEW,
    "manual": TriageCategory.MANUAL_REVIEW,
    "blocking": TriageCategory.BLOCKING,
    "rejection": TriageCategory.BLOCKING,
    "general": TriageCategory.GENERAL,
}

PAGE_REFERENCE_PATTERN = re.compile(
    r"(?:^|[#&?])page=(\d+)(?:$|[&#])",
    re.IGNORECASE,
)

CLAUSE_REFERENCE_PATTERN = re.compile(
    r"(?:^|[#&?])clause=([^&#]+)",
    re.IGNORECASE,
)

SEVERITY_DEFAULT_SCORES: dict[TriageSeverity, int] = {
    TriageSeverity.LOW: 25,
    TriageSeverity.MEDIUM: 50,
    TriageSeverity.HIGH: 75,
    TriageSeverity.CRITICAL: 100,
}

SEVERITY_RANK: dict[TriageSeverity, int] = {
    TriageSeverity.LOW: 1,
    TriageSeverity.MEDIUM: 2,
    TriageSeverity.HIGH: 3,
    TriageSeverity.CRITICAL: 4,
}

RISK_LEVEL_RANK: dict[TriageRiskLevel, int] = {
    TriageRiskLevel.LOW: 1,
    TriageRiskLevel.MEDIUM: 2,
    TriageRiskLevel.HIGH: 3,
    TriageRiskLevel.CRITICAL: 4,
}

GENERIC_FINDING_TYPES: set[str] = {
    "unspecified",
    "validation_finding",
    "risk_finding",
    "counterparty_finding",
    "general_finding",
}

FALLBACK_CATEGORY_SIGNALS: dict[TriageCategory, set[str]] = {
    TriageCategory.FINANCE: {
        "payment_terms",
        "payment_terms_days",
        "payment_terms_exceeded",
        "invoice",
        "net_30",
        "net_60",
        "net_90",
    },
    TriageCategory.LEGAL: {
        "liability_cap",
        "missing_liability_cap",
        "governing_law",
        "conflicting_governing_law",
        "jurisdiction_conflict",
        "auto_renewal_without_opt_out",
        "missing_required_clause",
        "termination",
    },
    TriageCategory.COMPLIANCE: {
        "gdpr",
        "missing_gdpr_clause",
        "data_processing",
        "personal_data",
        "high_risk_jurisdiction",
        "compliance",
    },
    TriageCategory.COUNTERPARTY: {
        "counterparty",
        "counterparty_mismatch",
        "counterparty_not_found",
        "unknown_counterparty",
        "high_risk_counterparty",
        "vendor",
        "unknown_vendor",
    },
    TriageCategory.MANUAL_REVIEW: {
        "low_confidence",
        "low_confidence_extraction",
        "low_confidence_signature",
        "manual_review_required",
        "signature_confidence",
    },
    TriageCategory.BLOCKING: {
        "reject_or_block",
        "blocked",
        "blocking",
        "prohibited",
    },
}


def _load_json_model(
    path: Path,
    model_type: type[ModelType],
) -> ModelType:
    """
    Read, parse, and validate one required JSON artifact.

    Parsing and Pydantic validation are kept separate so Agent H can
    distinguish malformed JSON from structurally invalid JSON.
    """

    try:
        raw_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise TriageInputError(
            f"Agent H could not read required input: {path}"
        ) from exc

    try:
        raw_data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise MalformedTriageJsonError(
            f"Malformed JSON in required Agent H input: {path}. "
            f"Line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    try:
        return model_type.model_validate(raw_data)
    except ValidationError as exc:
        raise TriageSchemaValidationError(
            f"Agent H input failed schema validation: {path}\n" f"{exc}"
        ) from exc


def _resolve_required_paths(
    run_dir: str | Path,
) -> dict[str, Path]:
    """
    Resolve every required Agent H input relative to one run directory.
    """

    run_path = Path(run_dir)

    if not run_path.exists():
        raise MissingTriageInputError(f"Run directory does not exist: {run_path}")

    if not run_path.is_dir():
        raise MissingTriageInputError(
            f"Agent H run path is not a directory: {run_path}"
        )

    resolved_paths = {
        name: run_path / relative_path
        for name, relative_path in REQUIRED_INPUT_PATHS.items()
    }

    missing_paths = sorted(
        str(path) for path in resolved_paths.values() if not path.is_file()
    )

    if missing_paths:
        formatted_paths = "\n".join(f"- {path}" for path in missing_paths)

        raise MissingTriageInputError(
            "Missing required Agent H input files:\n" + formatted_paths
        )

    return resolved_paths


def _enum_or_string_value(value: object | None) -> str | None:
    """
    Return the string value of an enum or ordinary value.
    """

    if value is None:
        return None

    enum_value = getattr(value, "value", value)
    return str(enum_value)


def _optional_text(value: object | None) -> str | None:
    """
    Convert an optional upstream value into clean text.

    Blank strings are treated as missing values.
    """

    if value is None:
        return None

    raw_value = getattr(value, "value", value)
    cleaned = str(raw_value).strip()

    return cleaned or None


def _optional_confidence(
    value: object | None,
) -> float | None:
    """
    Normalize an optional confidence value.

    The returned value is always a float between 0 and 1.
    """

    if value is None:
        return None

    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise TriageConsistencyError(f"Invalid confidence value: {value!r}.") from exc

    if not 0.0 <= confidence <= 1.0:
        raise TriageConsistencyError(
            "Confidence must be between 0 and 1, " f"received {confidence!r}."
        )

    return confidence


def _optional_string_list(
    value: object | None,
) -> list[str]:
    """
    Normalize optional human-review questions or similar text lists.
    """

    if value is None:
        return []

    if isinstance(value, str):
        cleaned = value.strip()
        return [cleaned] if cleaned else []

    if not isinstance(value, (list, tuple, set)):
        raise TriageConsistencyError(
            "Expected a collection of strings, " f"received {type(value).__name__}."
        )

    return _unique_nonblank_values(list(value))


def _humanize_identifier(value: object | None) -> str:
    """
    Convert a machine-readable identifier into a readable title.

    Example:
        payment_terms_exceeded
        -> Payment terms exceeded
    """

    normalized = _optional_text(value)

    if normalized is None:
        return "Unspecified finding"

    readable = normalized.replace("_", " ").replace("-", " ")
    readable = " ".join(readable.split())

    return readable[:1].upper() + readable[1:]


def _normalize_triage_severity(
    value: object,
) -> TriageSeverity:
    """
    Convert an upstream severity or risk-tier value into Agent H's
    severity enum.
    """

    normalized = _normalize_signal(value)

    try:
        return TriageSeverity(normalized)
    except ValueError as exc:
        raise TriageConsistencyError(
            f"Unsupported finding severity: {value!r}."
        ) from exc


def _normalize_triage_risk_level(
    value: object,
) -> TriageRiskLevel:
    """
    Convert an upstream risk value into Agent H's risk-level enum.
    """

    normalized = _normalize_signal(value)

    try:
        return TriageRiskLevel(normalized)
    except ValueError as exc:
        raise TriageConsistencyError(
            f"Unsupported finding risk level: {value!r}."
        ) from exc


def _strongest_severity(
    left: TriageSeverity,
    right: TriageSeverity,
) -> TriageSeverity:
    """
    Return the more severe value.
    """

    return max(
        (left, right),
        key=lambda value: SEVERITY_RANK[value],
    )


def _strongest_risk_level(
    left: TriageRiskLevel,
    right: TriageRiskLevel,
) -> TriageRiskLevel:
    """
    Return the higher risk level.
    """

    return max(
        (left, right),
        key=lambda value: RISK_LEVEL_RANK[value],
    )


def _choose_most_complete_text(
    *values: str | None,
) -> str | None:
    """
    Select the most complete nonblank text deterministically.

    Longer text is considered more complete. Alphabetical ordering
    breaks ties so the result does not depend on input order.
    """

    candidates = {
        value.strip() for value in values if value is not None and value.strip()
    }

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda value: (
            len(value),
            value.casefold(),
            value,
        ),
    )


def _choose_finding_type(
    left: str,
    right: str,
) -> str:
    """
    Prefer a specific business finding type over a generic fallback.
    """

    left_normalized = _normalize_signal(left)
    right_normalized = _normalize_signal(right)

    left_is_generic = left_normalized in GENERIC_FINDING_TYPES
    right_is_generic = right_normalized in GENERIC_FINDING_TYPES

    if left_is_generic and not right_is_generic:
        return right

    if right_is_generic and not left_is_generic:
        return left

    selected = _choose_most_complete_text(
        left,
        right,
    )

    return selected or "unspecified"


def _merge_categories(
    left: TriageCategory,
    right: TriageCategory,
) -> TriageCategory:
    """
    Resolve compatible category differences deterministically.

    General and operational categories may be replaced by a more
    specific category. Counterparty and procurement are considered
    compatible because both route to vendor/procurement review.
    Material category conflicts are rejected rather than hidden.
    """

    if left == right:
        return left

    weak_categories = {
        TriageCategory.GENERAL,
        TriageCategory.OPERATIONAL,
    }

    if left in weak_categories and right not in weak_categories:
        return right

    if right in weak_categories and left not in weak_categories:
        return left

    counterparty_categories = {
        TriageCategory.COUNTERPARTY,
        TriageCategory.PROCUREMENT,
    }

    if {
        left,
        right,
    } <= counterparty_categories:
        return TriageCategory.COUNTERPARTY

    raise TriageConsistencyError(
        "Duplicate findings contain conflicting actionable "
        f"categories: {left.value!r} and {right.value!r}."
    )


def _categories_are_compatible(
    left: TriageCategory,
    right: TriageCategory,
) -> bool:
    """
    Return whether two categories can describe the same issue.

    General and operational categories may be merged with a more
    specific category. Counterparty and procurement are compatible
    because both represent vendor-related review.
    """

    if left == right:
        return True

    weak_categories = {
        TriageCategory.GENERAL,
        TriageCategory.OPERATIONAL,
    }

    if left in weak_categories or right in weak_categories:
        return True

    counterparty_categories = {
        TriageCategory.COUNTERPARTY,
        TriageCategory.PROCUREMENT,
    }

    return {
        left,
        right,
    } <= counterparty_categories


def _normalized_values_match(
    left: object | None,
    right: object | None,
) -> bool:
    """
    Return True only when two nonblank normalized values match.
    """

    normalized_left = _normalize_signal(left)
    normalized_right = _normalize_signal(right)

    return bool(
        normalized_left and normalized_right and normalized_left == normalized_right
    )


def _normalized_values_conflict(
    left: object | None,
    right: object | None,
) -> bool:
    """
    Return True when both values exist but normalize differently.
    """

    normalized_left = _normalize_signal(left)
    normalized_right = _normalize_signal(right)

    return bool(
        normalized_left and normalized_right and normalized_left != normalized_right
    )


def _evidence_supported_duplicate(
    left: TriageFinding,
    right: TriageFinding,
) -> bool:
    """
    Detect duplicates supported by overlapping evidence.

    Evidence overlap alone is not sufficient because the same clause
    can support more than one distinct finding. At least one stable
    business anchor must also match.
    """

    shared_evidence_ids = set(left.evidence_ids) & set(right.evidence_ids)

    if not shared_evidence_ids:
        return False

    if not _categories_are_compatible(
        left.category,
        right.category,
    ):
        return False

    # Findings with contradictory expected or actual values should
    # remain separate even if they cite the same evidence.
    if _normalized_values_conflict(
        left.expected_value,
        right.expected_value,
    ):
        return False

    if _normalized_values_conflict(
        left.actual_value,
        right.actual_value,
    ):
        return False

    stable_anchor_matches = (
        _normalized_values_match(
            left.finding_type,
            right.finding_type,
        ),
        _normalized_values_match(
            left.policy_rule,
            right.policy_rule,
        ),
        _normalized_values_match(
            left.clause_id,
            right.clause_id,
        ),
        _normalized_values_match(
            left.field_reference,
            right.field_reference,
        ),
    )

    return any(stable_anchor_matches)


def _business_issue_key(
    finding: TriageFinding,
) -> tuple[str, ...]:
    """
    Build a deterministic business key for duplicate detection.

    Source-agent names and source IDs are intentionally excluded
    because different agents may report the same underlying issue.
    """

    finding_type = _normalize_signal(finding.finding_type)

    policy_rule = _normalize_signal(finding.policy_rule)

    field_reference = _normalize_signal(finding.field_reference)

    clause_id = _normalize_signal(finding.clause_id)

    # A generic finding without policy, field, or clause lineage
    # needs its title to prevent unrelated generic findings from
    # being merged.
    include_title = finding_type in GENERIC_FINDING_TYPES and not any(
        (
            policy_rule,
            field_reference,
            clause_id,
        )
    )

    title = _normalize_signal(finding.title) if include_title else ""

    return (
        finding_type,
        finding.category.value,
        clause_id,
        field_reference,
        policy_rule,
        _normalize_signal(finding.expected_value),
        _normalize_signal(finding.actual_value),
        title,
    )


def findings_are_duplicates(
    left: TriageFinding,
    right: TriageFinding,
) -> bool:
    """
    Determine whether two normalized findings represent the same issue.

    Duplicate detection considers:

    1. identical deterministic Agent H IDs;
    2. shared upstream finding IDs;
    3. identical normalized business keys;
    4. overlapping evidence plus a matching stable business anchor.
    """

    if left.finding_id == right.finding_id:
        return True

    left_source_ids = set(left.source_finding_ids)

    right_source_ids = set(right.source_finding_ids)

    if left_source_ids & right_source_ids:
        return True

    if _business_issue_key(left) == _business_issue_key(right):
        return True

    return _evidence_supported_duplicate(
        left,
        right,
    )


def _evidence_location_key(
    finding: TriageFinding,
) -> tuple[object, ...]:
    """
    Rank how complete a finding's primary evidence location is.
    """

    populated_fields = sum(
        value is not None
        for value in (
            finding.document_name,
            finding.page_number,
            finding.bbox,
            finding.text_excerpt,
        )
    )

    bbox_text = (
        json.dumps(
            finding.bbox.model_dump(mode="json"),
            sort_keys=True,
        )
        if finding.bbox is not None
        else ""
    )

    return (
        populated_fields,
        len(finding.text_excerpt or ""),
        finding.document_name or "",
        finding.page_number or 0,
        bbox_text,
        finding.text_excerpt or "",
        finding.finding_id,
    )


def _choose_page_number(
    left: TriageFinding,
    right: TriageFinding,
) -> int | None:
    """
    Preserve an available page number deterministically.

    When both findings contain conflicting pages, use the page from
    the finding with the richer overall evidence location.
    """

    if left.page_number is None:
        return right.page_number

    if right.page_number is None:
        return left.page_number

    if left.page_number == right.page_number:
        return left.page_number

    richer_finding = _finding_with_richer_evidence_location(
        left,
        right,
    )

    return richer_finding.page_number


def _choose_bbox(
    left: TriageFinding,
    right: TriageFinding,
):
    """
    Preserve an available bounding box deterministically.
    """

    if left.bbox is None:
        return right.bbox

    if right.bbox is None:
        return left.bbox

    left_serialized = json.dumps(
        left.bbox.model_dump(mode="json"),
        sort_keys=True,
    )

    right_serialized = json.dumps(
        right.bbox.model_dump(mode="json"),
        sort_keys=True,
    )

    if left_serialized == right_serialized:
        return left.bbox

    richer_finding = _finding_with_richer_evidence_location(
        left,
        right,
    )

    if richer_finding.bbox is not None:
        return richer_finding.bbox

    # Deterministic fallback.
    return max(
        (left.bbox, right.bbox),
        key=lambda bbox: json.dumps(
            bbox.model_dump(mode="json"),
            sort_keys=True,
        ),
    )


def merge_triage_findings(
    left: TriageFinding,
    right: TriageFinding,
) -> TriageFinding:
    """
    Merge two duplicate findings without losing useful metadata.
    """

    category = _merge_categories(
        left.category,
        right.category,
    )

    finding_type = _choose_finding_type(
        left.finding_type,
        right.finding_type,
    )

    clause_id = _choose_most_complete_text(
        left.clause_id,
        right.clause_id,
    )

    clause_type = _choose_most_complete_text(
        left.clause_type,
        right.clause_type,
    )

    field_reference = _choose_most_complete_text(
        left.field_reference,
        right.field_reference,
    )

    policy_rule = _choose_most_complete_text(
        left.policy_rule,
        right.policy_rule,
    )

    expected_value = _choose_most_complete_text(
        left.expected_value,
        right.expected_value,
    )

    actual_value = _choose_most_complete_text(
        left.actual_value,
        right.actual_value,
    )

    primary_recommendation = _choose_most_complete_text(
        left.recommendation,
        right.recommendation,
        *left.additional_recommendations,
        *right.additional_recommendations,
    )

    all_recommendations = _unique_nonblank_values(
        [
            left.recommendation,
            right.recommendation,
            *left.additional_recommendations,
            *right.additional_recommendations,
        ]
    )

    additional_recommendations = [
        recommendation
        for recommendation in all_recommendations
        if recommendation != primary_recommendation
    ]

    confidence_values = [
        confidence
        for confidence in (
            left.confidence,
            right.confidence,
        )
        if confidence is not None
    ]

    merged_confidence = max(confidence_values) if confidence_values else None

    merged_document_name = _choose_most_complete_text(
        left.document_name,
        right.document_name,
    )

    merged_page_number = _choose_page_number(
        left,
        right,
    )

    merged_bbox = _choose_bbox(
        left,
        right,
    )

    merged_text_excerpt = _choose_most_complete_text(
        left.text_excerpt,
        right.text_excerpt,
    )

    merged_finding_id = _stable_triage_finding_id(
        finding_type=finding_type,
        category=category,
        clause_id=clause_id,
        field_reference=field_reference,
        policy_rule=policy_rule,
        expected_value=expected_value,
        actual_value=actual_value,
    )

    return TriageFinding(
        finding_id=merged_finding_id,
        source_finding_ids=_unique_nonblank_values(
            [
                *left.source_finding_ids,
                *right.source_finding_ids,
            ]
        ),
        source_agents=_unique_nonblank_values(
            [
                *left.source_agents,
                *right.source_agents,
            ]
        ),
        finding_type=finding_type,
        title=(
            _choose_most_complete_text(
                left.title,
                right.title,
            )
            or _humanize_identifier(finding_type)
        ),
        category=category,
        severity=_strongest_severity(
            left.severity,
            right.severity,
        ),
        risk_level=_strongest_risk_level(
            left.risk_level,
            right.risk_level,
        ),
        score=max(left.score, right.score),
        confidence=merged_confidence,
        clause_id=clause_id,
        clause_type=clause_type,
        field_reference=field_reference,
        policy_rule=policy_rule,
        expected_value=expected_value,
        actual_value=actual_value,
        recommendation=primary_recommendation,
        additional_recommendations=(additional_recommendations),
        open_questions=_unique_nonblank_values(
            [
                *left.open_questions,
                *right.open_questions,
            ]
        ),
        requires_human_review=(
            left.requires_human_review or right.requires_human_review
        ),
        evidence_ids=_unique_nonblank_values(
            [
                *left.evidence_ids,
                *right.evidence_ids,
            ]
        ),
        document_name=merged_document_name,
        page_number=merged_page_number,
        bbox=merged_bbox,
        text_excerpt=merged_text_excerpt,
    )


def sort_triage_findings(
    findings: list[TriageFinding],
) -> list[TriageFinding]:
    """
    Return findings in deterministic triage order.

    Stronger findings appear first. Stable business fields break ties.
    """

    return sorted(
        findings,
        key=lambda finding: (
            -SEVERITY_RANK[finding.severity],
            -RISK_LEVEL_RANK[finding.risk_level],
            -finding.score,
            finding.category.value,
            _normalize_signal(finding.finding_type),
            _normalize_signal(finding.clause_id),
            _normalize_signal(finding.field_reference),
            finding.finding_id,
        ),
    )


def deduplicate_findings(
    findings: list[TriageFinding],
) -> tuple[list[TriageFinding], int]:
    """
    Merge duplicate normalized findings deterministically.

    Returns:
        A tuple containing:
        1. consolidated findings;
        2. number of duplicate entries removed.
    """

    ordered_input = sorted(
        findings,
        key=lambda finding: (
            _business_issue_key(finding),
            finding.finding_id,
            tuple(sorted(finding.source_agents)),
            tuple(sorted(finding.source_finding_ids)),
        ),
    )

    consolidated: list[TriageFinding] = []

    for finding in ordered_input:
        duplicate_index: int | None = None

        for index, existing in enumerate(consolidated):
            if findings_are_duplicates(
                existing,
                finding,
            ):
                duplicate_index = index
                break

        if duplicate_index is None:
            consolidated.append(finding)
            continue

        consolidated[duplicate_index] = merge_triage_findings(
            consolidated[duplicate_index],
            finding,
        )

    sorted_findings = sort_triage_findings(consolidated)

    duplicates_removed = len(findings) - len(sorted_findings)

    return sorted_findings, duplicates_removed


def prepare_triage_findings(
    inputs: TriageInputs,
) -> TriagePreparationResult:
    """
    Validate, normalize, categorize, consolidate, and sort all findings
    required by DZ-02.1.

    This function operates on already loaded and validated inputs.

    It does not produce the final approval_packet.json. Later Agent H
    subtasks will use this prepared result for routing and approval
    packet generation.
    """

    # Validate that evidence IDs are unique even when the run contains
    # no findings. Duplicate IDs would make later evidence references
    # ambiguous.
    build_evidence_lookup(inputs.evidence_index)

    normalized_findings = collect_normalized_findings(inputs)

    consolidated_findings, duplicates_removed = deduplicate_findings(
        normalized_findings
    )

    findings_by_source = {
        "agent_c": sum(
            "agent_c" in finding.source_agents for finding in normalized_findings
        ),
        "agent_d": sum(
            "agent_d" in finding.source_agents for finding in normalized_findings
        ),
        "agent_e": sum(
            "agent_e" in finding.source_agents for finding in normalized_findings
        ),
    }

    return TriagePreparationResult(
        run_id=inputs.context_packet.run_id,
        contract_id=inputs.context_packet.contract_id,
        findings_received=len(normalized_findings),
        duplicates_removed=duplicates_removed,
        findings_by_source=findings_by_source,
        consolidated_findings=consolidated_findings,
    )


def prepare_triage_run(
    run_dir: str | Path,
) -> TriagePreparationResult:
    """
    Execute the complete DZ-02.1 preparation process for one run.

    Processing order:

    1. verify required files;
    2. parse JSON and YAML;
    3. validate all schemas;
    4. validate run and contract consistency;
    5. normalize Agent C, D, and E findings;
    6. resolve and validate evidence;
    7. categorize findings;
    8. deduplicate and merge findings;
    9. return deterministic consolidated output.
    """

    inputs = load_triage_inputs(run_dir)

    return prepare_triage_findings(inputs)


def write_triage_preparation_result(
    result: TriagePreparationResult,
    output_path: str | Path,
) -> None:
    """
    Write a deterministic normalized preparation artifact.

    This is an internal Agent H artifact and is not the final
    approval_packet.json.
    """

    destination = Path(output_path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        result.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def _finding_with_richer_evidence_location(
    left: TriageFinding,
    right: TriageFinding,
) -> TriageFinding:
    """
    Select the finding with the more complete evidence location.
    """

    return max(
        (left, right),
        key=_evidence_location_key,
    )


def _default_score_for_severity(
    severity: TriageSeverity,
) -> int:
    """
    Return a deterministic score when the upstream agent does not
    provide one.
    """

    return SEVERITY_DEFAULT_SCORES[severity]


def _requires_review_from_severity(
    severity: TriageSeverity,
) -> bool:
    """
    Infer human-review need for upstream schemas that do not expose
    requires_human_review.
    """

    return severity in {
        TriageSeverity.MEDIUM,
        TriageSeverity.HIGH,
        TriageSeverity.CRITICAL,
    }


def _stable_triage_finding_id(
    *,
    finding_type: str,
    category: TriageCategory,
    clause_id: str | None,
    field_reference: str | None,
    policy_rule: str | None,
    expected_value: str | None,
    actual_value: str | None,
) -> str:
    """
    Generate a deterministic Agent H finding ID from business fields.

    Python's built-in hash() is intentionally not used because it can
    produce different values across processes.
    """

    fingerprint_parts = (
        finding_type,
        category.value,
        clause_id,
        field_reference,
        policy_rule,
        expected_value,
        actual_value,
    )

    fingerprint = "\x1f".join(_normalize_signal(value) for value in fingerprint_parts)

    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16].upper()

    return f"TRIAGE-{digest}"


def _unique_nonblank_values(
    values: list[object | None],
) -> list[str]:
    """
    Return unique, nonblank strings in deterministic sorted order.
    """

    normalized = {
        cleaned for value in values if (cleaned := _optional_text(value)) is not None
    }

    return sorted(normalized)


def _primary_evidence_fields(
    evidence_index: EvidenceIndex,
    evidence_ids: list[str],
    *,
    fallback_page_number: int | None = None,
    fallback_bbox: object | None = None,
) -> tuple[
    str | None,
    int | None,
    object | None,
    str | None,
]:
    """
    Return the main evidence location while preserving upstream
    location values when no indexed evidence item is available.
    """

    primary_item = get_primary_evidence_item(
        evidence_index,
        evidence_ids,
    )

    if primary_item is None:
        return (
            None,
            fallback_page_number,
            fallback_bbox,
            None,
        )

    return (
        primary_item.filename,
        primary_item.page,
        primary_item.bbox if primary_item.bbox is not None else fallback_bbox,
        primary_item.text_excerpt,
    )


def normalize_agent_c_finding(
    finding: UnifiedFinding,
    approval_policy: ApprovalPolicy,
    evidence_index: EvidenceIndex,
) -> TriageFinding:
    """
    Convert one Agent C UnifiedFinding into Agent H's normalized
    TriageFinding format.
    """

    finding_type = (
        _optional_text(finding.finding_type)
        or _optional_text(finding.policy_rule)
        or _optional_text(finding.field)
        or "counterparty_finding"
    )

    title = _optional_text(finding.message) or _humanize_identifier(finding_type)

    field_reference = _optional_text(finding.field)
    policy_rule = _optional_text(finding.policy_rule)
    clause_id = _optional_text(finding.clause_id)

    expected_value = _optional_text(finding.expected) or _optional_text(
        finding.expected_value
    )

    actual_value = _optional_text(finding.actual) or _optional_text(
        finding.actual_value
    )

    category = categorize_finding(
        approval_policy,
        explicit_category=finding.category,
        finding_type=finding_type,
        policy_rule=policy_rule,
        field_reference=field_reference,
        title=title,
    )

    severity = _normalize_triage_severity(finding.severity)

    risk_level = _normalize_triage_risk_level(finding.risk_level)

    evidence_ids = resolve_evidence_ids(
        evidence_index,
        explicit_evidence_ids=list(finding.evidence_ids),
        evidence_ref=_optional_text(finding.evidence_ref),
        clause_id=clause_id,
        page_number=finding.page_number,
    )

    (
        document_name,
        resolved_page_number,
        resolved_bbox,
        text_excerpt,
    ) = _primary_evidence_fields(
        evidence_index,
        evidence_ids,
        fallback_page_number=finding.page_number,
        fallback_bbox=finding.bbox,
    )

    raw_source_agent = _optional_text(finding.source_agent)

    source_agent = (
        "agent_c" if raw_source_agent in {None, "unknown"} else raw_source_agent
    )

    triage_finding_id = _stable_triage_finding_id(
        finding_type=finding_type,
        category=category,
        clause_id=clause_id,
        field_reference=field_reference,
        policy_rule=policy_rule,
        expected_value=expected_value,
        actual_value=actual_value,
    )

    return TriageFinding(
        finding_id=triage_finding_id,
        source_finding_ids=[finding.finding_id],
        source_agents=[source_agent],
        finding_type=finding_type,
        title=title,
        category=category,
        severity=severity,
        risk_level=risk_level,
        score=finding.score,
        confidence=_optional_confidence(getattr(finding, "confidence", None)),
        clause_id=clause_id,
        clause_type=_optional_text(getattr(finding, "clause_type", None)),
        field_reference=field_reference,
        policy_rule=policy_rule,
        expected_value=expected_value,
        actual_value=actual_value,
        recommendation=_optional_text(finding.recommendation),
        additional_recommendations=[],
        open_questions=_optional_string_list(getattr(finding, "open_questions", None)),
        requires_human_review=(finding.requires_human_review),
        evidence_ids=evidence_ids,
        document_name=document_name,
        page_number=resolved_page_number,
        bbox=resolved_bbox,
        text_excerpt=text_excerpt,
    )


def normalize_agent_d_finding(
    finding: ValidationFinding,
    approval_policy: ApprovalPolicy,
    evidence_index: EvidenceIndex,
) -> TriageFinding:
    """
    Convert one Agent D ValidationFinding into Agent H's normalized
    TriageFinding format.
    """

    field_reference = _optional_text(finding.field)
    policy_rule = _optional_text(finding.policy_rule)
    clause_id = _optional_text(finding.clause_id)

    finding_type = policy_rule or field_reference or "validation_finding"

    title = _optional_text(finding.message) or _humanize_identifier(finding_type)

    expected_value = _optional_text(finding.expected_value)

    actual_value = _optional_text(finding.actual_value)

    category = categorize_finding(
        approval_policy,
        finding_type=finding_type,
        policy_rule=policy_rule,
        field_reference=field_reference,
        title=title,
    )

    severity = _normalize_triage_severity(finding.severity)

    risk_level = _normalize_triage_risk_level(severity.value)

    evidence_ids = resolve_evidence_ids(
        evidence_index,
        evidence_ref=_optional_text(finding.evidence_ref),
        clause_id=clause_id,
        page_number=finding.page_number,
    )

    (
        document_name,
        resolved_page_number,
        resolved_bbox,
        text_excerpt,
    ) = _primary_evidence_fields(
        evidence_index,
        evidence_ids,
        fallback_page_number=finding.page_number,
        fallback_bbox=finding.bbox,
    )

    triage_finding_id = _stable_triage_finding_id(
        finding_type=finding_type,
        category=category,
        clause_id=clause_id,
        field_reference=field_reference,
        policy_rule=policy_rule,
        expected_value=expected_value,
        actual_value=actual_value,
    )

    return TriageFinding(
        finding_id=triage_finding_id,
        source_finding_ids=[finding.finding_id],
        source_agents=["agent_d"],
        finding_type=finding_type,
        title=title,
        category=category,
        severity=severity,
        risk_level=risk_level,
        score=_default_score_for_severity(severity),
        confidence=_optional_confidence(getattr(finding, "confidence", None)),
        clause_id=clause_id,
        clause_type=_optional_text(getattr(finding, "clause_type", None)),
        field_reference=field_reference,
        policy_rule=policy_rule,
        expected_value=expected_value,
        actual_value=actual_value,
        recommendation=_optional_text(finding.recommendation),
        additional_recommendations=[],
        open_questions=_optional_string_list(getattr(finding, "open_questions", None)),
        requires_human_review=(_requires_review_from_severity(severity)),
        evidence_ids=evidence_ids,
        document_name=document_name,
        page_number=resolved_page_number,
        bbox=resolved_bbox,
        text_excerpt=text_excerpt,
    )


def normalize_agent_e_finding(
    finding: ScoredRiskFinding,
    approval_policy: ApprovalPolicy,
    evidence_index: EvidenceIndex,
) -> TriageFinding:
    """
    Convert one Agent E ScoredRiskFinding into Agent H's normalized
    TriageFinding format.
    """

    field_reference = _optional_text(finding.field)
    policy_rule = _optional_text(finding.policy_rule)
    clause_id = _optional_text(finding.clause_id)

    finding_type = policy_rule or field_reference or "risk_finding"

    title = _humanize_identifier(finding_type)

    expected_value = _optional_text(finding.expected_value)

    actual_value = _optional_text(finding.actual_value)

    category = categorize_finding(
        approval_policy,
        explicit_category=finding.category,
        finding_type=finding_type,
        policy_rule=policy_rule,
        field_reference=field_reference,
        title=title,
    )

    severity = _normalize_triage_severity(finding.risk_tier)

    risk_level = _normalize_triage_risk_level(finding.risk_tier)

    evidence_ids = resolve_evidence_ids(
        evidence_index,
        evidence_ref=_optional_text(finding.evidence_ref),
        clause_id=clause_id,
        page_number=finding.page_number,
    )

    (
        document_name,
        resolved_page_number,
        resolved_bbox,
        text_excerpt,
    ) = _primary_evidence_fields(
        evidence_index,
        evidence_ids,
        fallback_page_number=finding.page_number,
        fallback_bbox=finding.bbox,
    )

    triage_finding_id = _stable_triage_finding_id(
        finding_type=finding_type,
        category=category,
        clause_id=clause_id,
        field_reference=field_reference,
        policy_rule=policy_rule,
        expected_value=expected_value,
        actual_value=actual_value,
    )

    return TriageFinding(
        finding_id=triage_finding_id,
        source_finding_ids=_unique_nonblank_values(
            [
                finding.finding_id,
                finding.source_finding_id,
            ]
        ),
        source_agents=["agent_e"],
        finding_type=finding_type,
        title=title,
        category=category,
        severity=severity,
        risk_level=risk_level,
        score=finding.score,
        confidence=_optional_confidence(getattr(finding, "confidence", None)),
        clause_id=clause_id,
        clause_type=_optional_text(getattr(finding, "clause_type", None)),
        field_reference=field_reference,
        policy_rule=policy_rule,
        expected_value=expected_value,
        actual_value=actual_value,
        recommendation=_optional_text(finding.recommendation),
        additional_recommendations=[],
        open_questions=_optional_string_list(getattr(finding, "open_questions", None)),
        requires_human_review=(
            category == TriageCategory.MANUAL_REVIEW
            or _requires_review_from_severity(severity)
        ),
        evidence_ids=evidence_ids,
        document_name=document_name,
        page_number=resolved_page_number,
        bbox=resolved_bbox,
        text_excerpt=text_excerpt,
    )


def normalize_agent_c_flag(
    flag: str,
    counterparty: NormalizedCounterparty,
    approval_policy: ApprovalPolicy,
) -> TriageFinding:
    """
    Convert one Agent C flag into an Agent H normalized finding.

    Agent C flags do not carry their own finding IDs or evidence
    metadata, so Agent H creates a deterministic finding while
    preserving the flag as the finding type and policy signal.
    """

    finding_type = _normalize_signal(flag)

    if not finding_type:
        raise TriageConsistencyError("Agent C contains a blank counterparty flag.")

    category = categorize_finding(
        approval_policy,
        explicit_category="counterparty",
        finding_type=finding_type,
        policy_rule=finding_type,
        title=_humanize_identifier(finding_type),
    )

    severity = _normalize_triage_severity(counterparty.risk_level)

    risk_level = _normalize_triage_risk_level(counterparty.risk_level)

    expected_value = _optional_text(counterparty.manifest_counterparty_name)

    actual_value = _optional_text(
        counterparty.extracted_counterparty_name
    ) or _optional_text(counterparty.input_counterparty_name)

    finding_id = _stable_triage_finding_id(
        finding_type=finding_type,
        category=category,
        clause_id=None,
        field_reference="counterparty",
        policy_rule=finding_type,
        expected_value=expected_value,
        actual_value=actual_value,
    )

    return TriageFinding(
        finding_id=finding_id,
        source_finding_ids=[],
        source_agents=["agent_c"],
        finding_type=finding_type,
        title=_humanize_identifier(finding_type),
        category=category,
        severity=severity,
        risk_level=risk_level,
        score=_default_score_for_severity(severity),
        confidence=None,
        clause_id=None,
        clause_type=None,
        field_reference="counterparty",
        policy_rule=finding_type,
        expected_value=expected_value,
        actual_value=actual_value,
        recommendation=None,
        additional_recommendations=[],
        open_questions=[],
        requires_human_review=(_requires_review_from_severity(severity)),
        evidence_ids=[],
        document_name=None,
        page_number=None,
        bbox=None,
        text_excerpt=None,
    )


def collect_normalized_findings(
    inputs: TriageInputs,
) -> list[TriageFinding]:
    """
    Collect and normalize findings and flags from Agents C, D, and E.

    An Agent C flag is converted only when no Agent C finding already
    represents the same normalized finding type or policy rule.
    """

    normalized_findings: list[TriageFinding] = []

    represented_agent_c_signals: set[str] = set()

    for finding in inputs.normalized_counterparty.findings:
        normalized_finding = normalize_agent_c_finding(
            finding,
            inputs.approval_policy,
            inputs.evidence_index,
        )

        normalized_findings.append(normalized_finding)

        represented_agent_c_signals.update(
            signal
            for signal in (
                _normalize_signal(normalized_finding.finding_type),
                _normalize_signal(normalized_finding.policy_rule),
            )
            if signal
        )

    for flag in sorted(
        inputs.normalized_counterparty.flags,
        key=_normalize_signal,
    ):
        normalized_flag = _normalize_signal(flag)

        if not normalized_flag:
            raise TriageConsistencyError(
                "Agent C contains a blank " "counterparty flag."
            )

        if normalized_flag in represented_agent_c_signals:
            continue

        normalized_findings.append(
            normalize_agent_c_flag(
                flag,
                inputs.normalized_counterparty,
                inputs.approval_policy,
            )
        )

    for finding in inputs.validation_result.findings:
        normalized_findings.append(
            normalize_agent_d_finding(
                finding,
                inputs.approval_policy,
                inputs.evidence_index,
            )
        )

    for finding in inputs.clause_analysis.findings:
        normalized_findings.append(
            normalize_agent_e_finding(
                finding,
                inputs.approval_policy,
                inputs.evidence_index,
            )
        )

    return normalized_findings


def _normalize_signal(value: object | None) -> str:
    """
    Normalize values for deterministic category comparisons.

    Examples:
        "Payment Terms" -> "payment_terms"
        "legal-review" -> "legal_review"
    """

    raw_value = _enum_or_string_value(value)

    if raw_value is None:
        return ""

    normalized = raw_value.strip().lower()
    normalized = normalized.replace("-", "_")
    normalized = normalized.replace(" ", "_")

    while "__" in normalized:
        normalized = normalized.replace("__", "_")

    return normalized


def _category_from_department(
    department: str,
) -> TriageCategory | None:
    """
    Convert a department-routing name into an Agent H category.
    """

    normalized_department = _normalize_signal(department)

    return CATEGORY_ALIASES.get(normalized_department)


def _category_from_policy_triggers(
    approval_policy: ApprovalPolicy,
    signals: set[str],
) -> TriageCategory | None:
    """
    Match finding signals against department_routing triggers.

    Exact normalized trigger matching is used so that unrelated
    findings are not assigned to a department accidentally.
    """

    matched_categories: set[TriageCategory] = set()

    for department, routing in sorted(approval_policy.department_routing.items()):
        category = _category_from_department(department)

        if category is None:
            continue

        normalized_triggers = {
            _normalize_signal(trigger) for trigger in routing.triggers
        }

        if signals & normalized_triggers:
            matched_categories.add(category)

    if len(matched_categories) > 1:
        categories = ", ".join(
            sorted(category.value for category in matched_categories)
        )

        raise TriageConsistencyError(
            "Finding matches multiple departmental categories: " f"{categories}."
        )

    if matched_categories:
        return next(iter(matched_categories))

    return None


def _category_from_fallback_signals(
    signals: set[str],
) -> TriageCategory | None:
    """
    Categorize findings whose upstream format has no explicit category
    and whose finding type is not present in department_routing.
    """

    for category, known_signals in FALLBACK_CATEGORY_SIGNALS.items():
        if signals & known_signals:
            return category

    return None


def categorize_finding(
    approval_policy: ApprovalPolicy,
    *,
    explicit_category: object | None = None,
    finding_type: object | None = None,
    policy_rule: object | None = None,
    field_reference: object | None = None,
    title: object | None = None,
) -> TriageCategory:
    """
    Determine the actionable category of one upstream finding.

    Resolution order:

    1. explicit valid category from the upstream agent;
    2. department_routing trigger from approval_policy.yaml;
    3. stable fallback signal mapping;
    4. operational category when no other category applies.
    """

    normalized_explicit_category = _normalize_signal(explicit_category)

    explicit_result = CATEGORY_ALIASES.get(normalized_explicit_category)

    if explicit_result is not None and explicit_result != TriageCategory.GENERAL:
        return explicit_result

    signals = {
        normalized
        for normalized in (
            _normalize_signal(finding_type),
            _normalize_signal(policy_rule),
            _normalize_signal(field_reference),
            _normalize_signal(title),
        )
        if normalized
    }

    policy_category = _category_from_policy_triggers(
        approval_policy,
        signals,
    )

    if policy_category is not None:
        return policy_category

    fallback_category = _category_from_fallback_signals(signals)

    if fallback_category is not None:
        return fallback_category

    if explicit_result == TriageCategory.GENERAL:
        return TriageCategory.GENERAL

    return TriageCategory.OPERATIONAL


def parse_evidence_reference(
    evidence_ref: str | None,
) -> tuple[str | None, int | None, str | None]:
    """
    Parse an upstream evidence pointer.

    Example input:
        contract.pdf#page=3&clause=payment_terms_001

    Example output:
        ("contract.pdf", 3, "payment_terms_001")
    """

    if evidence_ref is None:
        return None, None, None

    cleaned_reference = evidence_ref.strip()

    if not cleaned_reference:
        return None, None, None

    filename_part = cleaned_reference.split(
        "#",
        maxsplit=1,
    )[0].strip()

    filename = filename_part or None

    page_match = PAGE_REFERENCE_PATTERN.search(cleaned_reference)

    clause_match = CLAUSE_REFERENCE_PATTERN.search(cleaned_reference)

    page_number = int(page_match.group(1)) if page_match is not None else None

    clause_id = clause_match.group(1).strip() if clause_match is not None else None

    return filename, page_number, clause_id


def build_evidence_lookup(
    evidence_index: EvidenceIndex,
) -> dict[str, EvidenceItem]:
    """
    Build a deterministic evidence lookup and reject duplicate IDs.
    """

    lookup: dict[str, EvidenceItem] = {}

    for item in evidence_index.evidence_items:
        if item.evidence_id in lookup:
            raise TriageEvidenceError(
                "Duplicate evidence ID found in "
                f"evidence_index.json: {item.evidence_id!r}."
            )

        lookup[item.evidence_id] = item

    return lookup


def resolve_evidence_ids(
    evidence_index: EvidenceIndex,
    *,
    explicit_evidence_ids: list[str] | None = None,
    evidence_ref: str | None = None,
    clause_id: str | None = None,
    page_number: int | None = None,
) -> list[str]:
    """
    Resolve and validate evidence used by one upstream finding.

    Agent C generally supplies explicit evidence IDs.

    Agents D and E generally supply evidence_ref values such as:
        contract.pdf#page=3&clause=payment_terms_001

    The result always contains valid evidence IDs from
    evidence_index.json in deterministic sorted order.
    """

    evidence_lookup = build_evidence_lookup(evidence_index)

    resolved_ids: set[str] = set()

    # Validate direct evidence IDs supplied by Agent C.
    for raw_evidence_id in explicit_evidence_ids or []:
        evidence_id = raw_evidence_id.strip()

        if not evidence_id:
            raise TriageEvidenceError("A finding contains a blank evidence ID.")

        if evidence_id not in evidence_lookup:
            raise TriageEvidenceError(
                "Finding references an unknown evidence ID: " f"{evidence_id!r}."
            )

        resolved_ids.add(evidence_id)

    # Some upstream artifacts may put the evidence ID directly in
    # evidence_ref instead of using a page/clause pointer.
    cleaned_reference = evidence_ref.strip() if evidence_ref is not None else None

    if cleaned_reference and cleaned_reference in evidence_lookup:
        resolved_ids.add(cleaned_reference)

    parsed_filename, parsed_page, parsed_clause = parse_evidence_reference(evidence_ref)

    target_clause_id = clause_id or parsed_clause
    target_page_number = page_number or parsed_page

    # Only perform location-based matching when the finding provides
    # a clause or page. A filename by itself would be too broad and
    # could accidentally attach every item in a document.
    has_location_selector = (
        target_clause_id is not None or target_page_number is not None
    )

    if has_location_selector:
        for item in evidence_index.evidence_items:
            filename_matches = (
                parsed_filename is None or item.filename == parsed_filename
            )

            clause_matches = (
                target_clause_id is None or item.clause_id == target_clause_id
            )

            page_matches = target_page_number is None or item.page == target_page_number

            if filename_matches and clause_matches and page_matches:
                resolved_ids.add(item.evidence_id)

    # If an upstream finding explicitly referenced evidence but
    # nothing in the index matched, the input is invalid.
    reference_was_provided = bool(cleaned_reference or explicit_evidence_ids)

    if reference_was_provided and not resolved_ids:
        raise TriageEvidenceError(
            "Finding evidence could not be resolved against "
            "evidence_index.json. "
            f"evidence_ref={evidence_ref!r}, "
            f"clause_id={clause_id!r}, "
            f"page_number={page_number!r}."
        )

    return sorted(resolved_ids)


def get_primary_evidence_item(
    evidence_index: EvidenceIndex,
    evidence_ids: list[str],
) -> EvidenceItem | None:
    """
    Return one deterministic primary evidence item.

    All evidence IDs remain preserved separately. This helper only
    selects the first location used to populate the finding's main
    document, page, bbox, and text excerpt fields.
    """

    evidence_lookup = build_evidence_lookup(evidence_index)

    matching_items = [
        evidence_lookup[evidence_id]
        for evidence_id in sorted(evidence_ids)
        if evidence_id in evidence_lookup
    ]

    if not matching_items:
        return None

    return matching_items[0]


def validate_triage_input_consistency(
    run_dir: str | Path,
    inputs: TriageInputs,
) -> None:
    """
    Validate identifiers and upstream execution state across Agent H inputs.

    Agent H cannot safely consolidate findings when artifacts belong to
    different runs or contracts.
    """

    run_path = Path(run_dir)

    # context_packet.run_id should identify the current run directory.
    expected_run_id = run_path.name
    actual_run_id = inputs.context_packet.run_id

    if actual_run_id != expected_run_id:
        raise TriageConsistencyError(
            "Run ID mismatch across Agent H inputs: "
            f"run directory is {expected_run_id!r}, but "
            f"context_packet.json contains {actual_run_id!r}."
        )

    authoritative_contract_id = inputs.context_packet.contract_id

    contract_ids = {
        "context_packet.json": authoritative_contract_id,
        "validation_result.json": (inputs.validation_result.contract_id),
        "clause_analysis.json": (inputs.clause_analysis.contract_id),
    }

    mismatched_contract_ids = {
        artifact_name: contract_id
        for artifact_name, contract_id in contract_ids.items()
        if contract_id != authoritative_contract_id
    }

    if mismatched_contract_ids:
        details = ", ".join(
            f"{artifact_name}={contract_id!r}"
            for artifact_name, contract_id in sorted(mismatched_contract_ids.items())
        )

        raise TriageConsistencyError(
            "Contract ID mismatch across Agent H inputs. "
            f"Expected {authoritative_contract_id!r}; found {details}."
        )

    # Agent D explicitly exposes a failed state. Agent H must not
    # continue when validation did not complete successfully.
    validation_status = inputs.validation_result.validation_status.value

    if validation_status == "failed":
        raise TriageConsistencyError(
            "Agent H cannot continue because "
            "validation_result.json reports status 'failed'."
        )


def load_triage_inputs(
    run_dir: str | Path,
) -> TriageInputs:
    """
    Load and validate all inputs required by Agent H.

    This function currently performs:

    1. run-directory validation;
    2. required-file validation;
    3. JSON parsing;
    4. Pydantic schema validation;
    5. approval-policy YAML validation.

    Cross-artifact identifier checks are added in the next step.
    """

    resolved_paths = _resolve_required_paths(run_dir)

    context_packet = _load_json_model(
        resolved_paths["context_packet"],
        ContextPacket,
    )

    normalized_counterparty = _load_json_model(
        resolved_paths["normalized_counterparty"],
        NormalizedCounterparty,
    )

    validation_result = _load_json_model(
        resolved_paths["validation_result"],
        ValidationResult,
    )

    clause_analysis = _load_json_model(
        resolved_paths["clause_analysis"],
        ClauseAnalysis,
    )

    evidence_index = _load_json_model(
        resolved_paths["evidence_index"],
        EvidenceIndex,
    )

    try:
        approval_policy = ApprovalPolicy.from_yaml_file(
            resolved_paths["approval_policy"]
        )
    except (
        OSError,
        UnicodeError,
        ValidationError,
        ValueError,
    ) as exc:
        raise ApprovalPolicyValidationError(
            "Agent H approval policy failed validation: "
            f"{resolved_paths['approval_policy']}\n{exc}"
        ) from exc

    inputs = TriageInputs(
        context_packet=context_packet,
        normalized_counterparty=normalized_counterparty,
        validation_result=validation_result,
        clause_analysis=clause_analysis,
        evidence_index=evidence_index,
        approval_policy=approval_policy,
    )

    validate_triage_input_consistency(
        run_dir=run_dir,
        inputs=inputs,
    )

    return inputs
