from enum import Enum
from typing import Annotated

from pydantic import Field, StrictBool, StrictInt, StrictStr

from app.schemas.extracted_contract import BoundingBox, StrictSchemaModel


class Severity(str, Enum):
    """
    Describes how severe a finding is.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FindingRiskLevel(str, Enum):
    """
    Risk levels used by downstream risk scoring and triage agents.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class UnifiedFinding(StrictSchemaModel):
    """
    Shared finding structure used by Agents C, D, E, and H.

    The new fields satisfy the ICRAS unified finding requirements.

    Legacy fields remain optional so existing Agent B/D code does not
    break immediately while the team migrates to the unified structure.
    """

    # Unique and deterministic finding identifier.
    finding_id: StrictStr = Field(
        ...,
        min_length=1,
    )

    # Agent that created the finding.
    #
    # Agent C must always use "agent_c".
    source_agent: StrictStr = Field(
        default="unknown",
        min_length=1,
    )

    # Business category such as counterparty, legal, finance,
    # compliance, or manual_review.
    category: StrictStr = Field(
        default="general",
        min_length=1,
    )

    # Severity of the finding.
    severity: Severity

    # Risk level consumed by downstream routing agents.
    risk_level: FindingRiskLevel = FindingRiskLevel.MEDIUM

    # Deterministic risk score from 0 to 100.
    score: Annotated[
        StrictInt,
        Field(ge=0, le=100),
    ] = 0

    # Machine-readable finding type.
    #
    # Examples:
    # - counterparty_mismatch
    # - unknown_counterparty
    # - high_risk_counterparty
    finding_type: StrictStr = Field(
        default="unspecified",
        min_length=1,
    )

    # Clause reference is nullable because counterparty findings
    # are generally not connected to one contract clause.
    clause_id: StrictStr | None = None

    # Deterministic policy or rule that generated the finding.
    policy_rule: StrictStr | None = None

    # Expected business value.
    expected: StrictStr | None = None

    # Actual value found by Agent C.
    actual: StrictStr | None = None

    # Recommended human or workflow action.
    recommendation: StrictStr | None = None

    # Evidence-index identifiers supporting the finding.
    evidence_ids: list[StrictStr] = Field(
        default_factory=list,
    )

    # Indicates whether automation may safely continue without review.
    requires_human_review: StrictBool = False

    # -----------------------------------------------------------------
    # Legacy compatibility fields
    # -----------------------------------------------------------------
    #
    # These fields were part of the earlier project schema.
    # They remain optional so existing agents and tests continue working.

    field: StrictStr | None = None
    message: StrictStr | None = None
    evidence_ref: StrictStr | None = None

    page_number: Annotated[StrictInt, Field(ge=1)] | None = None

    bbox: BoundingBox | None = None

    expected_value: StrictStr | None = None
    actual_value: StrictStr | None = None
