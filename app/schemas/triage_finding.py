from enum import Enum
from typing import Annotated

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr

from app.schemas.extracted_contract import BoundingBox, StrictSchemaModel


class TriageCategory(str, Enum):
    FINANCE = "finance"
    LEGAL = "legal"
    COMPLIANCE = "compliance"
    COUNTERPARTY = "counterparty"
    PROCUREMENT = "procurement"
    OPERATIONAL = "operational"
    MANUAL_REVIEW = "manual_review"
    BLOCKING = "blocking"
    GENERAL = "general"


class TriageSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TriageRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TriageFinding(StrictSchemaModel):
    """
    Agent H's normalized finding.

    Agent H converts findings from Agents C, D, and E into this format
    without changing the schemas owned by those agents.
    """

    # Deterministic identifier created by Agent H.
    finding_id: StrictStr = Field(..., min_length=1)

    # Original finding identifiers from upstream agents.
    source_finding_ids: list[StrictStr] = Field(default_factory=list)

    # All agents that reported or contributed to the issue.
    source_agents: list[StrictStr] = Field(default_factory=list)

    finding_type: StrictStr = Field(..., min_length=1)
    title: StrictStr = Field(..., min_length=1)

    category: TriageCategory
    severity: TriageSeverity
    risk_level: TriageRiskLevel

    score: Annotated[
        StrictInt,
        Field(ge=0, le=100),
    ] = 0

    confidence: (
        Annotated[
            StrictFloat,
            Field(ge=0.0, le=1.0),
        ]
        | None
    ) = None

    clause_id: StrictStr | None = None
    clause_type: StrictStr | None = None
    field_reference: StrictStr | None = None

    policy_rule: StrictStr | None = None
    expected_value: StrictStr | None = None
    actual_value: StrictStr | None = None

    recommendation: StrictStr | None = None
    additional_recommendations: list[StrictStr] = Field(default_factory=list)
    open_questions: list[StrictStr] = Field(default_factory=list)

    requires_human_review: StrictBool = False

    evidence_ids: list[StrictStr] = Field(default_factory=list)

    document_name: StrictStr | None = None

    page_number: (
        Annotated[
            StrictInt,
            Field(ge=1),
        ]
        | None
    ) = None

    bbox: BoundingBox | None = None
    text_excerpt: StrictStr | None = None


class TriagePreparationResult(StrictSchemaModel):
    """
    Result of DZ-02.1.

    This is not approval_packet.json. It is the validated and
    consolidated input that later routing subtasks will consume.
    """

    run_id: StrictStr = Field(..., min_length=1)
    contract_id: StrictStr = Field(..., min_length=1)

    findings_received: StrictInt = Field(ge=0)
    duplicates_removed: StrictInt = Field(ge=0)

    findings_by_source: dict[StrictStr, StrictInt] = Field(default_factory=dict)

    consolidated_findings: list[TriageFinding] = Field(default_factory=list)
