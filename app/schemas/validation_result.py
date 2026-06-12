from enum import Enum
from pathlib import Path
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
)

# This file defines the official output contract for Agent D.
# Agent D validates and normalizes extracted contract data before risk scoring.
#
# The output of this schema is validation_result.json.
# Agent E can use this artifact directly for rules engine and risk scoring.

PositivePageNumber = Annotated[StrictInt, Field(ge=1)]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]

# Dates must be normalized to ISO format: YYYY-MM-DD.
IsoDateString = Annotated[StrictStr, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]


class StrictSchemaModel(BaseModel):
    # Reject unknown fields so the pipeline stays deterministic.
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class ValidationStatus(str, Enum):
    # completed = validation finished with no important findings.
    COMPLETED = "completed"

    # completed_with_findings = validation finished but found issues or notes.
    COMPLETED_WITH_FINDINGS = "completed_with_findings"

    # failed = validation could not complete.
    FAILED = "failed"


class Severity(str, Enum):
    # Severity levels used by validation findings.
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NormalizedFields(StrictSchemaModel):
    # Stores cleaned contract fields in a format that is easy for rules engine.
    # Example: instead of reading "ninety days" from text, Agent D stores 90.

    payment_terms_days: NonNegativeInt | None = None
    governing_law: StrictStr | None = None
    effective_date: IsoDateString | None = None

    # Boolean flags that downstream risk logic can check directly.
    liability_cap_present: StrictBool | None = None
    gdpr_clause_present: StrictBool | None = None
    auto_renewal_present: StrictBool | None = None

    # Optional opt-out window for auto-renewal clauses.
    opt_out_window_days: NonNegativeInt | None = None

    # Counterparty name can come from manifest, extracted text, or normalized data.
    counterparty_name: StrictStr | None = None

    # True when extracted clauses have low confidence and need manual review.
    has_low_confidence_extraction: StrictBool = False


class ValidationFinding(StrictSchemaModel):
    # Represents one validation finding.
    # Findings explain what was detected, how severe it is, and where the evidence is.

    finding_id: StrictStr = Field(..., min_length=1)
    field: StrictStr = Field(..., min_length=1)
    severity: Severity
    message: StrictStr = Field(..., min_length=1)
    evidence_ref: StrictStr = Field(..., min_length=1)

    # Optional reference back to the clause that caused the finding.
    clause_id: StrictStr | None = None
    page_number: PositivePageNumber | None = None

    @field_validator("finding_id", "field", "message", "evidence_ref")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        # Prevent empty validation messages or missing evidence references.
        if not value.strip():
            raise ValueError("Value cannot be blank.")
        return value


class ValidationResult(StrictSchemaModel):
    # Top-level schema for validation_result.json.
    # This artifact is consumed by downstream agents, especially Agent E.

    schema_version: StrictStr = "1.0"
    contract_id: StrictStr = Field(..., min_length=1)
    validation_status: ValidationStatus
    normalized_fields: NormalizedFields
    findings: list[ValidationFinding] = Field(default_factory=list)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ValidationResult":
        # Helper method used by tests or agents to load and validate JSON.
        file_path = Path(path)
        return cls.model_validate_json(file_path.read_text(encoding="utf-8"))

    def to_json_file(self, path: str | Path) -> None:
        # Helper method used by agents to save validated JSON artifacts.
        file_path = Path(path)
        file_path.write_text(
            self.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
