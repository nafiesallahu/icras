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

PositivePageNumber = Annotated[StrictInt, Field(ge=1)]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
IsoDateString = Annotated[StrictStr, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]


class StrictSchemaModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class ValidationStatus(str, Enum):
    COMPLETED = "completed"
    COMPLETED_WITH_FINDINGS = "completed_with_findings"
    FAILED = "failed"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NormalizedFields(StrictSchemaModel):
    payment_terms_days: NonNegativeInt | None = None
    governing_law: StrictStr | None = None
    effective_date: IsoDateString | None = None
    liability_cap_present: StrictBool | None = None
    gdpr_clause_present: StrictBool | None = None
    auto_renewal_present: StrictBool | None = None
    opt_out_window_days: NonNegativeInt | None = None
    counterparty_name: StrictStr | None = None
    has_low_confidence_extraction: StrictBool = False


class ValidationFinding(StrictSchemaModel):
    finding_id: StrictStr = Field(..., min_length=1)
    field: StrictStr = Field(..., min_length=1)
    severity: Severity
    message: StrictStr = Field(..., min_length=1)
    evidence_ref: StrictStr = Field(..., min_length=1)
    clause_id: StrictStr | None = None
    page_number: PositivePageNumber | None = None

    @field_validator("finding_id", "field", "message", "evidence_ref")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value cannot be blank.")
        return value


class ValidationResult(StrictSchemaModel):
    schema_version: StrictStr = "1.0"
    contract_id: StrictStr = Field(..., min_length=1)
    validation_status: ValidationStatus
    normalized_fields: NormalizedFields
    findings: list[ValidationFinding] = Field(default_factory=list)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ValidationResult":
        file_path = Path(path)
        return cls.model_validate_json(file_path.read_text(encoding="utf-8"))

    def to_json_file(self, path: str | Path) -> None:
        file_path = Path(path)
        file_path.write_text(
            self.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
