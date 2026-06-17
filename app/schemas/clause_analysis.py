from enum import Enum
from pathlib import Path
from typing import Annotated

from pydantic import Field, StrictInt, StrictStr

from app.schemas.extracted_contract import BoundingBox, StrictSchemaModel

RiskScore = Annotated[StrictInt, Field(ge=0, le=100)]


class RiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskCategory(str, Enum):
    LEGAL = "legal"
    FINANCE = "finance"
    COMPLIANCE = "compliance"
    MANUAL_REVIEW = "manual_review"
    COUNTERPARTY = "counterparty"


class ScoredRiskFinding(StrictSchemaModel):
    finding_id: StrictStr = Field(..., min_length=1)
    source_finding_id: StrictStr = Field(..., min_length=1)
    field: StrictStr = Field(..., min_length=1)

    category: RiskCategory
    risk_tier: RiskTier
    score: RiskScore

    policy_rule: StrictStr = Field(..., min_length=1)
    expected_value: StrictStr = Field(..., min_length=1)
    actual_value: StrictStr = Field(..., min_length=1)
    recommendation: StrictStr = Field(..., min_length=1)
    evidence_ref: StrictStr = Field(..., min_length=1)
    clause_id: StrictStr | None = None
    page_number: StrictInt | None = None
    bbox: BoundingBox | None = None
    tolerance_band: StrictStr | None = None


class ClauseAnalysis(StrictSchemaModel):
    schema_version: StrictStr = "1.0"
    contract_id: StrictStr = Field(..., min_length=1)

    total_score: RiskScore
    risk_tier: RiskTier
    jurisdictions_evaluated: list[StrictStr] = Field(default_factory=list)

    findings: list[ScoredRiskFinding] = Field(default_factory=list)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ClauseAnalysis":
        file_path = Path(path)
        return cls.model_validate_json(file_path.read_text(encoding="utf-8"))

    def to_json_file(self, path: str | Path) -> None:
        file_path = Path(path)
        file_path.write_text(
            self.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )