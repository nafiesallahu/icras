from enum import Enum
from pydantic import Field, StrictStr, StrictInt
from typing import Annotated
from app.schemas.extracted_contract import StrictSchemaModel, BoundingBox

class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class UnifiedFinding(StrictSchemaModel):
    finding_id: StrictStr = Field(..., min_length=1)
    field: StrictStr = Field(..., min_length=1)
    severity: Severity
    message: StrictStr = Field(..., min_length=1)
    evidence_ref: StrictStr = Field(..., min_length=1)
    clause_id: StrictStr | None = None
    page_number: Annotated[StrictInt, Field(ge=1)] | None = None
    bbox: BoundingBox | None = None
    
    policy_rule: StrictStr | None = None
    expected_value: StrictStr | None = None
    actual_value: StrictStr | None = None
    recommendation: StrictStr | None = None