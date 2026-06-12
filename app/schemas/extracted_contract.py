from enum import Enum
from pathlib import Path
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
)

PrimitiveValue = StrictStr | StrictInt | StrictFloat | StrictBool | None
ConfidenceScore = Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
PositivePageNumber = Annotated[StrictInt, Field(ge=1)]
NonNegativeCoordinate = Annotated[StrictFloat, Field(ge=0.0)]


class StrictSchemaModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class ExtractionMode(str, Enum):
    PDF_PARSER = "pdf_parser"
    LLM_STRUCTURED = "llm_structured"
    SYNTHETIC_FALLBACK = "synthetic_fallback"


class ClauseType(str, Enum):
    PAYMENT_TERMS = "payment_terms"
    LIABILITY_CAP = "liability_cap"
    GOVERNING_LAW = "governing_law"
    TERMINATION = "termination"
    AUTO_RENEWAL = "auto_renewal"
    GDPR_DATA_PROCESSING = "gdpr_data_processing"
    CONFIDENTIALITY = "confidentiality"
    SIGNATURE_BLOCK = "signature_block"
    EFFECTIVE_DATE = "effective_date"
    COUNTERPARTY = "counterparty"
    OTHER = "other"


class BoundingBox(StrictSchemaModel):
    x0: NonNegativeCoordinate
    top: NonNegativeCoordinate
    x1: NonNegativeCoordinate
    bottom: NonNegativeCoordinate


class ContractClause(StrictSchemaModel):
    clause_id: StrictStr = Field(..., min_length=1)
    clause_type: ClauseType
    text: StrictStr = Field(..., min_length=1)
    page_number: PositivePageNumber
    confidence_score: ConfidenceScore
    evidence_ref: StrictStr = Field(..., min_length=1)
    structured_fields: dict[str, PrimitiveValue] = Field(default_factory=dict)
    bbox: BoundingBox | None = None

    @field_validator("clause_id", "text", "evidence_ref")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value cannot be blank.")
        return value


class ExtractedContract(StrictSchemaModel):
    schema_version: StrictStr = "1.0"
    contract_id: StrictStr = Field(..., min_length=1)
    source_file: StrictStr = Field(..., min_length=1)
    extraction_mode: ExtractionMode
    overall_confidence_score: ConfidenceScore
    clauses: list[ContractClause] = Field(..., min_length=1)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ExtractedContract":
        file_path = Path(path)
        return cls.model_validate_json(file_path.read_text(encoding="utf-8"))

    def to_json_file(self, path: str | Path) -> None:
        file_path = Path(path)
        file_path.write_text(
            self.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
