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

# This file defines the official output contract for Agent B.
# Agent B is responsible for clause extraction from PDF, LLM structured output,
# or synthetic fallback.
#
# The goal of this schema is to make sure every downstream agent receives
# the same predictable JSON structure: extracted_contract.json.


# Only strict primitive values are allowed inside structured_fields.
# This prevents nested unpredictable objects from being passed downstream.
PrimitiveValue = StrictStr | StrictInt | StrictFloat | StrictBool | None

# Confidence must always be between 0 and 1.
ConfidenceScore = Annotated[StrictFloat, Field(ge=0.0, le=1.0)]

# Page numbers start from 1 because they should match human-readable PDF pages.
PositivePageNumber = Annotated[StrictInt, Field(ge=1)]

# Bounding box coordinates cannot be negative.
NonNegativeCoordinate = Annotated[StrictFloat, Field(ge=0.0)]


class StrictSchemaModel(BaseModel):
    # extra="forbid" means unexpected fields are rejected.
    # This is important because all agents must follow the same contract.
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class ExtractionMode(str, Enum):
    # Real PDF parser extraction.
    PDF_PARSER = "pdf_parser"

    # LLM is used only for structured extraction, not final risk decisions.
    LLM_STRUCTURED = "llm_structured"

    # Used when real extraction is not ready or PDF parsing fails.
    SYNTHETIC_FALLBACK = "synthetic_fallback"


class ClauseType(str, Enum):
    # These are the clause categories Agent B should detect.
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
    # Optional PDF location metadata.
    # This can be used later for evidence tracking or highlighting text in the PDF.
    x0: NonNegativeCoordinate
    top: NonNegativeCoordinate
    x1: NonNegativeCoordinate
    bottom: NonNegativeCoordinate


class ContractClause(StrictSchemaModel):
    # Represents one extracted clause from the contract.
    # Each clause must be traceable through page_number and evidence_ref.

    clause_id: StrictStr = Field(..., min_length=1)
    clause_type: ClauseType
    text: StrictStr = Field(..., min_length=1)
    page_number: PositivePageNumber
    confidence_score: ConfidenceScore
    evidence_ref: StrictStr = Field(..., min_length=1)

    # structured_fields stores normalized primitive values extracted from clause text.
    # Example: {"payment_terms_days": 90}
    structured_fields: dict[str, PrimitiveValue] = Field(default_factory=dict)

    # bbox is optional because synthetic fallback may not have PDF coordinates.
    bbox: BoundingBox | None = None

    @field_validator("clause_id", "text", "evidence_ref")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        # Prevent empty strings like "" or "   ".
        if not value.strip():
            raise ValueError("Value cannot be blank.")
        return value


class ExtractedContract(StrictSchemaModel):
    # Top-level schema for extracted_contract.json.
    # This is the artifact Agent B must produce and downstream agents can consume.

    schema_version: StrictStr = "1.0"
    contract_id: StrictStr = Field(..., min_length=1)
    source_file: StrictStr = Field(..., min_length=1)
    extraction_mode: ExtractionMode
    overall_confidence_score: ConfidenceScore
    clauses: list[ContractClause] = Field(..., min_length=1)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ExtractedContract":
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
