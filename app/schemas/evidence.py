"""Evidence index schema for the ICRAS pipeline.

The *Universal Evidence Index* (``evidence_index.json``) is initialized by the
intake agent (Agent A) with one document entry per admitted bundle file and an
empty ``evidence_items`` list. Agent B appends one clause-level evidence item
per extracted clause location so every downstream finding can be traced back to
an exact page (and, when available, bounding box) of the source contract.

This module defines the Pydantic models used to validate the evidence index
before Agent B writes it back to disk:

* :class:`EvidenceItem` — a single clause-level evidence record.
* :class:`EvidenceIndex` — the full index document (documents + evidence items).

``EvidenceIndex.documents`` is intentionally left loosely typed (``list[dict]``)
so the index round-trips the document entries written by Agent A without
coupling Agent B to that exact shape.
"""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator

from app.schemas.extracted_contract import (
    BoundingBox,
    PositivePageNumber,
    StrictSchemaModel,
)


class EvidenceItem(StrictSchemaModel):
    """A single clause-level evidence record.

    Each item pins one clause location to a concrete page (and optional bounding
    box) of a source document, providing audit traceability for downstream
    agents.
    """

    evidence_id: StrictStr = Field(..., min_length=1)
    document_id: StrictStr = Field(..., min_length=1)
    filename: StrictStr = Field(..., min_length=1)
    page: PositivePageNumber
    clause_id: StrictStr = Field(..., min_length=1)

    # bbox is optional because not every clause location has computable PDF
    # coordinates (for example, synthetic fallback locations).
    bbox: BoundingBox | None = None

    text_excerpt: StrictStr = Field(..., min_length=1)

    @field_validator(
        "evidence_id",
        "document_id",
        "filename",
        "clause_id",
        "text_excerpt",
    )
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Value cannot be blank.")
        return value


class EvidenceIndex(BaseModel):
    """The full ``evidence_index.json`` document.

    ``documents`` mirrors the document entries written by Agent A and is kept
    loosely typed so the index round-trips cleanly. ``evidence_items`` is the
    clause-level evidence appended by Agent B.
    """

    # Allow the document entries written by other agents to round-trip without
    # being rejected, while still validating the evidence items strictly.
    model_config = ConfigDict(extra="forbid")

    documents: list[dict] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "EvidenceIndex":
        """Load and validate an evidence index from a JSON file."""
        file_path = Path(path)
        return cls.model_validate_json(file_path.read_text(encoding="utf-8"))

    def to_json_file(self, path: str | Path) -> None:
        """Serialize the validated evidence index to a formatted JSON file."""
        file_path = Path(path)
        file_path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
