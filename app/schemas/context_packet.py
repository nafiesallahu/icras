from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import (
    Field,
    StrictStr,
    field_validator,
)

from app.schemas.extracted_contract import StrictSchemaModel

# This file defines the official input contract for the ICRAS intake pipeline.
# The Intake Agent is the entry gate for raw contract packages. It produces a
# ContextPacket that downstream agents rely on for run tracking and lineage.
#
# The artifact produced from this schema is context_packet.json, written into
# an isolated per-run directory under runs/.


class DocumentType(str, Enum):
    """Enumerates the categories of documents the intake gate accepts.

    The enum extends ``str`` so values serialize cleanly to JSON and compare
    directly against raw strings, matching the convention used by the other
    ICRAS schema enums.
    """

    # A master/services agreement bundle.
    SERVICES_AGREEMENT = "SERVICES_AGREEMENT"

    # A non-disclosure agreement bundle.
    NDA = "NDA"

    # An amendment to an existing agreement.
    AMENDMENT = "AMENDMENT"


class ContextPacket(StrictSchemaModel):
    """Strict, immutable record describing a single intake run.

    ``StrictSchemaModel`` already enforces ``frozen=True`` and
    ``extra="forbid"`` (plus whitespace stripping), so this model rejects
    unexpected fields and cannot be mutated after construction. This guarantees
    that every downstream agent receives a predictable, tamper-proof packet.

    Fields:
        contract_id: Non-empty session/contract tracking identifier.
        bundle_path: Non-empty string pointing at the source bundle directory.
        received_timestamp: Absolute capture time in strict ISO 8601 format.
        document_type: One of the accepted :class:`DocumentType` values.
    """

    contract_id: StrictStr = Field(..., min_length=1)
    bundle_path: StrictStr = Field(..., min_length=1)
    received_timestamp: StrictStr = Field(..., min_length=1)
    document_type: DocumentType

    @field_validator("contract_id", "bundle_path")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        """Reject empty or whitespace-only identifiers and paths."""
        if not value.strip():
            raise ValueError("Value cannot be blank.")
        return value

    @field_validator("received_timestamp")
    @classmethod
    def must_be_iso_8601(cls, value: str) -> str:
        """Ensure the timestamp parses as strict ISO 8601.

        ``datetime.fromisoformat`` accepts the full ISO 8601 representation
        produced by ``datetime.isoformat`` (including timezone offsets). Any
        value it cannot parse is rejected so malformed timestamps never reach
        downstream agents.
        """
        try:
            datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(
                "received_timestamp must be a valid ISO 8601 timestamp."
            ) from exc
        return value

    @classmethod
    def from_json_file(cls, path: str | Path) -> "ContextPacket":
        """Load and validate a context packet from a JSON file."""
        file_path = Path(path)
        return cls.model_validate_json(file_path.read_text(encoding="utf-8"))

    def to_json_file(self, path: str | Path) -> None:
        """Serialize the validated packet to a formatted JSON file."""
        file_path = Path(path)
        file_path.write_text(
            self.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
