from datetime import datetime
from enum import Enum
from pathlib import Path

from pydantic import (
    Field,
    StrictBool,
    StrictStr,
    field_validator,
)

from app.schemas.extracted_contract import StrictSchemaModel

# This file defines the official input contract for the ICRAS intake pipeline.
# The Intake Agent (Agent A) is the entry gate for raw contract bundles. It
# produces a ContextPacket that downstream agents rely on for run tracking,
# lineage, document classification, and the initial (intake-level) risk signal.
#
# The artifact produced from this schema is context_packet.json, written into
# an isolated per-run directory under runs/.


class DocumentType(str, Enum):
    """High-level classification of the primary contract in a bundle.

    The enum extends ``str`` so values serialize cleanly to JSON and compare
    directly against raw strings, matching the convention used by the other
    ICRAS schema enums.

    This is distinct from the per-file classification stored inside
    :attr:`ContextPacket.documents` (``primary_contract``, ``manifest``, ...).
    ``DocumentType`` describes the *kind of agreement* the bundle represents.
    """

    # A master/services agreement bundle.
    SERVICES_AGREEMENT = "SERVICES_AGREEMENT"

    # A non-disclosure agreement bundle.
    NDA = "NDA"

    # An amendment to an existing agreement.
    AMENDMENT = "AMENDMENT"

    # Any agreement type that does not match a known category. Intake never
    # fails on an unrecognized contract_type; it falls back to OTHER so the run
    # can proceed and downstream agents can decide what to do.
    OTHER = "OTHER"

    # Maps free-form manifest ``contract_type`` strings onto enum members.
    @classmethod
    def from_contract_type(cls, contract_type: str | None) -> "DocumentType":
        """Resolve a manifest ``contract_type`` string into a member.

        Args:
            contract_type: Free-form contract type from the manifest (for
                example ``"services_agreement"``). May be ``None`` or empty.

        Returns:
            The best-matching :class:`DocumentType`, defaulting to
            :attr:`DocumentType.OTHER` for unknown or missing values.
        """
        if not contract_type:
            return cls.OTHER

        normalized = contract_type.strip().lower()
        mapping = {
            "services_agreement": cls.SERVICES_AGREEMENT,
            "master_services_agreement": cls.SERVICES_AGREEMENT,
            "msa": cls.SERVICES_AGREEMENT,
            "supply_agreement": cls.SERVICES_AGREEMENT,
            "subscription_agreement": cls.SERVICES_AGREEMENT,
            "nda": cls.NDA,
            "mutual_nda": cls.NDA,
            "non_disclosure_agreement": cls.NDA,
            "mutual_non_disclosure_agreement": cls.NDA,
            "amendment": cls.AMENDMENT,
        }
        return mapping.get(normalized, cls.OTHER)


class ContextPacket(StrictSchemaModel):
    """Strict, immutable record describing a single intake run.

    ``StrictSchemaModel`` already enforces ``frozen=True`` and
    ``extra="forbid"`` (plus whitespace stripping), so this model rejects
    unexpected fields and cannot be mutated after construction. This guarantees
    that every downstream agent receives a predictable, tamper-proof packet.

    Fields:
        run_id: Unique identifier for this intake run (also the run directory
            name under ``runs/``).
        bundle_id: Logical bundle identifier read from the manifest.
        contract_id: Stable contract identifier read from the manifest.
        contract_type: Free-form contract type from the manifest (for example
            ``"services_agreement"``).
        input_file: Primary contract filename declared by the manifest.
        bundle_path: Filesystem path to the source bundle directory.
        run_directory: Filesystem path to this run's working directory.
        input_snapshot_directory: Path to the immutable copy of inputs.
        contract_sha256: SHA256 hex digest of the primary contract.
        received_timestamp: Absolute capture time in strict ISO 8601 format.
        document_type: High-level classification of the primary contract.
        jurisdiction: Governing jurisdiction declared by the manifest, if any.
        counterparty_name_from_manifest: Counterparty declared by the manifest,
            if any.
        contains_personal_data: Whether the manifest flags personal data.
        policy_files: Mapping of policy/document category to snapshot path.
        documents: Classified documents admitted into the run.
        ignored_files: Files that were filtered out, with reasons.
        initial_risk_indicators: Intake-level risk signals for downstream
            agents. Agent A does NOT perform final risk scoring.
        status: Lifecycle status of the run (for example
            ``"intake_completed"``).
    """

    run_id: StrictStr = Field(..., min_length=1)
    bundle_id: StrictStr = Field(..., min_length=1)
    contract_id: StrictStr = Field(..., min_length=1)
    contract_type: StrictStr = Field(..., min_length=1)
    input_file: StrictStr = Field(..., min_length=1)
    bundle_path: StrictStr = Field(..., min_length=1)
    run_directory: StrictStr = Field(..., min_length=1)
    input_snapshot_directory: StrictStr = Field(..., min_length=1)
    contract_sha256: StrictStr = Field(..., min_length=1)
    received_timestamp: StrictStr = Field(..., min_length=1)
    document_type: DocumentType

    jurisdiction: StrictStr | None = None
    counterparty_name_from_manifest: StrictStr | None = None
    contains_personal_data: StrictBool = False

    policy_files: dict[str, str] = Field(default_factory=dict)
    documents: list[dict] = Field(default_factory=list)
    ignored_files: list[dict] = Field(default_factory=list)
    initial_risk_indicators: list[str] = Field(default_factory=list)

    status: StrictStr = Field(..., min_length=1)

    @field_validator(
        "run_id",
        "bundle_id",
        "contract_id",
        "contract_type",
        "input_file",
        "bundle_path",
        "run_directory",
        "input_snapshot_directory",
        "status",
    )
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        """Reject empty or whitespace-only required identifiers and paths."""
        if not value.strip():
            raise ValueError("Value cannot be blank.")
        return value

    @field_validator("contract_sha256")
    @classmethod
    def must_be_sha256_hex(cls, value: str) -> str:
        """Ensure the digest is a 64-character lowercase hexadecimal string."""
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(c not in "0123456789abcdef" for c in normalized):
            raise ValueError(
                "contract_sha256 must be a 64-character hexadecimal SHA256 digest."
            )
        return normalized

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
