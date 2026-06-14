"""Intake Agent for the ICRAS ingestion framework.

The Intake Agent is the entry gate for raw contract packages. It accepts a
bundle directory of source documents, validates that the bundle is usable,
mints a unique session-tracking identifier, captures the intake timestamp, and
provisions an isolated per-run working directory under ``runs/``. The validated
:class:`~app.schemas.context_packet.ContextPacket` is both persisted to disk as
``context_packet.json`` and returned to the caller so downstream agents can
pick up the run.
"""

from datetime import datetime
from pathlib import Path
import uuid

from app.schemas.context_packet import ContextPacket, DocumentType

# Root directory under which every isolated intake run directory is created.
# This is gitignored so local runs are never committed.
RUNS_ROOT = Path("runs")

# Strftime pattern that yields an OS-filesystem-safe directory token, e.g.
# 20260614_121900.
DIRECTORY_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


class IntakeAgent:
    """Manages intake of raw contract bundles into isolated run directories.

    The agent is intentionally stateless between runs: every call to
    :meth:`run` produces a fresh session identifier and run directory, so a
    single instance can be safely reused for many bundles.
    """

    def __init__(self, runs_root: str | Path = RUNS_ROOT) -> None:
        """Initialize the agent.

        Args:
            runs_root: Base directory under which per-run directories are
                created. Defaults to ``runs/`` in the current working
                directory.
        """
        self._runs_root = Path(runs_root)

    def run(self, bundle_path: str, document_type: str) -> ContextPacket:
        """Ingest a contract bundle and produce a validated context packet.

        The method performs the following sequential business logic:

        1. Verify ``bundle_path`` is a non-empty string pointing at an existing
           directory that contains at least one file. Raise
           :class:`FileNotFoundError` if the path is empty, missing, not a
           directory, or contains no files.
        2. Resolve ``document_type`` into a :class:`DocumentType` enum member,
           raising :class:`ValueError` for unknown values.
        3. Generate a cryptographically random session-tracking identifier via
           ``uuid.uuid4().hex`` and use it as the ``contract_id``.
        4. Capture the current absolute timestamp in ISO 8601 format.
        5. Provision an isolated run directory named
           ``run_{YYYYMMDD_HHMMSS}_{contract_id}`` under the runs root.
        6. Build the strict :class:`ContextPacket` and persist it to
           ``context_packet.json`` inside the run directory.

        Args:
            bundle_path: Filesystem path to the source bundle directory.
            document_type: Document category; must match a
                :class:`DocumentType` value.

        Returns:
            The validated :class:`ContextPacket` instance.

        Raises:
            FileNotFoundError: If the bundle path is empty, does not exist, is
                not a directory, or contains no files.
            ValueError: If ``document_type`` is not a recognized
                :class:`DocumentType`.
            OSError: If the run directory cannot be created or the packet
                cannot be written (for example, due to permission issues).
        """
        resolved_document_type = self._resolve_document_type(document_type)
        bundle_directory = self._validate_bundle(bundle_path)

        contract_id = uuid.uuid4().hex
        captured_at = datetime.now().astimezone()
        received_timestamp = captured_at.isoformat()
        timestamp_token = captured_at.strftime(DIRECTORY_TIMESTAMP_FORMAT)

        run_directory = self._create_run_directory(timestamp_token, contract_id)

        context_packet = ContextPacket(
            contract_id=contract_id,
            bundle_path=str(bundle_directory),
            received_timestamp=received_timestamp,
            document_type=resolved_document_type,
        )

        self._persist_packet(context_packet, run_directory)

        return context_packet

    @staticmethod
    def _resolve_document_type(document_type: str) -> DocumentType:
        """Coerce a raw string into a :class:`DocumentType` member.

        Args:
            document_type: Raw document type value.

        Returns:
            The matching :class:`DocumentType` member.

        Raises:
            ValueError: If the value does not match any known document type.
        """
        try:
            return DocumentType(document_type)
        except ValueError as exc:
            allowed = ", ".join(member.value for member in DocumentType)
            raise ValueError(
                f"Unsupported document_type '{document_type}'. "
                f"Expected one of: {allowed}."
            ) from exc

    @staticmethod
    def _validate_bundle(bundle_path: str) -> Path:
        """Validate that the bundle path exists and contains files.

        Args:
            bundle_path: Filesystem path to the source bundle directory.

        Returns:
            The validated bundle directory as a :class:`~pathlib.Path`.

        Raises:
            FileNotFoundError: If the path is empty, missing, not a directory,
                or contains no files.
        """
        if not bundle_path or not bundle_path.strip():
            raise FileNotFoundError("bundle_path cannot be empty.")

        bundle_directory = Path(bundle_path)

        if not bundle_directory.exists():
            raise FileNotFoundError(
                f"Bundle path does not exist: {bundle_directory}"
            )

        if not bundle_directory.is_dir():
            raise FileNotFoundError(
                f"Bundle path is not a directory: {bundle_directory}"
            )

        try:
            has_files = any(entry.is_file() for entry in bundle_directory.iterdir())
        except OSError as exc:
            raise FileNotFoundError(
                f"Bundle path could not be read: {bundle_directory}"
            ) from exc

        if not has_files:
            raise FileNotFoundError(
                f"Bundle directory contains no files: {bundle_directory}"
            )

        return bundle_directory

    def _create_run_directory(self, timestamp_token: str, contract_id: str) -> Path:
        """Create the isolated run directory for this intake.

        Args:
            timestamp_token: Filesystem-safe timestamp string.
            contract_id: Session-tracking identifier for this run.

        Returns:
            The created run directory as a :class:`~pathlib.Path`.

        Raises:
            OSError: If the directory cannot be created (for example, due to
                permission issues).
        """
        run_directory = self._runs_root / f"run_{timestamp_token}_{contract_id}"
        try:
            run_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(
                f"Failed to create run directory: {run_directory} ({exc})"
            ) from exc
        return run_directory

    @staticmethod
    def _persist_packet(context_packet: ContextPacket, run_directory: Path) -> None:
        """Serialize the context packet into the run directory.

        Args:
            context_packet: The validated packet to persist.
            run_directory: The directory into which ``context_packet.json`` is
                written.

        Raises:
            OSError: If the packet cannot be written (for example, due to
                permission issues).
        """
        packet_path = run_directory / "context_packet.json"
        try:
            context_packet.to_json_file(packet_path)
        except OSError as exc:
            raise OSError(
                f"Failed to write context packet: {packet_path} ({exc})"
            ) from exc
