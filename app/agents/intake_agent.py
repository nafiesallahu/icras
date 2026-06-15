"""Agent A — Intake & Context Agent / Gatekeeper for ICRAS.

Agent A is the entry gate of the Intelligent Contract Review & Risk Analysis
System. It accepts a *contract bundle* directory, validates that the bundle is
complete, classifies every file, snapshots the inputs into an isolated run
directory, and builds the canonical
:class:`~app.schemas.context_packet.ContextPacket` that every downstream agent
depends on.

Agent A covers four project requirements:

1. **Contract Processing Gateway** — loads a bundle and classifies document
   types to initiate the workflow.
2. **Context Packet Construction** — aggregates manifest metadata, jurisdiction
   and counterparty references, and initial risk indicators.
3. **Universal Evidence Index** — initializes ``evidence_index.json`` so
   downstream agents can attach clause-level evidence to source documents.
4. **Risk Detection & Filtering** — applies intake-level heuristics (new
   counterparty, high-risk jurisdiction, personal data, ignored files) and
   filters out unsupported files.

Design rules (deliberately enforced here):

* No LLM is used.
* No final risk scoring, approval routing, or clause extraction is performed.
* The pipeline is file-based and deterministic except for ``run_id`` and
  timestamps.

Running :meth:`IntakeAgent.run` produces, under ``runs/<run_id>/``:

* ``input_snapshot/`` — immutable copies of the admitted bundle files.
* ``context_packet.json`` — the validated :class:`ContextPacket`.
* ``evidence_index.json`` — document-level evidence index (no items yet).
* ``audit_log.md`` — a human-readable intake audit trail.
* ``metrics.json`` — machine-readable run metrics.
"""

from __future__ import annotations

import csv
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import yaml

from app.schemas.context_packet import ContextPacket, DocumentType
from app.utils.hashing import sha256_file

# Root directory under which every isolated intake run directory is created.
# This is gitignored so local runs are never committed.
RUNS_ROOT = Path("runs")

# Strftime pattern that yields an OS-filesystem-safe directory token, e.g.
# 20260614_121900.
DIRECTORY_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# Canonical bundle filenames mapped to their in-bundle classification category.
# Order matters: it fixes the deterministic ``doc_NNN`` assignment and the order
# documents appear in the context packet / evidence index.
PRIMARY_CONTRACT_FILENAME = "contract.pdf"
REQUIRED_FILE_CATEGORIES: dict[str, str] = {
    "contract.pdf": "primary_contract",
    "manifest.yaml": "manifest",
    "playbook.yaml": "playbook",
    "approval_policy.yaml": "approval_policy",
    "vendor_master.csv": "vendor_master",
    "jurisdiction_rules.yaml": "jurisdiction_rules",
}

# Categories that represent policy/configuration inputs (everything except the
# primary contract). Used to populate ContextPacket.policy_files.
POLICY_CATEGORIES = {
    "manifest",
    "playbook",
    "approval_policy",
    "vendor_master",
    "jurisdiction_rules",
}

# Manifest fields that must be present for a usable run.
CRITICAL_MANIFEST_FIELDS = ("bundle_id", "contract_id", "input_file")

# Jurisdiction risk levels considered elevated at the intake stage. Agent A only
# raises a *candidate* indicator; Agent E performs the final scoring.
ELEVATED_JURISDICTION_RISK_LEVELS = {"high", "critical"}

# Reason recorded for any file that is not one of the recognized bundle files.
IGNORED_FILE_REASON = "unsupported_or_irrelevant_file_type"

STATUS_COMPLETED = "intake_completed"
STATUS_FAILED = "intake_failed"


class IntakeError(Exception):
    """Raised when a contract bundle cannot be ingested by Agent A."""


class IntakeAgent:
    """Ingests raw contract bundles into isolated, validated run directories.

    The agent is stateless between runs: every call to :meth:`run` mints a fresh
    ``run_id`` and run directory, so a single instance can be reused safely for
    many bundles.
    """

    def __init__(self, runs_root: str | Path = RUNS_ROOT) -> None:
        """Initialize the agent.

        Args:
            runs_root: Base directory under which per-run directories are
                created. Defaults to ``runs/`` in the current working
                directory.
        """
        self._runs_root = Path(runs_root)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, bundle_path: str | Path) -> ContextPacket:
        """Ingest a contract bundle and produce a validated context packet.

        Args:
            bundle_path: Filesystem path to the source bundle directory.

        Returns:
            The validated :class:`ContextPacket` for this run.

        Raises:
            IntakeError: If the bundle is missing/invalid, a required file is
                absent, or a policy/manifest file cannot be parsed.
        """
        bundle_directory = self._validate_bundle_directory(bundle_path)

        run_id = self._generate_run_id()
        captured_at = datetime.now().astimezone()
        received_timestamp = captured_at.isoformat()

        run_directory = self._create_run_directory(run_id)
        snapshot_directory = run_directory / "input_snapshot"
        snapshot_directory.mkdir(parents=True, exist_ok=True)

        # Classify everything in the bundle before doing anything else so that
        # failure artifacts can report what was found.
        included, ignored, missing = self._classify_bundle(bundle_directory)

        if missing:
            indicators = [f"missing_required_file:{name}" for name in missing]
            message = (
                "Missing required bundle file(s): " + ", ".join(missing)
            )
            self._write_failure_artifacts(
                run_id=run_id,
                run_directory=run_directory,
                bundle_directory=bundle_directory,
                received_timestamp=received_timestamp,
                included=included,
                ignored=ignored,
                indicators=indicators,
                error_message=message,
            )
            raise IntakeError(message)

        # Hash and snapshot every admitted file.
        documents = self._build_documents(included, bundle_directory)
        self._snapshot_files(included, bundle_directory, snapshot_directory)

        contract_sha256 = documents[0]["sha256"]

        # Parse policy/manifest inputs (each may raise IntakeError on bad data).
        manifest = self._read_manifest(
            bundle_directory / "manifest.yaml",
            run_id=run_id,
            run_directory=run_directory,
            bundle_directory=bundle_directory,
            received_timestamp=received_timestamp,
            included=included,
            ignored=ignored,
        )
        jurisdiction_rules = self._read_jurisdiction_rules(
            bundle_directory / "jurisdiction_rules.yaml",
            run_id=run_id,
            run_directory=run_directory,
            bundle_directory=bundle_directory,
            received_timestamp=received_timestamp,
            included=included,
            ignored=ignored,
        )
        vendor_names = self._read_vendor_master(
            bundle_directory / "vendor_master.csv",
            run_id=run_id,
            run_directory=run_directory,
            bundle_directory=bundle_directory,
            received_timestamp=received_timestamp,
            included=included,
            ignored=ignored,
        )

        counterparty = self._clean_optional(manifest.get("counterparty_name"))
        jurisdiction = self._clean_optional(manifest.get("jurisdiction"))
        contains_personal_data = bool(manifest.get("contains_personal_data", False))

        risk_indicators = self._detect_risk_indicators(
            counterparty=counterparty,
            jurisdiction=jurisdiction,
            contains_personal_data=contains_personal_data,
            vendor_names=vendor_names,
            jurisdiction_rules=jurisdiction_rules,
            ignored=ignored,
        )

        # Snapshot-relative paths keep policy_files deterministic across runs
        # while still letting downstream agents resolve them against
        # ``run_directory``.
        policy_files = {
            document["document_type"]: f"input_snapshot/{document['filename']}"
            for document in documents
            if document["document_type"] in POLICY_CATEGORIES
        }

        context_packet = ContextPacket(
            run_id=run_id,
            bundle_id=str(manifest["bundle_id"]),
            contract_id=str(manifest["contract_id"]),
            contract_type=str(manifest.get("contract_type") or "unknown"),
            input_file=str(manifest["input_file"]),
            bundle_path=str(bundle_directory),
            run_directory=str(run_directory),
            input_snapshot_directory=str(snapshot_directory),
            contract_sha256=contract_sha256,
            received_timestamp=received_timestamp,
            document_type=DocumentType.from_contract_type(manifest.get("contract_type")),
            jurisdiction=jurisdiction,
            counterparty_name_from_manifest=counterparty,
            contains_personal_data=contains_personal_data,
            policy_files=policy_files,
            documents=documents,
            ignored_files=ignored,
            initial_risk_indicators=risk_indicators,
            status=STATUS_COMPLETED,
        )

        self._persist_packet(context_packet, run_directory)
        self._write_evidence_index(documents, run_directory)
        self._write_success_audit_log(context_packet, run_directory)
        self._write_metrics(
            run_directory=run_directory,
            run_id=run_id,
            contract_id=context_packet.contract_id,
            started_at=received_timestamp,
            completed_at=datetime.now().astimezone().isoformat(),
            documents_found=len(included) + len(ignored),
            documents_included=len(included),
            documents_ignored=len(ignored),
            risk_indicator_count=len(risk_indicators),
            status=STATUS_COMPLETED,
        )

        return context_packet

    # ------------------------------------------------------------------
    # Bundle validation & classification
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_bundle_directory(bundle_path: str | Path) -> Path:
        """Validate that the bundle path exists and is a non-empty directory.

        Raises:
            IntakeError: If the path is empty, missing, not a directory, or
                contains no files.
        """
        if bundle_path is None or not str(bundle_path).strip():
            raise IntakeError("bundle_path cannot be empty.")

        bundle_directory = Path(bundle_path)

        if not bundle_directory.exists():
            raise IntakeError(f"Bundle path does not exist: {bundle_directory}")

        if not bundle_directory.is_dir():
            raise IntakeError(f"Bundle path is not a directory: {bundle_directory}")

        try:
            has_files = any(entry.is_file() for entry in bundle_directory.iterdir())
        except OSError as exc:
            raise IntakeError(
                f"Bundle path could not be read: {bundle_directory} ({exc})"
            ) from exc

        if not has_files:
            raise IntakeError(
                f"Bundle directory contains no files: {bundle_directory}"
            )

        return bundle_directory

    @staticmethod
    def _classify_bundle(
        bundle_directory: Path,
    ) -> tuple[list[tuple[str, str]], list[dict], list[str]]:
        """Classify every file in the bundle.

        Returns:
            A tuple ``(included, ignored, missing)`` where:
                * ``included`` is an ordered list of ``(filename, category)``
                  for recognized files, in canonical bundle order.
                * ``ignored`` is a list of ``{"filename", "reason"}`` dicts for
                  unsupported/irrelevant files.
                * ``missing`` is a list of required filenames that are absent.
        """
        present = {
            entry.name
            for entry in bundle_directory.iterdir()
            if entry.is_file()
        }

        included: list[tuple[str, str]] = [
            (name, category)
            for name, category in REQUIRED_FILE_CATEGORIES.items()
            if name in present
        ]

        ignored: list[dict] = [
            {"filename": name, "reason": IGNORED_FILE_REASON}
            for name in sorted(present - set(REQUIRED_FILE_CATEGORIES))
        ]

        missing = [
            name
            for name in REQUIRED_FILE_CATEGORIES
            if name not in present
        ]

        return included, ignored, missing

    @staticmethod
    def _build_documents(
        included: list[tuple[str, str]], bundle_directory: Path
    ) -> list[dict]:
        """Build the classified, hashed document entries for the packet.

        The primary contract is always first so ``documents[0]`` is the contract
        and its hash is the canonical ``contract_sha256``.
        """
        documents: list[dict] = []
        for index, (filename, category) in enumerate(included, start=1):
            documents.append(
                {
                    "document_id": f"doc_{index:03d}",
                    "filename": filename,
                    "document_type": category,
                    "sha256": sha256_file(bundle_directory / filename),
                    "included": True,
                    "reason": None,
                }
            )
        return documents

    @staticmethod
    def _snapshot_files(
        included: list[tuple[str, str]],
        bundle_directory: Path,
        snapshot_directory: Path,
    ) -> None:
        """Copy admitted bundle files into the immutable input snapshot."""
        for filename, _category in included:
            shutil.copy2(
                bundle_directory / filename,
                snapshot_directory / filename,
            )

    # ------------------------------------------------------------------
    # Policy / manifest parsing
    # ------------------------------------------------------------------
    def _read_manifest(
        self,
        manifest_path: Path,
        **failure_context: object,
    ) -> dict:
        """Parse and minimally validate ``manifest.yaml``.

        Raises:
            IntakeError: If the manifest cannot be parsed or a critical field
                (``bundle_id``, ``contract_id``, ``input_file``) is missing.
        """
        try:
            manifest = self._load_yaml_mapping(manifest_path)
        except IntakeError as exc:
            self._write_failure_artifacts(
                error_message=str(exc), **failure_context  # type: ignore[arg-type]
            )
            raise

        missing_fields = [
            field
            for field in CRITICAL_MANIFEST_FIELDS
            if not str(manifest.get(field) or "").strip()
        ]
        if missing_fields:
            message = (
                "manifest.yaml is missing required field(s): "
                + ", ".join(missing_fields)
            )
            self._write_failure_artifacts(
                error_message=message, **failure_context  # type: ignore[arg-type]
            )
            raise IntakeError(message)

        return manifest

    def _read_jurisdiction_rules(
        self,
        rules_path: Path,
        **failure_context: object,
    ) -> dict:
        """Parse ``jurisdiction_rules.yaml`` into a mapping.

        Raises:
            IntakeError: If the file cannot be parsed.
        """
        try:
            return self._load_yaml_mapping(rules_path)
        except IntakeError as exc:
            self._write_failure_artifacts(
                error_message=str(exc), **failure_context  # type: ignore[arg-type]
            )
            raise

    def _read_vendor_master(
        self,
        vendor_path: Path,
        **failure_context: object,
    ) -> set[str]:
        """Parse ``vendor_master.csv`` and return the set of vendor names.

        Raises:
            IntakeError: If the file cannot be read or parsed.
        """
        try:
            with vendor_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or "vendor_name" not in reader.fieldnames:
                    raise IntakeError(
                        "vendor_master.csv must contain a 'vendor_name' column."
                    )
                names = {
                    (row.get("vendor_name") or "").strip()
                    for row in reader
                    if (row.get("vendor_name") or "").strip()
                }
        except IntakeError as exc:
            self._write_failure_artifacts(
                error_message=str(exc), **failure_context  # type: ignore[arg-type]
            )
            raise
        except (OSError, csv.Error) as exc:
            message = f"Could not read vendor_master.csv: {vendor_path} ({exc})"
            self._write_failure_artifacts(
                error_message=message, **failure_context  # type: ignore[arg-type]
            )
            raise IntakeError(message) from exc

        return names

    @staticmethod
    def _load_yaml_mapping(path: Path) -> dict:
        """Load a YAML file that must deserialize into a mapping.

        Raises:
            IntakeError: If the file is unreadable, malformed, or not a mapping.
        """
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise IntakeError(f"Could not read {path.name}: {path} ({exc})") from exc

        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise IntakeError(f"Could not parse {path.name}: {exc}") from exc

        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise IntakeError(
                f"{path.name} must contain a mapping at the top level."
            )
        return data

    @staticmethod
    def _clean_optional(value: object) -> str | None:
        """Normalize an optional manifest string value to ``str`` or ``None``."""
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    # ------------------------------------------------------------------
    # Risk detection (intake-level only)
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_risk_indicators(
        *,
        counterparty: str | None,
        jurisdiction: str | None,
        contains_personal_data: bool,
        vendor_names: set[str],
        jurisdiction_rules: dict,
        ignored: list[dict],
    ) -> list[str]:
        """Compute deterministic intake-level risk indicators.

        These are *candidates* only. Agent A never performs final risk scoring,
        approval routing, or clause-level findings.
        """
        indicators: list[str] = []

        if counterparty:
            exact_match = counterparty in vendor_names
            if not exact_match:
                case_insensitive_match = counterparty.lower() in {
                    name.lower() for name in vendor_names
                }
                indicators.append("unknown_counterparty_candidate")
                if not case_insensitive_match:
                    indicators.append("new_counterparty_candidate")

        if jurisdiction:
            risk_level = IntakeAgent._jurisdiction_risk_level(
                jurisdiction, jurisdiction_rules
            )
            if risk_level in ELEVATED_JURISDICTION_RISK_LEVELS:
                indicators.append("high_risk_jurisdiction_candidate")

        if contains_personal_data:
            indicators.append("contains_personal_data")

        if ignored:
            indicators.append("ignored_files_present")
            indicators.extend(
                f"unsupported_file_type:{entry['filename']}" for entry in ignored
            )

        return indicators

    @staticmethod
    def _jurisdiction_risk_level(
        jurisdiction: str, jurisdiction_rules: dict
    ) -> str | None:
        """Look up the risk level for a jurisdiction from the rules mapping."""
        jurisdictions = jurisdiction_rules.get("jurisdictions")
        if not isinstance(jurisdictions, dict):
            return None
        entry = jurisdictions.get(jurisdiction)
        if not isinstance(entry, dict):
            return None
        level = entry.get("risk_level")
        return str(level).strip().lower() if level is not None else None

    # ------------------------------------------------------------------
    # Run directory & artifact writers
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_run_id() -> str:
        """Mint a unique, human-readable run identifier."""
        timestamp_token = datetime.now().astimezone().strftime(
            DIRECTORY_TIMESTAMP_FORMAT
        )
        return f"run_{timestamp_token}_{uuid.uuid4().hex[:8]}"

    def _create_run_directory(self, run_id: str) -> Path:
        """Create the isolated run directory for this intake.

        Raises:
            IntakeError: If the directory cannot be created.
        """
        run_directory = self._runs_root / run_id
        try:
            run_directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise IntakeError(
                f"Failed to create run directory: {run_directory} ({exc})"
            ) from exc
        return run_directory

    @staticmethod
    def _persist_packet(context_packet: ContextPacket, run_directory: Path) -> None:
        """Serialize the context packet into the run directory."""
        packet_path = run_directory / "context_packet.json"
        try:
            context_packet.to_json_file(packet_path)
        except OSError as exc:
            raise IntakeError(
                f"Failed to write context packet: {packet_path} ({exc})"
            ) from exc

    @staticmethod
    def _write_evidence_index(documents: list[dict], run_directory: Path) -> None:
        """Initialize ``evidence_index.json`` with documents and no items.

        Agent B will later append clause-level evidence items (page, bbox, text
        excerpts, clause IDs) to ``evidence_items``.
        """
        evidence_index = {
            "documents": [
                {
                    "document_id": document["document_id"],
                    "filename": document["filename"],
                    "document_type": document["document_type"],
                    "sha256": document["sha256"],
                    "included": document["included"],
                }
                for document in documents
            ],
            "evidence_items": [],
        }
        (run_directory / "evidence_index.json").write_text(
            json.dumps(evidence_index, indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_success_audit_log(
        context_packet: ContextPacket, run_directory: Path
    ) -> None:
        """Write the human-readable intake audit log for a successful run."""
        indicators = context_packet.initial_risk_indicators
        indicators_text = ", ".join(indicators) if indicators else "none"
        lines = [
            f"# Audit Log — {context_packet.run_id}",
            "",
            "## Step 1: Intake",
            f"- Loaded bundle: {context_packet.bundle_path}",
            f"- Bundle ID: {context_packet.bundle_id}",
            f"- Contract ID: {context_packet.contract_id}",
            f"- Primary contract: {PRIMARY_CONTRACT_FILENAME}",
            f"- Contract SHA256: {context_packet.contract_sha256}",
            "- Required files validated: yes",
            "- Input snapshot created: yes",
            f"- Documents classified: {len(context_packet.documents)}",
            f"- Ignored files: {len(context_packet.ignored_files)}",
            f"- Initial risk indicators: {indicators_text}",
            f"- Status: {context_packet.status}",
            "",
        ]
        (run_directory / "audit_log.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    @staticmethod
    def _write_metrics(
        *,
        run_directory: Path,
        run_id: str,
        contract_id: str | None,
        started_at: str,
        completed_at: str | None,
        documents_found: int,
        documents_included: int,
        documents_ignored: int,
        risk_indicator_count: int,
        status: str,
    ) -> None:
        """Write/refresh ``metrics.json`` for the Agent A stage."""
        metrics = {
            "run_id": run_id,
            "contract_id": contract_id,
            "started_at": started_at,
            "intake_completed_at": completed_at,
            "agents_completed": ["intake"] if status == STATUS_COMPLETED else [],
            "documents_found": documents_found,
            "documents_included": documents_included,
            "documents_ignored": documents_ignored,
            "initial_risk_indicators_count": risk_indicator_count,
            "status": status,
        }
        (run_directory / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )

    def _write_failure_artifacts(
        self,
        *,
        run_id: str,
        run_directory: Path,
        bundle_directory: Path,
        received_timestamp: str,
        included: list[tuple[str, str]],
        ignored: list[dict],
        error_message: str,
        indicators: list[str] | None = None,
    ) -> None:
        """Record a failed intake in ``audit_log.md`` and ``metrics.json``.

        Best-effort: writing failure artifacts must never mask the original
        error, so any I/O problem here is swallowed.
        """
        indicators = indicators or []
        indicators_text = ", ".join(indicators) if indicators else "none"
        try:
            run_directory.mkdir(parents=True, exist_ok=True)
            audit_lines = [
                f"# Audit Log — {run_id}",
                "",
                "## Step 1: Intake",
                f"- Loaded bundle: {bundle_directory}",
                f"- Documents classified: {len(included)}",
                f"- Ignored files: {len(ignored)}",
                f"- Initial risk indicators: {indicators_text}",
                f"- Error: {error_message}",
                f"- Status: {STATUS_FAILED}",
                "",
            ]
            (run_directory / "audit_log.md").write_text(
                "\n".join(audit_lines), encoding="utf-8"
            )
            self._write_metrics(
                run_directory=run_directory,
                run_id=run_id,
                contract_id=None,
                started_at=received_timestamp,
                completed_at=None,
                documents_found=len(included) + len(ignored),
                documents_included=len(included),
                documents_ignored=len(ignored),
                risk_indicator_count=len(indicators),
                status=STATUS_FAILED,
            )
        except OSError:
            # Never let failure-logging hide the real intake error.
            pass
