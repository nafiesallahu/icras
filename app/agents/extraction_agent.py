"""Extraction Agent (Agent B) for the ICRAS pipeline.

Agent B converts an unstructured born-digital contract PDF into the structured,
machine-readable ``extracted_contract.json`` artifact that downstream agents
consume. It also updates the shared run artifacts (``evidence_index.json``,
``audit_log.md``, ``metrics.json``) with clause-level evidence and run metrics.

Routing per run:

* ``ENV_MODE == "test"`` or ``MOCK_PIPELINE == "True"`` force the deterministic
  synthetic fallback (no live parsing) — preserved for tests and offline runs.
* Otherwise Agent B performs **real PDF clause extraction** with
  :class:`~app.services.pdf_parser.PdfParser` and
  :class:`~app.services.clause_detector.ClauseDetector`.
* If live PDF parsing fails, Agent B falls back to the synthetic fixture for the
  contract (``extraction_mode = synthetic_fallback``, overall confidence 0.60)
  so the pipeline can continue.

Agent B only extracts and structures. It performs no risk scoring, approval
routing, or final decision logic, and uses no LLM.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.schemas.context_packet import ContextPacket
from app.schemas.evidence import EvidenceIndex, EvidenceItem
from app.schemas.extracted_contract import (
    ClauseLocation,
    ContractClause,
    ExtractedContract,
    ExtractionMode,
)
from app.services.clause_detector import (
    DEFAULT_MINIMUM_CLAUSE_CONFIDENCE,
    DEFAULT_MINIMUM_SIGNATURE_CONFIDENCE,
    ClauseDetector,
)
from app.services.pdf_parser import ParsedPdf, PdfParseError, PdfParser
from app.services.synthetic_extractor import SyntheticFallbackEngine

logger = logging.getLogger(__name__)

CONTEXT_PACKET_FILENAME = "context_packet.json"
EXTRACTED_CONTRACT_FILENAME = "extracted_contract.json"
EVIDENCE_INDEX_FILENAME = "evidence_index.json"
AUDIT_LOG_FILENAME = "audit_log.md"
METRICS_FILENAME = "metrics.json"
PLAYBOOK_SNAPSHOT_NAME = "playbook.yaml"

# Scenario used when no scenario can be inferred while in synthetic mode.
DEFAULT_SCENARIO_ID = "scenario_01"

ENV_MODE_VARIABLE = "ENV_MODE"
ENV_MODE_TEST_VALUE = "test"
MOCK_PIPELINE_VARIABLE = "MOCK_PIPELINE"
MOCK_PIPELINE_ENABLED_VALUE = "True"

# Overall confidence assigned when synthetic fallback is used because live PDF
# parsing failed (Masterplan requirement).
PARSER_FAILURE_FALLBACK_CONFIDENCE = 0.60

# Fallback reason tags recorded in audit/metrics.
REASON_ENV_FORCED = "env_forced"

STATUS_COMPLETED = "extraction_completed"
STATUS_COMPLETED_WITH_FALLBACK = "extraction_completed_with_fallback"
STATUS_FAILED = "extraction_failed"


@dataclass(frozen=True)
class _ExtractionOutcome:
    """Internal record describing how a single extraction resolved."""

    contract: ExtractedContract
    pages_parsed: int | None
    synthetic_fallback_used: bool
    fallback_reason: str | None
    status: str


class ExtractionAgent:
    """Orchestrates contract extraction for a single run."""

    def __init__(
        self,
        synthetic_engine: SyntheticFallbackEngine | None = None,
        pdf_parser: PdfParser | None = None,
        clause_detector: ClauseDetector | None = None,
    ) -> None:
        self._synthetic_engine = synthetic_engine or SyntheticFallbackEngine()
        self._pdf_parser = pdf_parser or PdfParser()
        self._clause_detector = clause_detector or ClauseDetector()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self, run_directory: str, scenario_id: str | None = None
    ) -> ExtractedContract:
        """Execute extraction for a run and persist all resulting artifacts.

        Args:
            run_directory: Path to the per-run working directory created by the
                intake agent.
            scenario_id: Optional scenario identifier used when the synthetic
                fallback is forced by the environment.

        Returns:
            The validated :class:`ExtractedContract` produced for this run.

        Raises:
            FileNotFoundError: If ``run_directory`` does not exist.
            PdfParseError / FileNotFoundError: If live parsing fails *and* no
                synthetic fallback fixture is available.
            ValueError / ValidationError: On unrecoverable schema problems.
            OSError: If artifacts cannot be written.
        """
        run_directory_path = self._resolve_run_directory(run_directory)
        context_packet = self._load_context_packet(run_directory_path)

        if self._is_synthetic_mode():
            outcome = self._run_env_forced_synthetic(scenario_id)
        else:
            outcome = self._run_live_extraction(
                run_directory_path, context_packet
            )

        if not isinstance(outcome.contract, ExtractedContract):
            raise TypeError(
                "Extraction did not produce an ExtractedContract instance; "
                f"got {type(outcome.contract).__name__}."
            )

        return self._finalize(outcome, run_directory_path, context_packet)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    def _run_env_forced_synthetic(
        self, scenario_id: str | None
    ) -> _ExtractionOutcome:
        """Resolve a synthetic fixture because the environment forces it."""
        effective_scenario_id = scenario_id or DEFAULT_SCENARIO_ID
        logger.info(
            "Synthetic mode active; bypassing live PDF extraction and loading "
            "scenario '%s'.",
            effective_scenario_id,
        )
        contract = self._synthetic_engine.get_scenario_fixture(
            effective_scenario_id
        )
        return _ExtractionOutcome(
            contract=contract,
            pages_parsed=None,
            synthetic_fallback_used=True,
            fallback_reason=REASON_ENV_FORCED,
            status=STATUS_COMPLETED,
        )

    def _run_live_extraction(
        self,
        run_directory_path: Path,
        context_packet: ContextPacket | None,
    ) -> _ExtractionOutcome:
        """Run real PDF clause extraction, falling back to synthetic on error."""
        try:
            pdf_path = self._resolve_contract_pdf(
                run_directory_path, context_packet
            )
            parsed = self._pdf_parser.parse(pdf_path)
            return self._extract_from_parsed_pdf(
                parsed, run_directory_path, context_packet
            )
        except (PdfParseError, FileNotFoundError) as exc:
            logger.warning(
                "Live PDF extraction failed (%s); attempting synthetic "
                "fallback.",
                exc,
            )
            return self._run_parser_failure_fallback(
                run_directory_path, context_packet, exc
            )

    def _extract_from_parsed_pdf(
        self,
        parsed: ParsedPdf,
        run_directory_path: Path,
        context_packet: ContextPacket | None,
    ) -> _ExtractionOutcome:
        """Detect clauses from a successfully parsed PDF."""
        minimum_clause, minimum_signature = self._load_confidence_thresholds(
            run_directory_path, context_packet
        )
        result = self._clause_detector.detect(
            parsed,
            minimum_clause_confidence=minimum_clause,
            minimum_signature_confidence=minimum_signature,
        )

        if not result.clauses:
            raise PdfParseError(
                "PDF parsed but no clauses were detected; treating as a parsing "
                "failure to trigger synthetic fallback."
            )

        contract_id = (
            context_packet.contract_id
            if context_packet is not None
            else parsed.source_file
        )
        contract = ExtractedContract(
            contract_id=contract_id,
            source_file=parsed.source_file,
            extraction_mode=ExtractionMode.PDF_PARSER,
            overall_confidence_score=result.overall_confidence_score,
            clauses=result.clauses,
        )
        return _ExtractionOutcome(
            contract=contract,
            pages_parsed=parsed.page_count,
            synthetic_fallback_used=False,
            fallback_reason=None,
            status=STATUS_COMPLETED,
        )

    def _run_parser_failure_fallback(
        self,
        run_directory_path: Path,
        context_packet: ContextPacket | None,
        error: Exception,
    ) -> _ExtractionOutcome:
        """Load the synthetic fixture after a live parsing failure.

        Sets ``extraction_mode = synthetic_fallback`` and overall confidence to
        :data:`PARSER_FAILURE_FALLBACK_CONFIDENCE`. If no fixture is available,
        writes failure artifacts and re-raises so the failure is loud.
        """
        scenario_id = self._scenario_id_from_context(context_packet)
        try:
            fixture = self._synthetic_engine.get_scenario_fixture(scenario_id)
        except Exception as fixture_error:  # noqa: BLE001
            logger.error(
                "Synthetic fallback fixture unavailable for scenario '%s': %s",
                scenario_id,
                fixture_error,
            )
            self._write_failure_artifacts(
                run_directory_path,
                context_packet,
                reason=(
                    f"PDF parsing failed: {error}; fallback fixture "
                    f"unavailable: {fixture_error}"
                ),
            )
            raise

        contract = fixture.model_copy(
            update={
                "extraction_mode": ExtractionMode.SYNTHETIC_FALLBACK,
                "overall_confidence_score": PARSER_FAILURE_FALLBACK_CONFIDENCE,
            }
        )
        return _ExtractionOutcome(
            contract=contract,
            pages_parsed=None,
            synthetic_fallback_used=True,
            fallback_reason=f"PDF parsing failed: {error}",
            status=STATUS_COMPLETED_WITH_FALLBACK,
        )

    # ------------------------------------------------------------------
    # Finalization (evidence index, persistence, audit log, metrics)
    # ------------------------------------------------------------------
    def _finalize(
        self,
        outcome: _ExtractionOutcome,
        run_directory_path: Path,
        context_packet: ContextPacket | None,
    ) -> ExtractedContract:
        """Wire evidence, persist the contract, and update audit + metrics."""
        contract_with_evidence, evidence_count = self._update_evidence_index(
            outcome.contract, run_directory_path, context_packet
        )

        self._persist_extracted_contract(
            contract_with_evidence, run_directory_path
        )

        low_confidence = [
            clause
            for clause in contract_with_evidence.clauses
            if clause.requires_manual_review
        ]

        self._append_audit_log(
            outcome=outcome,
            contract=contract_with_evidence,
            run_directory_path=run_directory_path,
            evidence_count=evidence_count,
            low_confidence_clauses=low_confidence,
        )
        self._update_metrics(
            outcome=outcome,
            contract=contract_with_evidence,
            run_directory_path=run_directory_path,
            evidence_count=evidence_count,
            low_confidence_count=len(low_confidence),
        )
        return contract_with_evidence

    def _update_evidence_index(
        self,
        contract: ExtractedContract,
        run_directory_path: Path,
        context_packet: ContextPacket | None,
    ) -> tuple[ExtractedContract, int]:
        """Append clause-level evidence items and back-reference their IDs.

        Returns the contract with ``evidence_ids``/``locations`` populated and
        the number of evidence items created. Deterministic across re-runs:
        previously-added extraction items (those carrying a ``clause_id``) are
        dropped before re-appending, and evidence IDs always restart at EV-001.
        """
        index = self._load_evidence_index(run_directory_path)
        document_id, filename = self._primary_document(index, contract)

        # Drop any previously-added extraction evidence for deterministic re-runs.
        index.evidence_items = [
            item for item in index.evidence_items if not item.clause_id
        ]

        new_items: list[EvidenceItem] = []
        updated_clauses: list[ContractClause] = []
        counter = 0

        for clause in contract.clauses:
            locations = self._clause_locations(clause)
            clause_evidence_ids: list[str] = []
            for location in locations:
                counter += 1
                evidence_id = f"EV-{counter:03d}"
                clause_evidence_ids.append(evidence_id)
                new_items.append(
                    EvidenceItem(
                        evidence_id=evidence_id,
                        document_id=document_id,
                        filename=filename,
                        page=location.page,
                        clause_id=clause.clause_id,
                        bbox=location.bbox,
                        text_excerpt=location.text_excerpt,
                    )
                )
            updated_clauses.append(
                clause.model_copy(
                    update={
                        "locations": locations,
                        "evidence_ids": clause_evidence_ids,
                    }
                )
            )

        index.evidence_items.extend(new_items)
        self._persist_evidence_index(index, run_directory_path)

        contract_with_evidence = contract.model_copy(
            update={"clauses": updated_clauses}
        )
        return contract_with_evidence, len(new_items)

    @staticmethod
    def _clause_locations(clause: ContractClause) -> list[ClauseLocation]:
        """Return the clause's locations, synthesizing one if none exist.

        Synthetic fixtures only carry ``page_number``/``bbox``; this turns those
        into a single :class:`ClauseLocation` so every clause has at least one
        traceable location.
        """
        if clause.locations:
            return list(clause.locations)
        excerpt = clause.text.strip()[:240] or clause.clause_id
        return [
            ClauseLocation(
                page=clause.page_number,
                bbox=clause.bbox,
                text_excerpt=excerpt,
            )
        ]

    def _load_evidence_index(self, run_directory_path: Path) -> EvidenceIndex:
        """Load the evidence index, tolerating a missing/seeded file."""
        path = run_directory_path / EVIDENCE_INDEX_FILENAME
        if not path.exists():
            return EvidenceIndex()
        try:
            return EvidenceIndex.from_json_file(path)
        except (OSError, ValueError) as exc:
            logger.warning(
                "Could not parse %s (%s); starting a fresh evidence index.",
                path,
                exc,
            )
            return EvidenceIndex()

    def _persist_evidence_index(
        self, index: EvidenceIndex, run_directory_path: Path
    ) -> None:
        index.to_json_file(run_directory_path / EVIDENCE_INDEX_FILENAME)

    @staticmethod
    def _primary_document(
        index: EvidenceIndex, contract: ExtractedContract
    ) -> tuple[str, str]:
        """Resolve the document_id/filename for the primary contract."""
        for document in index.documents:
            if document.get("document_type") == "primary_contract":
                return (
                    str(document.get("document_id", "doc_001")),
                    str(document.get("filename", contract.source_file)),
                )
        if index.documents:
            first = index.documents[0]
            return (
                str(first.get("document_id", "doc_001")),
                str(first.get("filename", contract.source_file)),
            )
        return "doc_001", contract.source_file

    # ------------------------------------------------------------------
    # Audit log & metrics
    # ------------------------------------------------------------------
    def _append_audit_log(
        self,
        *,
        outcome: _ExtractionOutcome,
        contract: ExtractedContract,
        run_directory_path: Path,
        evidence_count: int,
        low_confidence_clauses: list[ContractClause],
    ) -> None:
        """Append the ``## Step 2: Clause Extraction`` section to the audit log."""
        fallback_used = "yes" if outcome.synthetic_fallback_used else "no"
        lines = [
            "",
            "## Step 2: Clause Extraction",
            f"- Extraction mode: {contract.extraction_mode.value}",
        ]
        if outcome.synthetic_fallback_used:
            lines.append(f"- Reason: {outcome.fallback_reason}")
        else:
            lines.append(f"- PDF parsed: {contract.source_file}")
            lines.append(f"- Pages parsed: {outcome.pages_parsed}")
        lines.extend(
            [
                f"- Clauses extracted: {len(contract.clauses)}",
                f"- Evidence items created: {evidence_count}",
                f"- Overall confidence: {contract.overall_confidence_score}",
                f"- Low-confidence clauses: {len(low_confidence_clauses)}",
                f"- Synthetic fallback used: {fallback_used}",
                f"- Status: {outcome.status}",
            ]
        )
        for clause in low_confidence_clauses:
            lines.append(
                f"- Low-confidence extraction detected: {clause.clause_id} "
                f"confidence={clause.confidence_score}"
            )
        lines.append("")

        audit_path = run_directory_path / AUDIT_LOG_FILENAME
        existing = (
            audit_path.read_text(encoding="utf-8")
            if audit_path.exists()
            else ""
        )
        audit_path.write_text(existing + "\n".join(lines), encoding="utf-8")

    def _update_metrics(
        self,
        *,
        outcome: _ExtractionOutcome,
        contract: ExtractedContract,
        run_directory_path: Path,
        evidence_count: int,
        low_confidence_count: int,
    ) -> None:
        """Merge extraction metrics into ``metrics.json`` (preserving Agent A)."""
        metrics_path = run_directory_path / METRICS_FILENAME
        metrics: dict = {}
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                metrics = {}
        if not isinstance(metrics, dict):
            metrics = {}

        agents_completed = metrics.get("agents_completed")
        if not isinstance(agents_completed, list):
            agents_completed = []
        if "extraction" not in agents_completed:
            agents_completed.append("extraction")
        metrics["agents_completed"] = agents_completed

        metrics["extraction_mode"] = contract.extraction_mode.value
        metrics["overall_extraction_confidence"] = (
            contract.overall_confidence_score
        )
        metrics["clauses_extracted"] = len(contract.clauses)
        metrics["evidence_items_created"] = evidence_count
        metrics["low_confidence_clauses_count"] = low_confidence_count
        metrics["pages_parsed"] = outcome.pages_parsed
        metrics["extraction_completed_at"] = (
            datetime.now().astimezone().isoformat()
        )
        metrics["status"] = outcome.status

        metrics_path.write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )

    def _write_failure_artifacts(
        self,
        run_directory_path: Path,
        context_packet: ContextPacket | None,
        *,
        reason: str,
    ) -> None:
        """Record an unrecoverable extraction failure in audit + metrics."""
        try:
            audit_path = run_directory_path / AUDIT_LOG_FILENAME
            existing = (
                audit_path.read_text(encoding="utf-8")
                if audit_path.exists()
                else ""
            )
            lines = [
                "",
                "## Step 2: Clause Extraction",
                "- Extraction mode: synthetic_fallback",
                f"- Reason: {reason}",
                f"- Status: {STATUS_FAILED}",
                "",
            ]
            audit_path.write_text(existing + "\n".join(lines), encoding="utf-8")

            metrics_path = run_directory_path / METRICS_FILENAME
            metrics: dict = {}
            if metrics_path.exists():
                try:
                    metrics = json.loads(
                        metrics_path.read_text(encoding="utf-8")
                    )
                except (OSError, ValueError):
                    metrics = {}
            if not isinstance(metrics, dict):
                metrics = {}
            metrics["status"] = STATUS_FAILED
            metrics["extraction_completed_at"] = (
                datetime.now().astimezone().isoformat()
            )
            metrics_path.write_text(
                json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
            )
        except OSError:
            # Never let failure-logging hide the real extraction error.
            pass

    # ------------------------------------------------------------------
    # PDF resolution & thresholds
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_contract_pdf(
        run_directory_path: Path, context_packet: ContextPacket | None
    ) -> Path:
        """Resolve the primary contract PDF inside the run's input snapshot.

        Tries, in order: the snapshot directory recorded in the context packet,
        ``<run_dir>/input_snapshot/<input_file>``, and
        ``<run_dir>/<snapshot_basename>/<input_file>``. Never reads from
        ``data/bundles`` directly.
        """
        candidates: list[Path] = []
        input_file = "contract.pdf"
        if context_packet is not None:
            input_file = context_packet.input_file
            snapshot_dir = Path(context_packet.input_snapshot_directory)
            candidates.append(snapshot_dir / input_file)
            candidates.append(
                run_directory_path / snapshot_dir.name / input_file
            )
        candidates.append(
            run_directory_path / "input_snapshot" / input_file
        )

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        raise FileNotFoundError(
            "Could not locate the contract PDF in the run's input snapshot. "
            f"Tried: {', '.join(str(c) for c in candidates)}"
        )

    def _load_confidence_thresholds(
        self,
        run_directory_path: Path,
        context_packet: ContextPacket | None,
    ) -> tuple[float, float]:
        """Read confidence thresholds from the snapshot playbook, else defaults.

        Looks for ``confidence_thresholds.minimum_clause_confidence`` and
        ``confidence_thresholds.minimum_signature_confidence``. The playbook is
        read with a plain YAML loader (not the strict policy loader, which would
        reject the optional key).
        """
        minimum_clause = DEFAULT_MINIMUM_CLAUSE_CONFIDENCE
        minimum_signature = DEFAULT_MINIMUM_SIGNATURE_CONFIDENCE

        playbook_path = self._resolve_playbook(run_directory_path, context_packet)
        if playbook_path is None or not playbook_path.is_file():
            return minimum_clause, minimum_signature

        try:
            import yaml

            data = yaml.safe_load(playbook_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, ImportError):
            return minimum_clause, minimum_signature

        if not isinstance(data, dict):
            return minimum_clause, minimum_signature
        thresholds = data.get("confidence_thresholds")
        if not isinstance(thresholds, dict):
            return minimum_clause, minimum_signature

        clause_value = thresholds.get("minimum_clause_confidence")
        signature_value = thresholds.get("minimum_signature_confidence")
        if isinstance(clause_value, (int, float)):
            minimum_clause = float(clause_value)
        if isinstance(signature_value, (int, float)):
            minimum_signature = float(signature_value)
        return minimum_clause, minimum_signature

    @staticmethod
    def _resolve_playbook(
        run_directory_path: Path, context_packet: ContextPacket | None
    ) -> Path | None:
        """Locate playbook.yaml inside the run snapshot, if present."""
        if context_packet is not None:
            policy_path = context_packet.policy_files.get("playbook")
            if policy_path:
                candidate = run_directory_path / policy_path
                if candidate.is_file():
                    return candidate
            snapshot_dir = Path(context_packet.input_snapshot_directory)
            candidate = snapshot_dir / PLAYBOOK_SNAPSHOT_NAME
            if candidate.is_file():
                return candidate
        return run_directory_path / "input_snapshot" / PLAYBOOK_SNAPSHOT_NAME

    @staticmethod
    def _scenario_id_from_context(
        context_packet: ContextPacket | None,
    ) -> str:
        """Derive a scenario id from the contract id (e.g. contract_003 -> 3)."""
        if context_packet is not None:
            match = re.search(r"(\d+)", context_packet.contract_id)
            if match:
                return str(int(match.group(1)))
        return DEFAULT_SCENARIO_ID

    # ------------------------------------------------------------------
    # Shared helpers (unchanged behavior)
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_run_directory(run_directory: str) -> Path:
        if not run_directory or not run_directory.strip():
            raise FileNotFoundError("run_directory cannot be empty.")
        run_directory_path = Path(run_directory)
        if not run_directory_path.exists():
            raise FileNotFoundError(
                f"Run directory does not exist: {run_directory_path}"
            )
        if not run_directory_path.is_dir():
            raise FileNotFoundError(
                f"Run directory is not a directory: {run_directory_path}"
            )
        return run_directory_path

    @staticmethod
    def _load_context_packet(run_directory_path: Path) -> ContextPacket | None:
        packet_path = run_directory_path / CONTEXT_PACKET_FILENAME
        try:
            return ContextPacket.from_json_file(packet_path)
        except FileNotFoundError:
            logger.warning(
                "No %s found in %s; proceeding without intake context.",
                CONTEXT_PACKET_FILENAME,
                run_directory_path,
            )
            return None

    @staticmethod
    def _is_synthetic_mode() -> bool:
        env_mode = os.getenv(ENV_MODE_VARIABLE)
        mock_pipeline = os.getenv(MOCK_PIPELINE_VARIABLE)
        return (
            env_mode == ENV_MODE_TEST_VALUE
            or mock_pipeline == MOCK_PIPELINE_ENABLED_VALUE
        )

    @staticmethod
    def _persist_extracted_contract(
        extracted_contract: ExtractedContract, run_directory_path: Path
    ) -> None:
        artifact_path = run_directory_path / EXTRACTED_CONTRACT_FILENAME
        try:
            extracted_contract.to_json_file(artifact_path)
        except OSError as exc:
            logger.error(
                "Failed to write extracted contract %s: %s", artifact_path, exc
            )
            raise OSError(
                f"Failed to write extracted contract: {artifact_path} ({exc})"
            ) from exc
        logger.info("Persisted extracted contract to %s.", artifact_path)
