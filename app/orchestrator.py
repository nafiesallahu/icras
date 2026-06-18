"""ICRAS master pipeline orchestrator (file-based, deterministic).

Runs the full Intelligent Contract Review & Risk Analysis pipeline end-to-end on
a single contract bundle:

    Agent A (intake) -> Agent B (extraction) -> Agent C (counterparty)
        -> Agent D (validation) -> Agent E (risk scoring) -> Agent H (triage)

Design rules enforced here:

* Agent A is the ONLY run-directory creator. Every later stage operates on the
  directory Agent A returns; the orchestrator never creates a second run dir.
* The pipeline is file-based: each stage reads/writes JSON/CSV/MD artifacts in
  the run directory.
* Failures stop the pipeline; a failed :class:`PipelineResult` is returned and
  the failure is recorded in ``audit_log.md`` and ``metrics.json``.
* Agent H is invoked only through :func:`app.agents.triage_agent.run_triage_agent`
  so the current deterministic mock can be swapped for the real Agent H without
  any orchestrator changes.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from app.agents.counterparty_agent import run_counterparty_agent
from app.agents.extraction_agent import ExtractionAgent
from app.agents.intake_agent import IntakeAgent, IntakeError
from app.agents.triage_agent import run_triage_agent
from app.agents.validation_agent import ValidationAgent
from app.schemas.clause_analysis import ClauseAnalysis
from app.schemas.context_packet import ContextPacket
from app.schemas.extracted_contract import ExtractedContract
from app.schemas.validation_result import ValidationResult
from app.services.policy_loader import load_playbook
from app.services.rules_engine import RulesEngine

# Files that must be present in a bundle before the pipeline starts.
REQUIRED_BUNDLE_FILES = (
    "contract.pdf",
    "manifest.yaml",
    "playbook.yaml",
    "approval_policy.yaml",
    "vendor_master.csv",
    "jurisdiction_rules.yaml",
)

# Final artifacts the orchestrator reports/checks on a completed run.
EXPECTED_FINAL_ARTIFACTS = (
    "context_packet.json",
    "evidence_index.json",
    "extracted_contract.json",
    "normalized_counterparty.json",
    "validation_result.json",
    "clause_analysis.json",
    "obligations.csv",
    "exceptions.md",
    "approval_packet.json",
    "posting_payload.json",
    "audit_log.md",
    "metrics.json",
)

AUDIT_LOG_FILENAME = "audit_log.md"
METRICS_FILENAME = "metrics.json"
EXTRACTED_CONTRACT_FILENAME = "extracted_contract.json"
VALIDATION_RESULT_FILENAME = "validation_result.json"
CLAUSE_ANALYSIS_FILENAME = "clause_analysis.json"
OBLIGATIONS_FILENAME = "obligations.csv"
PLAYBOOK_SNAPSHOT = "input_snapshot/playbook.yaml"

OBLIGATIONS_MODE_FALLBACK = "minimal_orchestrator_fallback"
OBLIGATIONS_HEADER = (
    "obligation_id,contract_id,clause_id,description,owner,due_date"
)


@dataclass
class PipelineResult:
    """Outcome of a single pipeline run."""

    run_id: str
    contract_id: str
    bundle_id: str
    run_directory: str
    status: str
    final_decision: str | None = None
    overall_risk: str | None = None
    agents_completed: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    error_message: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"


class _StageError(Exception):
    """Internal error raised when a pipeline stage fails or its outputs are bad."""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        self.message = message
        super().__init__(message)


# ----------------------------------------------------------------------------
# Shared artifact helpers
# ----------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _append_audit(run_dir: Path, lines: list[str]) -> None:
    audit_path = run_dir / AUDIT_LOG_FILENAME
    existing = (
        audit_path.read_text(encoding="utf-8") if audit_path.exists() else ""
    )
    audit_path.write_text(existing + "\n".join(lines), encoding="utf-8")


def _load_metrics(run_dir: Path) -> dict[str, Any]:
    metrics_path = run_dir / METRICS_FILENAME
    if not metrics_path.exists():
        return {}
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_metrics(run_dir: Path, metrics: dict[str, Any]) -> None:
    (run_dir / METRICS_FILENAME).write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )


def _append_agent_completed(run_dir: Path, agent_name: str) -> None:
    """Append an agent to the flat ``agents_completed`` list (dedup)."""
    metrics = _load_metrics(run_dir)
    completed = metrics.get("agents_completed")
    if not isinstance(completed, list):
        completed = []
    if agent_name not in completed:
        completed.append(agent_name)
    metrics["agents_completed"] = completed
    _write_metrics(run_dir, metrics)


def _check_files_exist(run_dir: Path, names: list[str], stage: str) -> None:
    missing = [name for name in names if not (run_dir / name).exists()]
    if missing:
        raise _StageError(
            stage,
            f"{stage} did not produce required artifact(s): "
            + ", ".join(missing),
        )


def _check_json_valid(run_dir: Path, names: list[str], stage: str) -> None:
    for name in names:
        path = run_dir / name
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise _StageError(
                stage, f"{stage} produced invalid JSON in {name}: {exc}"
            ) from exc


def _scenario_id_from_bundle(bundle_path: Path) -> str | None:
    """Derive ``scenario_NN`` from a bundle folder name, else ``None``."""
    match = re.match(r"(scenario_\d+)", bundle_path.name)
    return match.group(1) if match else None


# ----------------------------------------------------------------------------
# Bundle validation
# ----------------------------------------------------------------------------
def validate_bundle(bundle_path: str | Path) -> list[str]:
    """Return a list of validation errors for a bundle (empty when valid)."""
    errors: list[str] = []
    bundle_dir = Path(bundle_path)

    if not bundle_dir.exists():
        return [f"Bundle path does not exist: {bundle_dir}"]
    if not bundle_dir.is_dir():
        return [f"Bundle path is not a directory: {bundle_dir}"]

    for required in REQUIRED_BUNDLE_FILES:
        if not (bundle_dir / required).is_file():
            errors.append(f"missing required bundle file: {required}")
    return errors


# ----------------------------------------------------------------------------
# Agent D / E run-directory wrappers
# ----------------------------------------------------------------------------
def _read_contract_type(run_dir: Path) -> str | None:
    """Best-effort read of ``contract_type`` from the run's context packet."""
    context_path = run_dir / "context_packet.json"
    if not context_path.is_file():
        return None
    try:
        data = json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    contract_type = data.get("contract_type")
    if isinstance(contract_type, str) and contract_type.strip():
        return contract_type
    return None


def run_validation_agent(run_directory: str | Path) -> ValidationResult:
    """Agent D wrapper: validate the extracted contract and persist the result."""
    run_dir = Path(run_directory)

    contract = ExtractedContract.from_json_file(
        run_dir / EXTRACTED_CONTRACT_FILENAME
    )

    # Surface the contract type recorded at intake so Agent D can apply
    # contract-type-aware mandatory-field rules (e.g. NDAs do not require
    # payment terms or a liability cap). Falls back to clause-based inference
    # inside the agent when the hint is unavailable.
    contract_type = _read_contract_type(run_dir)

    agent = ValidationAgent()
    # Pin the run-snapshot playbook so validation uses the exact policy captured
    # for this run; fall back silently to the agent's default playbook.
    snapshot_playbook = run_dir / PLAYBOOK_SNAPSHOT
    if snapshot_playbook.is_file():
        try:
            agent.playbook = load_playbook(snapshot_playbook)
        except Exception:  # noqa: BLE001 - keep default playbook on any issue
            pass

    result = agent.validate(contract, contract_type=contract_type)
    result.to_json_file(run_dir / VALIDATION_RESULT_FILENAME)

    _append_audit(
        run_dir,
        [
            "",
            "## Step 4: Validation",
            "- Agent: D",
            f"- Validation status: {result.validation_status.value}",
            f"- Findings: {len(result.findings)}",
            f"- Generated artifacts: {VALIDATION_RESULT_FILENAME}",
            "- Status: validation_completed",
            "",
        ],
    )
    _append_agent_completed(run_dir, "validation")
    return result


def _write_minimal_obligations(
    run_dir: Path, validation_result: ValidationResult
) -> None:
    """Write a deterministic, minimal obligations.csv fallback."""
    rows = [OBLIGATIONS_HEADER]
    contract_id = validation_result.contract_id
    for index, finding in enumerate(validation_result.findings, start=1):
        description = (finding.recommendation or finding.message or "").replace(
            ",", ";"
        )
        clause_id = finding.clause_id or ""
        rows.append(
            f"OBL-{index:03d},{contract_id},{clause_id},{description},,"
        )
    (run_dir / OBLIGATIONS_FILENAME).write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )


def run_risk_scoring_agent(run_directory: str | Path) -> ClauseAnalysis:
    """Agent E wrapper: score the validation result and persist clause analysis."""
    run_dir = Path(run_directory)

    validation_result = ValidationResult.from_json_file(
        run_dir / VALIDATION_RESULT_FILENAME
    )

    engine = RulesEngine()
    clause_analysis = engine.score_validation_result(validation_result)
    engine.write_clause_analysis(run_dir, clause_analysis)

    _write_minimal_obligations(run_dir, validation_result)

    _append_audit(
        run_dir,
        [
            "",
            "## Step 5: Risk Scoring",
            "- Agent: E",
            f"- Total score: {clause_analysis.total_score}",
            f"- Risk tier: {clause_analysis.risk_tier.value}",
            f"- Scored findings: {len(clause_analysis.findings)}",
            f"- obligations_mode: {OBLIGATIONS_MODE_FALLBACK}",
            f"- Generated artifacts: {CLAUSE_ANALYSIS_FILENAME}, "
            f"{OBLIGATIONS_FILENAME}",
            "- Status: risk_scoring_completed",
            "",
        ],
    )
    _append_agent_completed(run_dir, "risk_scoring")

    metrics = _load_metrics(run_dir)
    metrics["obligations_mode"] = OBLIGATIONS_MODE_FALLBACK
    _write_metrics(run_dir, metrics)

    return clause_analysis


# ----------------------------------------------------------------------------
# Failure recording
# ----------------------------------------------------------------------------
def _record_failure(
    run_dir: Path | None,
    *,
    stage: str,
    message: str,
    run_id: str,
    contract_id: str,
    bundle_id: str,
    started_perf: float,
) -> PipelineResult:
    status = f"failed_at_{stage}"
    if run_dir is not None:
        try:
            _append_audit(
                run_dir,
                [
                    "",
                    "## Pipeline Failure",
                    f"- Failed agent: {stage}",
                    f"- Error: {message}",
                    f"- Status: {status}",
                    "",
                ],
            )
        except OSError:
            pass
        try:
            metrics = _load_metrics(run_dir)
            metrics["status"] = status
            metrics["failed_agent"] = stage
            metrics["error_message"] = message
            metrics["finished_at"] = _now_iso()
            metrics["duration_seconds"] = round(
                time.perf_counter() - started_perf, 3
            )
            metrics["deterministic_run"] = True
            _write_metrics(run_dir, metrics)
        except OSError:
            pass

    return PipelineResult(
        run_id=run_id,
        contract_id=contract_id,
        bundle_id=bundle_id,
        run_directory=str(run_dir) if run_dir is not None else "",
        status=status,
        error_message=message,
        artifacts=_collect_artifacts(run_dir) if run_dir is not None else {},
    )


def _collect_artifacts(run_dir: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    snapshot = run_dir / "input_snapshot"
    if snapshot.is_dir():
        artifacts["input_snapshot"] = str(snapshot)
    for name in EXPECTED_FINAL_ARTIFACTS:
        path = run_dir / name
        if path.exists():
            artifacts[name] = str(path)
    return artifacts


# ----------------------------------------------------------------------------
# Pipeline entrypoint
# ----------------------------------------------------------------------------
def run_pipeline(
    bundle_path: str | Path, runs_root: str | Path = "runs"
) -> PipelineResult:
    """Execute the full ICRAS pipeline on a single bundle.

    Args:
        bundle_path: Path to the source contract bundle directory.
        runs_root: Base directory under which Agent A creates the run directory.

    Returns:
        A :class:`PipelineResult` describing the outcome. On failure the result
        carries ``status="failed_at_<stage>"`` and an ``error_message``.
    """
    started_perf = time.perf_counter()
    bundle_dir = Path(bundle_path)

    # Step 0: validate the bundle before touching any agent.
    bundle_errors = validate_bundle(bundle_dir)
    if bundle_errors:
        return PipelineResult(
            run_id="",
            contract_id="",
            bundle_id="",
            run_directory="",
            status="failed_at_validation_input",
            error_message="; ".join(bundle_errors),
        )

    scenario_id = _scenario_id_from_bundle(bundle_dir)

    # Step 1: Agent A (intake) — the only run-directory creator.
    try:
        context_packet: ContextPacket = IntakeAgent(runs_root=runs_root).run(
            str(bundle_dir)
        )
    except IntakeError as exc:
        return PipelineResult(
            run_id="",
            contract_id="",
            bundle_id="",
            run_directory="",
            status="failed_at_intake",
            error_message=str(exc),
        )

    run_dir = Path(context_packet.run_directory)
    run_id = context_packet.run_id
    contract_id = context_packet.contract_id
    bundle_id = context_packet.bundle_id

    try:
        _check_files_exist(
            run_dir,
            [
                "context_packet.json",
                "evidence_index.json",
                AUDIT_LOG_FILENAME,
                METRICS_FILENAME,
            ],
            "intake",
        )

        # Step 2: Agent B (extraction).
        ExtractionAgent().run(str(run_dir), scenario_id=scenario_id)
        _check_files_exist(
            run_dir,
            [EXTRACTED_CONTRACT_FILENAME, "evidence_index.json"],
            "extraction",
        )
        _check_json_valid(
            run_dir,
            [EXTRACTED_CONTRACT_FILENAME, "evidence_index.json"],
            "extraction",
        )

        # Step 3: Agent C (counterparty resolution).
        run_counterparty_agent(run_dir=run_dir)
        _check_files_exist(
            run_dir, ["normalized_counterparty.json"], "counterparty"
        )
        _check_json_valid(
            run_dir, ["normalized_counterparty.json"], "counterparty"
        )
        _append_agent_completed(run_dir, "counterparty")

        # Step 4: Agent D (validation).
        run_validation_agent(run_dir)
        _check_files_exist(run_dir, [VALIDATION_RESULT_FILENAME], "validation")
        _check_json_valid(run_dir, [VALIDATION_RESULT_FILENAME], "validation")

        # Step 5: Agent E (risk scoring).
        run_risk_scoring_agent(run_dir)
        _check_files_exist(
            run_dir,
            [CLAUSE_ANALYSIS_FILENAME, OBLIGATIONS_FILENAME],
            "risk_scoring",
        )
        _check_json_valid(run_dir, [CLAUSE_ANALYSIS_FILENAME], "risk_scoring")

        # Step 6: Agent H boundary (deterministic mock triage).
        triage = run_triage_agent(run_dir)
        _check_files_exist(
            run_dir,
            ["exceptions.md", "approval_packet.json", "posting_payload.json"],
            "triage",
        )
        _check_json_valid(
            run_dir,
            ["approval_packet.json", "posting_payload.json"],
            "triage",
        )
        _append_agent_completed(run_dir, "triage")
    except _StageError as exc:
        return _record_failure(
            run_dir,
            stage=exc.stage,
            message=exc.message,
            run_id=run_id,
            contract_id=contract_id,
            bundle_id=bundle_id,
            started_perf=started_perf,
        )
    except Exception as exc:  # noqa: BLE001 - any agent error fails the pipeline
        stage = _infer_stage(run_dir)
        return _record_failure(
            run_dir,
            stage=stage,
            message=f"{type(exc).__name__}: {exc}",
            run_id=run_id,
            contract_id=contract_id,
            bundle_id=bundle_id,
            started_perf=started_perf,
        )

    overall_risk = str(triage.get("overall_risk")) if triage else None
    final_decision = str(triage.get("final_decision")) if triage else None

    # Finalize metrics.
    metrics = _load_metrics(run_dir)
    metrics["run_id"] = run_id
    metrics["contract_id"] = contract_id
    metrics.setdefault("started_at", context_packet.received_timestamp)
    metrics["finished_at"] = _now_iso()
    metrics["duration_seconds"] = round(time.perf_counter() - started_perf, 3)
    metrics["status"] = "completed"
    metrics["overall_risk"] = overall_risk
    metrics["final_decision"] = final_decision
    metrics["deterministic_run"] = True
    completed = metrics.get("agents_completed")
    if not isinstance(completed, list):
        completed = []
    for agent_name in ("intake", "extraction", "counterparty", "validation",
                       "risk_scoring", "triage"):
        if agent_name not in completed:
            completed.append(agent_name)
    metrics["agents_completed"] = completed
    _write_metrics(run_dir, metrics)

    return PipelineResult(
        run_id=run_id,
        contract_id=contract_id,
        bundle_id=bundle_id,
        run_directory=str(run_dir),
        status="completed",
        final_decision=final_decision,
        overall_risk=overall_risk,
        agents_completed=completed,
        artifacts=_collect_artifacts(run_dir),
    )


def _infer_stage(run_dir: Path) -> str:
    """Best-effort stage inference for an unexpected exception."""
    ordered = [
        (EXTRACTED_CONTRACT_FILENAME, "extraction"),
        ("normalized_counterparty.json", "counterparty"),
        (VALIDATION_RESULT_FILENAME, "validation"),
        (CLAUSE_ANALYSIS_FILENAME, "risk_scoring"),
        ("approval_packet.json", "triage"),
    ]
    # The first missing artifact, in pipeline order, names the failing stage.
    for filename, stage in ordered:
        if not (run_dir / filename).exists():
            return stage
    return "triage"
