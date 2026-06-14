"""Extraction Agent (Agent B) for the ICRAS pipeline.

Agent B is the orchestrator responsible for producing the
``extracted_contract.json`` artifact that downstream agents (such as the
validation engine) consume. It decides, per run, whether to route extraction
through the live LLM/PDF stack or to intercept the request with a deterministic
synthetic fallback.

Routing is controlled by environment variables so the same code path can be
exercised in tests and CI without any live model access:

* ``ENV_MODE == "test"`` forces synthetic fallback.
* ``MOCK_PIPELINE == "True"`` also forces synthetic fallback.

In every mode the agent guarantees that the produced artifact strictly
validates as an :class:`~app.schemas.extracted_contract.ExtractedContract` and
is persisted into the per-run working directory created by the intake agent.
"""

import logging
import os
from pathlib import Path

from app.schemas.context_packet import ContextPacket
from app.schemas.extracted_contract import ExtractedContract
from app.services.synthetic_extractor import SyntheticFallbackEngine

logger = logging.getLogger(__name__)

# Filename written by the intake agent into each run directory; read here to
# preserve execution-context continuity across the pipeline.
CONTEXT_PACKET_FILENAME = "context_packet.json"

# Filename of the artifact Agent B produces. Downstream agents discover the
# extraction output by this exact name inside the run directory.
EXTRACTED_CONTRACT_FILENAME = "extracted_contract.json"

# Scenario used when the caller does not specify one while in synthetic mode.
# scenario_01 is the canonical, fully-populated baseline fixture.
DEFAULT_SCENARIO_ID = "scenario_01"

# Environment variable names and the sentinel values that activate synthetic
# fallback (bypassing all live LLM logic).
ENV_MODE_VARIABLE = "ENV_MODE"
ENV_MODE_TEST_VALUE = "test"
MOCK_PIPELINE_VARIABLE = "MOCK_PIPELINE"
MOCK_PIPELINE_ENABLED_VALUE = "True"


class ExtractionAgent:
    """Orchestrates contract extraction for a single run.

    The agent is stateless between runs aside from its injected synthetic
    engine, so a single instance can be reused for many run directories.

    Args:
        synthetic_engine: Engine used to resolve synthetic fixtures. A fresh
            :class:`SyntheticFallbackEngine` is created when not supplied,
            which is the common case; injection exists primarily for testing.
    """

    def __init__(
        self, synthetic_engine: SyntheticFallbackEngine | None = None
    ) -> None:
        self._synthetic_engine = synthetic_engine or SyntheticFallbackEngine()

    def run(
        self, run_directory: str, scenario_id: str | None = None
    ) -> ExtractedContract:
        """Execute extraction for a run and persist the resulting artifact.

        The method performs the following sequential business logic:

        1. Read ``context_packet.json`` from ``run_directory`` to maintain
           execution-context continuity. A missing packet is logged and
           tolerated, since the extraction itself does not strictly depend on
           it.
        2. Determine the routing mode from the environment. When
           ``ENV_MODE == "test"`` or ``MOCK_PIPELINE == "True"`` the live LLM
           logic is bypassed entirely in favor of the synthetic fallback.
        3. In synthetic mode, resolve the fixture for ``scenario_id`` (falling
           back to :data:`DEFAULT_SCENARIO_ID` when none is provided).
        4. Confirm the produced object is an :class:`ExtractedContract`.
        5. Persist the artifact to ``extracted_contract.json`` inside
           ``run_directory`` for downstream discovery.
        6. Return the validated :class:`ExtractedContract`.

        Args:
            run_directory: Path to the isolated per-run working directory
                created by the intake agent.
            scenario_id: Optional scenario identifier used when synthetic
                fallback is active. Defaults to :data:`DEFAULT_SCENARIO_ID`.

        Returns:
            The validated :class:`ExtractedContract` produced for this run.

        Raises:
            FileNotFoundError: If ``run_directory`` does not exist, or if a
                requested synthetic fixture cannot be found.
            NotImplementedError: If live extraction is requested (neither
                synthetic environment flag is set); the live LLM path is not
                yet wired into this orchestrator shell.
            ValueError: If ``scenario_id`` cannot be normalized to a known
                scenario.
            OSError: If the produced artifact cannot be written to disk.
            ValidationError: If a synthetic fixture fails schema validation.
        """
        run_directory_path = self._resolve_run_directory(run_directory)

        context_packet = self._load_context_packet(run_directory_path)
        if context_packet is not None:
            logger.debug(
                "Extraction running within context for contract_id=%s.",
                context_packet.contract_id,
            )

        if self._is_synthetic_mode():
            effective_scenario_id = scenario_id or DEFAULT_SCENARIO_ID
            logger.info(
                "Synthetic mode active; bypassing live LLM extraction and "
                "loading scenario '%s'.",
                effective_scenario_id,
            )
            extracted_contract = self._synthetic_engine.get_scenario_fixture(
                effective_scenario_id
            )
        else:
            extracted_contract = self._run_live_extraction(
                run_directory_path, context_packet
            )

        if not isinstance(extracted_contract, ExtractedContract):
            raise TypeError(
                "Extraction did not produce an ExtractedContract instance; "
                f"got {type(extracted_contract).__name__}."
            )

        self._persist_extracted_contract(extracted_contract, run_directory_path)

        return extracted_contract

    @staticmethod
    def _resolve_run_directory(run_directory: str) -> Path:
        """Validate and resolve the run directory path.

        Args:
            run_directory: Path to the per-run working directory.

        Returns:
            The validated run directory as a :class:`~pathlib.Path`.

        Raises:
            FileNotFoundError: If the path is empty, missing, or not a
                directory.
        """
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
        """Load the intake context packet if present.

        The packet provides run lineage for logging and continuity, but the
        extraction step itself does not strictly require it. A missing packet
        is therefore tolerated and reported rather than fatal.

        Args:
            run_directory_path: The resolved run directory.

        Returns:
            The validated :class:`ContextPacket`, or ``None`` if no packet file
            exists in the run directory.
        """
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
        """Determine whether synthetic fallback should intercept extraction.

        Returns:
            ``True`` if either ``ENV_MODE == "test"`` or
            ``MOCK_PIPELINE == "True"``; otherwise ``False``.
        """
        env_mode = os.getenv(ENV_MODE_VARIABLE)
        mock_pipeline = os.getenv(MOCK_PIPELINE_VARIABLE)
        return (
            env_mode == ENV_MODE_TEST_VALUE
            or mock_pipeline == MOCK_PIPELINE_ENABLED_VALUE
        )

    @staticmethod
    def _run_live_extraction(
        run_directory_path: Path, context_packet: ContextPacket | None
    ) -> ExtractedContract:
        """Execute the live LLM/PDF extraction path.

        This orchestrator shell intentionally does not yet wire up the live
        extraction stack. Until it does, requesting live extraction is an
        explicit, loud failure rather than a silent fallback, so misconfigured
        environments are caught immediately.

        Args:
            run_directory_path: The resolved run directory.
            context_packet: The intake context, if available.

        Raises:
            NotImplementedError: Always, until live extraction is implemented.
        """
        logger.error(
            "Live extraction requested for %s but the live LLM path is not "
            "implemented. Set ENV_MODE=test or MOCK_PIPELINE=True to use the "
            "synthetic fallback.",
            run_directory_path,
        )
        raise NotImplementedError(
            "Live LLM extraction is not yet implemented in the ExtractionAgent "
            "orchestrator shell. Enable synthetic fallback via ENV_MODE=test or "
            "MOCK_PIPELINE=True."
        )

    @staticmethod
    def _persist_extracted_contract(
        extracted_contract: ExtractedContract, run_directory_path: Path
    ) -> None:
        """Serialize the extracted contract into the run directory.

        Args:
            extracted_contract: The validated contract to persist.
            run_directory_path: The directory into which
                ``extracted_contract.json`` is written.

        Raises:
            OSError: If the artifact cannot be written (for example, due to
                permission issues).
        """
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
