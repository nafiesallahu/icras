"""Synthetic fallback extraction engine for ICRAS Agent B.

During testing and offline development the live LLM/PDF extraction stack is
either unavailable or undesirable (cost, determinism, network access). The
:class:`SyntheticFallbackEngine` replaces that stack with a deterministic
source of truth: a curated set of pre-validated fixtures stored on disk at
``data/synthetic_fixtures/scenario_[xx]_extracted_contract.json``.

Given a scenario identifier the engine resolves the matching fixture file,
reads it safely, and validates it against the canonical
:class:`~app.schemas.extracted_contract.ExtractedContract` schema so that
callers always receive the exact same artifact shape that the live extractor
would produce.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.schemas.extracted_contract import ExtractedContract

logger = logging.getLogger(__name__)

# Root directory under which every synthetic scenario fixture lives. Kept
# relative to the current working directory to mirror the convention used by
# the intake agent's ``runs/`` root.
FIXTURES_ROOT = Path("data/synthetic_fixtures")

# Suffix shared by every extracted-contract fixture file. The normalized
# scenario token is prepended to build the full filename.
FIXTURE_FILENAME_SUFFIX = "_extracted_contract.json"

# Matches a bare zero-or-more-digit scenario index, optionally already prefixed
# with ``scenario_`` (case-insensitive). The numeric capture group is what we
# normalize and zero-pad.
_SCENARIO_ID_PATTERN = re.compile(r"^(?:scenario[_-]?)?(\d+)$", re.IGNORECASE)

# Width to which the numeric scenario index is zero-padded, e.g. ``1`` -> ``01``
# so it matches the physical ``scenario_01_...`` filenames.
_SCENARIO_INDEX_WIDTH = 2


class SyntheticFallbackEngine:
    """Resolves scenario identifiers to validated :class:`ExtractedContract` objects.

    The engine is stateless aside from the configured fixtures root, so a single
    instance can be reused safely across many lookups. It performs no network or
    LLM calls; every result is loaded directly from a local fixture file.
    """

    def __init__(self, fixtures_root: str | Path = FIXTURES_ROOT) -> None:
        """Initialize the engine.

        Args:
            fixtures_root: Base directory containing the
                ``scenario_[xx]_extracted_contract.json`` fixture files. Defaults
                to :data:`FIXTURES_ROOT`.
        """
        self._fixtures_root = Path(fixtures_root)

    def get_scenario_fixture(self, scenario_id: str) -> ExtractedContract:
        """Load and validate the fixture for a single scenario.

        The supplied ``scenario_id`` is normalized so that callers may pass any
        of the following equivalent forms: ``"scenario_01"``, ``"scenario-1"``,
        ``"01"``, or ``"1"``. All resolve to the physical fixture file
        ``scenario_01_extracted_contract.json``.

        Args:
            scenario_id: The scenario identifier to resolve. Must contain a
                numeric index, optionally prefixed with ``scenario_``.

        Returns:
            The validated :class:`ExtractedContract` parsed from the fixture.

        Raises:
            ValueError: If ``scenario_id`` is empty or does not contain a
                recognizable numeric scenario index.
            FileNotFoundError: If no fixture file exists for the resolved
                scenario.
            OSError: If the fixture file exists but cannot be read.
            json.JSONDecodeError: If the fixture file does not contain valid
                JSON.
            ValidationError: If the fixture JSON does not satisfy the
                :class:`ExtractedContract` schema.
        """
        normalized_id = self._normalize_scenario_id(scenario_id)
        fixture_path = self._resolve_fixture_path(normalized_id)

        logger.debug(
            "Resolving synthetic fixture for scenario '%s' -> %s",
            scenario_id,
            fixture_path,
        )

        raw_json = self._read_fixture_text(fixture_path)
        parsed_json = self._parse_fixture_json(raw_json, fixture_path)
        contract = self._validate_fixture(parsed_json, fixture_path)

        logger.info(
            "Loaded synthetic fixture '%s' (contract_id=%s, clauses=%d).",
            normalized_id,
            contract.contract_id,
            len(contract.clauses),
        )
        return contract

    def _resolve_fixture_path(self, normalized_id: str) -> Path:
        """Build the absolute fixture path for a normalized scenario id.

        Args:
            normalized_id: A normalized identifier such as ``"scenario_01"``.

        Returns:
            The :class:`~pathlib.Path` to the expected fixture file.
        """
        return self._fixtures_root / f"{normalized_id}{FIXTURE_FILENAME_SUFFIX}"

    @staticmethod
    def _normalize_scenario_id(scenario_id: str) -> str:
        """Normalize an arbitrary scenario identifier to ``scenario_XX`` form.

        Args:
            scenario_id: The raw identifier provided by the caller.

        Returns:
            A normalized identifier with a zero-padded numeric index, e.g.
            ``"scenario_01"``.

        Raises:
            ValueError: If the identifier is empty, not a string, or does not
                contain a recognizable numeric scenario index.
        """
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            raise ValueError("scenario_id must be a non-empty string.")

        match = _SCENARIO_ID_PATTERN.match(scenario_id.strip())
        if match is None:
            raise ValueError(
                f"Unrecognized scenario_id '{scenario_id}'. Expected a numeric "
                "index optionally prefixed with 'scenario_', for example "
                "'scenario_01', '01', or '1'."
            )

        index = int(match.group(1))
        padded_index = str(index).zfill(_SCENARIO_INDEX_WIDTH)
        return f"scenario_{padded_index}"

    @staticmethod
    def _read_fixture_text(fixture_path: Path) -> str:
        """Read the raw text content of a fixture file.

        Args:
            fixture_path: Path to the fixture file.

        Returns:
            The decoded UTF-8 contents of the file.

        Raises:
            FileNotFoundError: If the fixture file does not exist.
            OSError: If the file exists but cannot be read.
        """
        try:
            return fixture_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            logger.error("Synthetic fixture not found: %s", fixture_path)
            raise FileNotFoundError(
                f"Synthetic fixture not found: {fixture_path}. Ensure the "
                "scenario fixture exists under the fixtures root."
            ) from exc
        except OSError as exc:
            logger.error(
                "Failed to read synthetic fixture %s: %s", fixture_path, exc
            )
            raise OSError(
                f"Failed to read synthetic fixture: {fixture_path} ({exc})"
            ) from exc

    @staticmethod
    def _parse_fixture_json(raw_json: str, fixture_path: Path) -> Any:
        """Parse fixture text into a Python object.

        Args:
            raw_json: The raw JSON text read from disk.
            fixture_path: Path the text was read from, used for error context.

        Returns:
            The decoded JSON document (typically a ``dict``).

        Raises:
            json.JSONDecodeError: If the text is not valid JSON.
        """
        try:
            return json.loads(raw_json)
        except json.JSONDecodeError as exc:
            logger.error(
                "Synthetic fixture %s contains invalid JSON: %s",
                fixture_path,
                exc,
            )
            raise

    @staticmethod
    def _validate_fixture(
        parsed_json: Any, fixture_path: Path
    ) -> ExtractedContract:
        """Validate a parsed JSON document against the contract schema.

        Args:
            parsed_json: The decoded JSON document.
            fixture_path: Path the document came from, used for error context.

        Returns:
            The validated :class:`ExtractedContract` instance.

        Raises:
            ValidationError: If the document violates the schema.
        """
        try:
            return ExtractedContract.model_validate(parsed_json)
        except ValidationError as exc:
            logger.error(
                "Synthetic fixture %s failed ExtractedContract validation: %s",
                fixture_path,
                exc,
            )
            raise
