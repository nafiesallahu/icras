import json
from pathlib import Path

from app.agents.counterparty_agent import (
    DEFAULT_COUNTERPARTY_MISMATCH_THRESHOLD,
    compare_counterparty_names,
    detect_counterparty_mismatch_from_run,
    read_counterparty_names,
)
from app.schemas.finding import Severity, UnifiedFinding


def write_json(
    path: Path,
    payload: dict,
) -> None:
    """
    Write a temporary JSON artifact for an Agent C unit test.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )


def test_matching_counterparty_names_do_not_create_finding() -> None:
    """
    Identical names must not produce a mismatch finding.
    """

    result = compare_counterparty_names(
        "Acme Services GmbH",
        "Acme Services GmbH",
    )

    assert result.similarity_score == 100

    assert result.mismatch_threshold == DEFAULT_COUNTERPARTY_MISMATCH_THRESHOLD

    assert result.is_mismatch is False
    assert result.finding is None


def test_minor_counterparty_variations_do_not_create_finding() -> None:
    """
    Casing, surrounding spaces, and punctuation differences
    must not cause a false mismatch.
    """

    result = compare_counterparty_names(
        "  ACME Services GmbH.  ",
        "Acme Services Gmbh",
    )

    # Both names normalize to "acme services gmbh".
    assert result.similarity_score == 100
    assert result.is_mismatch is False
    assert result.finding is None


def test_material_counterparty_mismatch_creates_finding() -> None:
    """
    Materially different company names must produce
    a counterparty_mismatch finding.
    """

    result = compare_counterparty_names(
        "Acme Services GmbH",
        "Global Data Ltd",
    )

    assert result.similarity_score < (DEFAULT_COUNTERPARTY_MISMATCH_THRESHOLD)

    assert result.is_mismatch is True
    assert result.finding is not None
    assert isinstance(result.finding, UnifiedFinding)

    # Verify the finding follows the current unified structure.
    assert result.finding.finding_id == "CP-MISMATCH-001"
    assert result.finding.field == "counterparty"
    assert result.finding.severity == Severity.HIGH

    assert "counterparty_mismatch" in result.finding.message

    assert result.finding.policy_rule == "counterparty.names_must_match"

    assert result.finding.expected_value == "Acme Services GmbH"

    assert result.finding.actual_value == "Global Data Ltd"

    assert result.finding.recommendation == (
        "Manually review the contract and manifest "
        "counterparty names before approval."
    )


def test_mismatch_severity_can_be_configured_as_medium() -> None:
    """
    Configuration may classify the material mismatch as medium
    instead of high.
    """

    result = compare_counterparty_names(
        "Acme Services GmbH",
        "Global Data Ltd",
        mismatch_severity=Severity.MEDIUM,
    )

    assert result.is_mismatch is True
    assert result.finding is not None
    assert result.finding.severity == Severity.MEDIUM


def test_agent_reads_counterparty_names_from_run_files(
    tmp_path: Path,
) -> None:
    """
    Agent C must read the manifest counterparty from context_packet.json
    and the extracted counterparty from extracted_contract.json.
    """

    run_dir = tmp_path / "runs" / "run_001"

    write_json(
        run_dir / "context_packet.json",
        {"counterparty_name_from_manifest": ("Acme Services GmbH")},
    )

    write_json(
        run_dir / "extracted_contract.json",
        {"parties": {"counterparty_name": "Global Data Ltd"}},
    )

    (
        manifest_counterparty_name,
        extracted_counterparty_name,
    ) = read_counterparty_names(run_dir)

    assert manifest_counterparty_name == "Acme Services GmbH"

    assert extracted_counterparty_name == "Global Data Ltd"

    # Confirm the values read from disk are used by
    # the actual mismatch detector.
    result = detect_counterparty_mismatch_from_run(run_dir)

    assert result.manifest_counterparty_name == ("Acme Services GmbH")

    assert result.extracted_counterparty_name == ("Global Data Ltd")

    assert result.is_mismatch is True
    assert result.finding is not None
