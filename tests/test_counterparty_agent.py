import json
from pathlib import Path

from app.agents.counterparty_agent import (
    COUNTERPARTY_NOT_FOUND_FLAG,
    DEFAULT_COUNTERPARTY_MISMATCH_THRESHOLD,
    HIGH_RISK_COUNTERPARTY_FLAG,
    MANUAL_REVIEW_REQUIRED_FLAG,
    NEW_COUNTERPARTY_FLAG,
    classify_counterparty_status,
    compare_counterparty_names,
    detect_counterparty_mismatch_from_run,
    read_counterparty_names,
)
from app.schemas.finding import Severity, UnifiedFinding
from app.schemas.normalized_counterparty import NormalizedCounterparty
from app.services.fuzzy_matcher import (
    FuzzyMatchResult,
    VendorRecord,
    normalize_company_name,
)


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


def create_match_result(
    vendor: VendorRecord,
    *,
    input_name: str,
    match_score: int,
    accepted: bool,
) -> FuzzyMatchResult:
    """
    Create a fuzzy-match result for classification tests.

    For unknown cases, the vendor remains the best_candidate,
    but matched_vendor is None because the threshold was not met.
    """

    return FuzzyMatchResult(
        input_counterparty_name=input_name,
        normalized_input_name=normalize_company_name(input_name),
        best_candidate=vendor,
        matched_vendor=vendor if accepted else None,
        match_score=match_score,
        threshold=85,
        match_status="matched" if accepted else "unknown",
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


def test_approved_vendor_returns_approved_status() -> None:
    """
    An accepted vendor with status approved must be classified
    as approved and retain its official vendor information.
    """

    vendor = VendorRecord(
        vendor_id="V001",
        vendor_name="Acme Services GmbH",
        country="Germany",
        status="approved",
        risk_level="low",
    )

    match_result = create_match_result(
        vendor,
        input_name="Acme Services Gmbh",
        match_score=97,
        accepted=True,
    )

    result = classify_counterparty_status(
        match_result,
        manifest_counterparty_name="Acme Services GmbH",
        extracted_counterparty_name="Acme Services GmbH",
    )

    assert isinstance(result, NormalizedCounterparty)

    assert result.matched_vendor_id == "V001"
    assert result.matched_vendor_name == "Acme Services GmbH"
    assert result.match_score == 97

    assert result.status == "approved"
    assert result.risk_level == "low"
    assert result.flags == []


def test_new_vendor_returns_new_status_and_flag() -> None:
    """
    An accepted vendor with status new must be classified as new
    and include the new_counterparty flag.
    """

    vendor = VendorRecord(
        vendor_id="V002",
        vendor_name="New Technology Ltd",
        country="United Kingdom",
        status="new",
        risk_level="medium",
    )

    match_result = create_match_result(
        vendor,
        input_name="New Technology Limited",
        match_score=100,
        accepted=True,
    )

    result = classify_counterparty_status(match_result)

    assert result.matched_vendor_id == "V002"
    assert result.matched_vendor_name == "New Technology Ltd"
    assert result.match_score == 100

    assert result.status == "new"
    assert result.risk_level == "medium"

    assert NEW_COUNTERPARTY_FLAG in result.flags
    assert result.flags == ["new_counterparty"]


def test_unknown_counterparty_returns_null_vendor_fields() -> None:
    """
    A best candidate below the matching threshold must not be
    treated as an accepted vendor.
    """

    closest_candidate = VendorRecord(
        vendor_id="V001",
        vendor_name="Acme Services GmbH",
        country="Germany",
        status="approved",
        risk_level="low",
    )

    match_result = create_match_result(
        closest_candidate,
        input_name="Zebra Quantum Mining PLC",
        match_score=28,
        accepted=False,
    )

    result = classify_counterparty_status(match_result)

    # The low-scoring candidate must not be exposed as a match.
    assert result.matched_vendor_id is None
    assert result.matched_vendor_name is None

    # The similarity score is still retained for audit.
    assert result.match_score == 28

    assert result.status == "unknown"
    assert result.risk_level == "high"

    assert COUNTERPARTY_NOT_FOUND_FLAG in result.flags
    assert MANUAL_REVIEW_REQUIRED_FLAG in result.flags


def test_high_risk_vendor_returns_high_risk_status_and_flag() -> None:
    """
    A matched vendor with risk_level high must override its
    approved/new status and be classified as high_risk.
    """

    vendor = VendorRecord(
        vendor_id="V003",
        vendor_name="Unknown Offshore LLC",
        country="HighRiskCountryX",
        status="new",
        risk_level="high",
    )

    match_result = create_match_result(
        vendor,
        input_name="Unknown Offshore LLC",
        match_score=100,
        accepted=True,
    )

    result = classify_counterparty_status(match_result)

    assert result.matched_vendor_id == "V003"
    assert result.matched_vendor_name == "Unknown Offshore LLC"

    # High risk takes priority over the vendor's "new" status.
    assert result.status == "high_risk"
    assert result.risk_level == "high"

    assert HIGH_RISK_COUNTERPARTY_FLAG in result.flags
    assert MANUAL_REVIEW_REQUIRED_FLAG in result.flags

def test_high_risk_vendor_status_triggers_high_risk_classification() -> None:
    """
    A blocked or sanctioned vendor must be high risk even when
    its risk_level field is not explicitly high.
    """

    vendor = VendorRecord(
        vendor_id="V004",
        vendor_name="Blocked Vendor Ltd",
        country="Germany",
        status="blocked",
        risk_level="medium",
    )

    match_result = create_match_result(
        vendor,
        input_name="Blocked Vendor Ltd",
        match_score=100,
        accepted=True,
    )

    result = classify_counterparty_status(match_result)

    assert result.status == "high_risk"
    assert result.risk_level == "high"
    assert HIGH_RISK_COUNTERPARTY_FLAG in result.flags