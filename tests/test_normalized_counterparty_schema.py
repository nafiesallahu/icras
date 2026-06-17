import pytest
from pydantic import ValidationError

# Import the real finding model used by this project.
from app.schemas.finding import Severity, UnifiedFinding

# Import the Agent C output schema.
from app.schemas.normalized_counterparty import NormalizedCounterparty


def valid_counterparty_payload() -> dict:
    """
    Return a valid base payload reused by multiple tests.

    flags and findings are intentionally omitted so the tests can verify
    that both fields default to separate empty lists.
    """
    return {
        "input_counterparty_name": "Acme Services Gmbh",
        "manifest_counterparty_name": "Acme Services GmbH",
        "extracted_counterparty_name": "Acme Services GmbH",
        "matched_vendor_id": "V001",
        "matched_vendor_name": "Acme Services GmbH",
        "match_score": 97,
        "status": "approved",
        "risk_level": "low",
    }


def test_normalized_counterparty_accepts_valid_data() -> None:
    """
    A complete and correctly typed Agent C result must pass validation.
    """
    result = NormalizedCounterparty(**valid_counterparty_payload())

    assert result.input_counterparty_name == "Acme Services Gmbh"
    assert result.matched_vendor_id == "V001"
    assert result.matched_vendor_name == "Acme Services GmbH"
    assert result.match_score == 97
    assert result.status == "approved"
    assert result.risk_level == "low"


def test_flags_and_findings_default_to_empty_lists() -> None:
    """
    Agent C should not need to manually provide empty lists when no risk
    flags or findings were generated.
    """
    result = NormalizedCounterparty(**valid_counterparty_payload())

    assert result.flags == []
    assert result.findings == []


@pytest.mark.parametrize("score", [None, 0, 50, 100])
def test_match_score_accepts_null_and_valid_boundaries(
    score: int | None,
) -> None:
    """
    The fuzzy-match score may be null or an integer from 0 through 100.
    """
    payload = valid_counterparty_payload()
    payload["match_score"] = score

    result = NormalizedCounterparty(**payload)

    assert result.match_score == score


@pytest.mark.parametrize("invalid_score", [-1, 101])
def test_match_score_rejects_values_outside_range(
    invalid_score: int,
) -> None:
    """
    Scores below 0 or above 100 are invalid.
    """
    payload = valid_counterparty_payload()
    payload["match_score"] = invalid_score

    with pytest.raises(ValidationError):
        NormalizedCounterparty(**payload)


def test_match_score_rejects_string_values() -> None:
    """
    Strict validation must not silently convert string scores to integers.
    """
    payload = valid_counterparty_payload()
    payload["match_score"] = "97"

    with pytest.raises(ValidationError):
        NormalizedCounterparty(**payload)


@pytest.mark.parametrize(
    "status",
    [
        "approved",
        "new",
        "unknown",
        "high_risk",
    ],
)
def test_status_accepts_all_supported_values(status: str) -> None:
    """
    Agent C must support every status listed in the requirements.
    """
    payload = valid_counterparty_payload()
    payload["status"] = status

    result = NormalizedCounterparty(**payload)

    assert result.status == status


def test_status_rejects_unsupported_value() -> None:
    """
    Arbitrary classifications must not enter normalized_counterparty.json.
    """
    payload = valid_counterparty_payload()
    payload["status"] = "blocked"

    with pytest.raises(ValidationError):
        NormalizedCounterparty(**payload)


def test_schema_rejects_unknown_extra_fields() -> None:
    """
    extra='forbid' must reject fields that are not part of the contract.
    """
    payload = valid_counterparty_payload()
    payload["unexpected_field"] = "not allowed"

    with pytest.raises(ValidationError):
        NormalizedCounterparty(**payload)


def test_schema_is_frozen_after_creation() -> None:
    """
    frozen=True prevents validated fields from being reassigned.
    """
    result = NormalizedCounterparty(**valid_counterparty_payload())

    with pytest.raises(ValidationError):
        result.status = "unknown"  # type: ignore[misc]


def test_nullable_vendor_fields_are_accepted_for_unknown_vendor() -> None:
    """
    When no vendor is found, vendor identifiers and the match score may
    legitimately be null.
    """
    payload = valid_counterparty_payload()
    payload.update(
        {
            "matched_vendor_id": None,
            "matched_vendor_name": None,
            "match_score": None,
            "status": "unknown",
            "risk_level": "high",
            "flags": [
                "counterparty_not_found",
                "manual_review_required",
            ],
        }
    )

    result = NormalizedCounterparty(**payload)

    assert result.matched_vendor_id is None
    assert result.matched_vendor_name is None
    assert result.match_score is None
    assert result.status == "unknown"
    assert "counterparty_not_found" in result.flags


def test_findings_accept_unified_finding_dictionary() -> None:
    """
    Verify that a dictionary following UnifiedFinding
    is accepted by NormalizedCounterparty.
    """

    # Start with a valid counterparty payload.
    payload = valid_counterparty_payload()

    # Change the payload to represent an unknown vendor.
    payload["matched_vendor_id"] = None
    payload["matched_vendor_name"] = None
    payload["match_score"] = None
    payload["status"] = "unknown"
    payload["risk_level"] = "high"

    # Add one finding that follows the real UnifiedFinding model.
    payload["findings"] = [
        {
            # Unique ID for this finding.
            "finding_id": "CP-001",
            # The field where the issue was detected.
            "field": "counterparty_name",
            # Severity uses the enum defined in finding.py.
            "severity": Severity.HIGH,
            # Human-readable explanation.
            "message": "No approved vendor match was found.",
            # Source used to verify the vendor.
            "evidence_ref": "vendor_master.csv",
            # No specific contract clause is connected to this finding.
            "clause_id": None,
            # No PDF page is needed for a CSV lookup.
            "page_number": None,
            # No PDF bounding box is available.
            "bbox": None,
            # Rule that caused the finding.
            "policy_rule": "vendor_master.match_required",
            # Expected system result.
            "expected_value": ("Counterparty should match an approved vendor"),
            # Actual system result.
            "actual_value": "No approved vendor match found",
            # Recommended next step.
            "recommendation": "Manual review required",
        }
    ]

    # Create and validate the model.
    result = NormalizedCounterparty(**payload)

    # Confirm that one finding was created.
    assert len(result.findings) == 1

    # Confirm that Pydantic converted the dictionary
    # into a UnifiedFinding object.
    assert isinstance(result.findings[0], UnifiedFinding)

    # Confirm that the important values are correct.
    assert result.findings[0].finding_id == "CP-001"
    assert result.findings[0].field == "counterparty_name"
    assert result.findings[0].severity == Severity.HIGH
    assert result.findings[0].policy_rule == "vendor_master.match_required"
