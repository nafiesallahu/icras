from app.agents.validation_agent import ValidationAgent
from app.schemas.extracted_contract import ExtractedContract
from app.schemas.validation_result import ValidationStatus


def test_validation_agent_normalizes_fields_from_extracted_contract():
    contract = ExtractedContract.from_json_file(
        "data/synthetic_fixtures/scenario_03_extracted_contract.json"
    )

    result = ValidationAgent().validate(contract)

    assert result.contract_id == "contract_003"
    assert result.normalized_fields.payment_terms_days == 90
    assert result.normalized_fields.governing_law == "Germany"
    assert result.normalized_fields.liability_cap_present is True
    assert result.normalized_fields.counterparty_name == "Acme Services GmbH"
    assert result.normalized_fields.has_low_confidence_extraction is True


def test_validation_agent_creates_payment_terms_finding_when_policy_exceeded():
    contract = ExtractedContract.from_json_file(
        "data/synthetic_fixtures/scenario_03_extracted_contract.json"
    )

    result = ValidationAgent().validate(contract)

    assert result.validation_status == ValidationStatus.COMPLETED_WITH_FINDINGS
    assert len(result.findings) == 1

    finding = result.findings[0]

    assert finding.field == "payment_terms_days"
    assert finding.policy_rule == "payment_terms.max_allowed_days"
    assert finding.expected_value == "30"
    assert finding.actual_value == "90"
    assert finding.recommendation is not None