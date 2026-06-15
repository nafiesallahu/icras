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

    fields = {finding.field for finding in result.findings}

    assert len(result.findings) == 2
    assert "payment_terms_days" in fields
    assert "effective_date" in fields

    finding = next(
        finding
        for finding in result.findings
        if finding.field == "payment_terms_days"
    )

    assert finding.policy_rule == "payment_terms.max_allowed_days"
    assert finding.expected_value == "30"
    assert finding.actual_value == "90"
    assert finding.recommendation is not None


def test_validation_agent_creates_missing_liability_cap_finding():
    contract = ExtractedContract.from_json_file(
        "data/synthetic_fixtures/scenario_02_extracted_contract.json"
    )

    result = ValidationAgent().validate(contract)

    assert result.validation_status == ValidationStatus.COMPLETED_WITH_FINDINGS
    assert result.normalized_fields.liability_cap_present is False
    assert len(result.findings) == 1

    finding = result.findings[0]

    assert finding.field == "liability_cap_present"
    assert finding.policy_rule == "liability_cap.required"
    assert finding.expected_value == "true"
    assert finding.actual_value == "false"
    assert finding.recommendation == "Add a liability cap clause before contract approval."


def test_validation_agent_creates_missing_gdpr_clause_finding():
    contract = ExtractedContract.from_json_file(
        "data/synthetic_fixtures/scenario_08_extracted_contract.json"
    )

    result = ValidationAgent().validate(contract)

    assert result.validation_status == ValidationStatus.COMPLETED_WITH_FINDINGS
    assert result.normalized_fields.gdpr_clause_present is False
    assert len(result.findings) == 1

    finding = result.findings[0]

    assert finding.field == "gdpr_clause_present"
    assert (
        finding.policy_rule
        == "gdpr.data_processing_clause_required_if.contains_personal_data"
    )
    assert finding.expected_value == "true"
    assert finding.actual_value == "false"
    assert finding.recommendation == (
        "Add a GDPR/data processing clause before contract approval."
    )


def test_validation_agent_creates_auto_renewal_no_opt_out_finding():
    contract = ExtractedContract.from_json_file(
        "data/synthetic_fixtures/scenario_05_extracted_contract.json"
    )

    result = ValidationAgent().validate(contract)

    assert result.validation_status == ValidationStatus.COMPLETED_WITH_FINDINGS
    assert result.normalized_fields.auto_renewal_present is True
    assert result.normalized_fields.opt_out_window_days is None
    assert len(result.findings) == 1

    finding = result.findings[0]

    assert finding.field == "opt_out_window_days"
    assert finding.policy_rule == "auto_renewal.minimum_opt_out_days"
    assert finding.expected_value == "30"
    assert finding.actual_value == "None"
    assert finding.recommendation == (
        "Add an opt-out window of at least 30 days for the auto-renewal clause."
    )


def test_validation_agent_creates_conflicting_governing_law_finding():
    contract = ExtractedContract.from_json_file(
        "data/synthetic_fixtures/scenario_06_extracted_contract.json"
    )

    result = ValidationAgent().validate(contract)

    assert result.validation_status == ValidationStatus.COMPLETED_WITH_FINDINGS
    assert len(result.findings) == 1

    finding = result.findings[0]

    assert finding.field == "governing_law"
    assert finding.policy_rule == "governing_law.must_be_consistent"
    assert finding.expected_value == "Single consistent governing law"
    assert finding.actual_value == "Germany, New York"
    assert finding.recommendation == (
        "Manual legal review required for conflicting governing law clauses."
    )


def test_validation_agent_creates_low_confidence_signature_finding():
    contract = ExtractedContract.from_json_file(
        "data/synthetic_fixtures/scenario_07_extracted_contract.json"
    )

    result = ValidationAgent().validate(contract)

    signature_finding = next(
        finding
        for finding in result.findings
        if finding.field == "signature_block"
    )

    assert result.validation_status == ValidationStatus.COMPLETED_WITH_FINDINGS

    assert signature_finding.field == "signature_block"
    assert (
        signature_finding.policy_rule
        == "confidence_thresholds.minimum_signature_confidence"
    )
    assert signature_finding.expected_value == "0.7"
    assert signature_finding.actual_value == "0.55"
    assert signature_finding.recommendation == (
        "Manual review required to confirm the signature block."
    )


def test_validation_agent_returns_no_findings_for_clean_contract():
    contract = ExtractedContract.from_json_file(
        "data/synthetic_fixtures/scenario_09_extracted_contract.json"
    )

    result = ValidationAgent().validate(contract)

    assert result.validation_status == ValidationStatus.COMPLETED
    assert len(result.findings) == 0

    assert result.normalized_fields.payment_terms_days == 30
    assert result.normalized_fields.governing_law == "Germany"
    assert result.normalized_fields.liability_cap_present is True
    assert result.normalized_fields.gdpr_clause_present is True
    assert result.normalized_fields.effective_date == "2025-07-01"


def test_validation_agent_creates_multiple_findings_for_high_risk_contract():
    contract = ExtractedContract.from_json_file(
        "data/synthetic_fixtures/scenario_10_extracted_contract.json"
    )

    result = ValidationAgent().validate(contract)

    assert result.validation_status == ValidationStatus.COMPLETED_WITH_FINDINGS
    assert result.normalized_fields.has_low_confidence_extraction is True

    fields = {finding.field for finding in result.findings}

    assert len(result.findings) == 5

    assert "payment_terms_days" in fields
    assert "liability_cap_present" in fields
    assert "gdpr_clause_present" in fields
    assert "opt_out_window_days" in fields
    assert "effective_date" in fields