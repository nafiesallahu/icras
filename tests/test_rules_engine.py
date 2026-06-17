from app.agents.validation_agent import ValidationAgent
from app.schemas.clause_analysis import (
    ClauseAnalysis,
    RiskCategory,
    RiskTier,
)
from app.schemas.extracted_contract import ExtractedContract
from app.services.rules_engine import RulesEngine


def test_rules_engine_scores_payment_terms_risk():
    contract = ExtractedContract.from_json_file(
        "data/synthetic_fixtures/scenario_03_extracted_contract.json"
    )

    validation_result = ValidationAgent().validate(contract)

    clause_analysis = RulesEngine().score_validation_result(
        validation_result
    )

    assert clause_analysis.contract_id == "contract_003"
    assert clause_analysis.total_score > 0
    assert clause_analysis.risk_tier in {
        RiskTier.LOW,
        RiskTier.MEDIUM,
        RiskTier.HIGH,
        RiskTier.CRITICAL,
    }
    assert len(clause_analysis.findings) > 0


def test_rules_engine_assigns_finance_category():
    contract = ExtractedContract.from_json_file(
        "data/synthetic_fixtures/scenario_03_extracted_contract.json"
    )

    validation_result = ValidationAgent().validate(contract)

    clause_analysis = RulesEngine().score_validation_result(
        validation_result
    )

    finding = clause_analysis.findings[0]

    assert finding.category == RiskCategory.FINANCE


def test_rules_engine_caps_payment_terms_and_missing_date_at_100():
    contract = ExtractedContract.from_json_file(
        "data/synthetic_fixtures/scenario_03_extracted_contract.json"
    )

    validation_result = ValidationAgent().validate(contract)

    clause_analysis = RulesEngine().score_validation_result(
        validation_result
    )

    fields = {finding.field for finding in clause_analysis.findings}

    assert clause_analysis.total_score == 100
    assert clause_analysis.risk_tier == RiskTier.CRITICAL

    assert "payment_terms_days" in fields
    assert "effective_date" in fields


def test_rules_engine_caps_multiple_risk_score_at_100():
    contract = ExtractedContract.from_json_file(
        "data/synthetic_fixtures/scenario_10_extracted_contract.json"
    )

    validation_result = ValidationAgent().validate(contract)

    clause_analysis = RulesEngine().score_validation_result(
        validation_result
    )

    assert clause_analysis.total_score == 100
    assert clause_analysis.risk_tier == RiskTier.CRITICAL
    assert len(clause_analysis.findings) >= 5

    fields = {finding.field for finding in clause_analysis.findings}
    policy_rules = {finding.policy_rule for finding in clause_analysis.findings}
    actual_values = {finding.actual_value for finding in clause_analysis.findings}

    assert "payment_terms_days" in fields
    assert "liability_cap_present" in fields
    assert "gdpr_clause_present" in fields
    assert "opt_out_window_days" in fields
    assert "governing_law" in fields

    assert "jurisdiction_rules.high_risk_jurisdiction" in policy_rules
    assert "Iran" in actual_values


def test_rules_engine_assigns_legal_category():
    contract = ExtractedContract.from_json_file(
        "data/synthetic_fixtures/scenario_02_extracted_contract.json"
    )

    validation_result = ValidationAgent().validate(contract)

    clause_analysis = RulesEngine().score_validation_result(
        validation_result
    )

    legal_finding = next(
        finding
        for finding in clause_analysis.findings
        if finding.field == "liability_cap_present"
    )

    assert legal_finding.category == RiskCategory.LEGAL


def test_rules_engine_assigns_compliance_category():
    contract = ExtractedContract.from_json_file(
        "data/synthetic_fixtures/scenario_08_extracted_contract.json"
    )

    validation_result = ValidationAgent().validate(contract)

    clause_analysis = RulesEngine().score_validation_result(
        validation_result
    )

    compliance_finding = next(
        finding
        for finding in clause_analysis.findings
        if finding.field == "gdpr_clause_present"
    )

    assert compliance_finding.category == RiskCategory.COMPLIANCE


def test_clause_analysis_can_be_written_and_loaded(tmp_path):
    contract = ExtractedContract.from_json_file(
        "data/synthetic_fixtures/scenario_03_extracted_contract.json"
    )

    validation_result = ValidationAgent().validate(contract)

    clause_analysis = RulesEngine().score_validation_result(
        validation_result
    )

    output_file = tmp_path / "clause_analysis.json"

    clause_analysis.to_json_file(output_file)

    loaded_analysis = ClauseAnalysis.from_json_file(output_file)

    assert loaded_analysis.contract_id == clause_analysis.contract_id
    assert loaded_analysis.total_score == clause_analysis.total_score
    assert loaded_analysis.risk_tier == clause_analysis.risk_tier
    assert len(loaded_analysis.findings) == len(
        clause_analysis.findings
    )


def test_rules_engine_creates_high_risk_jurisdiction_finding():
    contract = ExtractedContract.from_json_file(
        "data/synthetic_fixtures/scenario_04_extracted_contract.json"
    )

    validation_result = ValidationAgent().validate(contract)

    clause_analysis = RulesEngine().score_validation_result(
        validation_result
    )

    jurisdiction_finding = next(
        finding
        for finding in clause_analysis.findings
        if finding.policy_rule == "jurisdiction_rules.high_risk_jurisdiction"
    )

    assert clause_analysis.total_score == 95
    assert clause_analysis.risk_tier == RiskTier.CRITICAL

    assert jurisdiction_finding.field == "governing_law"
    assert jurisdiction_finding.category == RiskCategory.COMPLIANCE
    assert jurisdiction_finding.score == 95
    assert jurisdiction_finding.actual_value == "Russia"


def test_rules_engine_preserves_clause_evidence_fields():
    contract = ExtractedContract.from_json_file(
        "data/synthetic_fixtures/scenario_03_extracted_contract.json"
    )

    validation_result = ValidationAgent().validate(contract)

    clause_analysis = RulesEngine().score_validation_result(
        validation_result
    )

    payment_finding = next(
        finding
        for finding in clause_analysis.findings
        if finding.field == "payment_terms_days"
    )

    assert payment_finding.clause_id == "payment_terms_001"
    assert payment_finding.page_number == 3
    assert payment_finding.evidence_ref == (
        "contract.pdf#page=3&clause=payment_terms_001"
    )

def test_rules_engine_writes_clause_analysis_to_run_directory(tmp_path):
    contract = ExtractedContract.from_json_file(
        "data/synthetic_fixtures/scenario_03_extracted_contract.json"
    )

    validation_result = ValidationAgent().validate(contract)

    rules_engine = RulesEngine()

    clause_analysis = rules_engine.score_validation_result(
        validation_result
    )

    output_path = rules_engine.write_clause_analysis(
        tmp_path,
        clause_analysis,
    )

    assert output_path.name == "clause_analysis.json"
    assert output_path.exists()

    loaded_analysis = ClauseAnalysis.from_json_file(output_path)

    assert loaded_analysis.contract_id == clause_analysis.contract_id
    assert loaded_analysis.total_score == clause_analysis.total_score
    assert loaded_analysis.risk_tier == clause_analysis.risk_tier


def test_rules_engine_scores_conflicting_governing_law():
    contract = ExtractedContract.from_json_file(
        "data/synthetic_fixtures/scenario_06_extracted_contract.json"
    )

    validation_result = ValidationAgent().validate(contract)

    clause_analysis = RulesEngine().score_validation_result(
        validation_result
    )

    conflict_finding = next(
        finding
        for finding in clause_analysis.findings
        if finding.field == "governing_law"
        and finding.policy_rule == "governing_law.must_be_consistent"
    )

    assert clause_analysis.total_score == 90
    assert clause_analysis.risk_tier == RiskTier.CRITICAL

    assert conflict_finding.category == RiskCategory.LEGAL
    assert conflict_finding.score == 90
    assert conflict_finding.actual_value == "Germany, New York"

def test_rules_engine_scores_auto_renewal_without_opt_out():
    contract = ExtractedContract.from_json_file(
        "data/synthetic_fixtures/scenario_05_extracted_contract.json"
    )

    validation_result = ValidationAgent().validate(contract)

    clause_analysis = RulesEngine().score_validation_result(
        validation_result
    )

    auto_renewal_finding = next(
        finding
        for finding in clause_analysis.findings
        if finding.field == "opt_out_window_days"
    )

    assert clause_analysis.total_score == 75
    assert clause_analysis.risk_tier == RiskTier.HIGH

    assert auto_renewal_finding.category == RiskCategory.LEGAL
    assert auto_renewal_finding.score == 75
    assert auto_renewal_finding.policy_rule == "auto_renewal.minimum_opt_out_days"

def test_rules_engine_returns_low_risk_for_clean_contract():
    contract = ExtractedContract.from_json_file(
        "data/synthetic_fixtures/scenario_09_extracted_contract.json"
    )

    validation_result = ValidationAgent().validate(contract)

    clause_analysis = RulesEngine().score_validation_result(
        validation_result
    )

    assert clause_analysis.total_score == 0
    assert clause_analysis.risk_tier == RiskTier.LOW
    assert len(clause_analysis.findings) == 0

def test_rules_engine_assigns_payment_terms_tolerance_band():
    contract = ExtractedContract.from_json_file(
        "data/synthetic_fixtures/scenario_03_extracted_contract.json"
    )

    validation_result = ValidationAgent().validate(contract)

    clause_analysis = RulesEngine().score_validation_result(
        validation_result
    )

    payment_finding = next(
        finding
        for finding in clause_analysis.findings
        if finding.field == "payment_terms_days"
    )

    assert payment_finding.tolerance_band == "material_tolerance_breach"


def test_rules_engine_tracks_multiple_jurisdictions_for_conflict():
    contract = ExtractedContract.from_json_file(
        "data/synthetic_fixtures/scenario_06_extracted_contract.json"
    )

    validation_result = ValidationAgent().validate(contract)

    clause_analysis = RulesEngine().score_validation_result(
        validation_result
    )

    assert "Germany" in clause_analysis.jurisdictions_evaluated
    assert "New York" in clause_analysis.jurisdictions_evaluated