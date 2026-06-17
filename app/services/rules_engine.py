from pathlib import Path

from app.schemas.clause_analysis import (
    ClauseAnalysis,
    RiskCategory,
    RiskTier,
    ScoredRiskFinding,
)

from app.schemas.validation_result import ValidationFinding, ValidationResult

from app.services.policy_loader import load_jurisdictions, load_playbook


class RulesEngine:
    def __init__(self):
        self.playbook = load_playbook()
        self.jurisdiction_policy = load_jurisdictions()

    def _get_score_for_finding(self, finding: ValidationFinding) -> int:
        score_map = {
            "payment_terms.max_allowed_days": (
                self.playbook.risk_scores["payment_terms_exceed_policy"]
            ),
            "liability_cap.required": (
                self.playbook.risk_scores["missing_liability_cap"]
            ),
            "gdpr.data_processing_clause_required_if.contains_personal_data": (
                self.playbook.risk_scores["missing_gdpr_clause"]
            ),
            "auto_renewal.minimum_opt_out_days": (
                self.playbook.risk_scores["auto_renewal_no_opt_out"]
            ),
            "governing_law.must_be_consistent": (
                self.playbook.risk_scores["conflicting_governing_law"]
            ),
            "confidence_thresholds.minimum_signature_confidence": (
                self.playbook.risk_scores["low_confidence_signature"]
            ),
            "effective_date.required": (
                self.playbook.risk_scores["missing_required_clause"]
            ),
            "counterparty.required": (
                self.playbook.risk_scores["missing_required_clause"]
            ),
            "governing_law.required": (
                self.playbook.risk_scores["missing_required_clause"]
            ),
            "payment_terms.required": (
                self.playbook.risk_scores["missing_required_clause"]
            ),
        }

        if finding.policy_rule is None:
            return 0

        return score_map.get(finding.policy_rule, 0)

    def _get_category_for_finding(
        self,
        finding: ValidationFinding,
    ) -> RiskCategory:
        if finding.policy_rule == "payment_terms.max_allowed_days":
            return RiskCategory.FINANCE

        if finding.policy_rule in {
            "liability_cap.required",
            "auto_renewal.minimum_opt_out_days",
            "governing_law.must_be_consistent",
            "confidence_thresholds.minimum_signature_confidence",
        }:
            return RiskCategory.LEGAL

        if finding.policy_rule == (
            "gdpr.data_processing_clause_required_if.contains_personal_data"
        ):
            return RiskCategory.COMPLIANCE

        return RiskCategory.MANUAL_REVIEW

    def _get_risk_tier(self, score: int) -> RiskTier:
        if score <= self.playbook.risk_thresholds["low"].max_score:
            return RiskTier.LOW

        if score <= self.playbook.risk_thresholds["medium"].max_score:
            return RiskTier.MEDIUM

        if score <= self.playbook.risk_thresholds["high"].max_score:
            return RiskTier.HIGH

        return RiskTier.CRITICAL

    def _create_jurisdiction_risk_finding(
        self,
        validation_result: ValidationResult,
    ) -> ScoredRiskFinding | None:
        governing_law = validation_result.normalized_fields.governing_law

        if governing_law is None:
            return None

        jurisdiction = self.jurisdiction_policy.jurisdictions.get(
            governing_law,
            self.jurisdiction_policy.default_jurisdiction,
        )

        if jurisdiction.risk_level not in {"high", "critical"}:
            return None

        score = self.playbook.risk_scores["high_risk_jurisdiction"]

        return ScoredRiskFinding(
            finding_id="RISK-JURISDICTION-001",
            source_finding_id="jurisdiction_rules",
            field="governing_law",
            category=RiskCategory.COMPLIANCE,
            risk_tier=self._get_risk_tier(score),
            score=score,
            policy_rule="jurisdiction_rules.high_risk_jurisdiction",
            expected_value="low or medium risk jurisdiction",
            actual_value=governing_law,
            recommendation="Compliance review required due to high-risk jurisdiction.",
            evidence_ref="jurisdiction_rules.yaml",
        )

    def _get_tolerance_band_for_finding(
        self,
        finding: ValidationFinding,
    ) -> str | None:
        if finding.policy_rule != "payment_terms.max_allowed_days":
            return None

        if finding.actual_value is None:
            return None

        try:
            payment_days = int(finding.actual_value)
        except ValueError:
            return None

        if payment_days <= self.playbook.payment_terms.max_allowed_days:
            return "compliant"

        if payment_days <= self.playbook.payment_terms.medium_risk_days:
            return "medium_tolerance_breach"

        if payment_days < self.playbook.payment_terms.high_risk_days:
            return "high_tolerance_breach"

        return "material_tolerance_breach"

    def _get_jurisdictions_evaluated(
        self,
        validation_result: ValidationResult,
    ) -> list[str]:
        jurisdictions_evaluated: list[str] = []

        governing_law = validation_result.normalized_fields.governing_law

        if governing_law is not None:
            jurisdictions_evaluated.append(governing_law)

        for finding in validation_result.findings:
            if (
                finding.policy_rule == "governing_law.must_be_consistent"
                and finding.actual_value
            ):
                parts = [
                    item.strip()
                    for item in finding.actual_value.split(",")
                ]

                for jurisdiction in parts:
                    if (
                        jurisdiction
                        and jurisdiction not in jurisdictions_evaluated
                    ):
                        jurisdictions_evaluated.append(jurisdiction)

        return jurisdictions_evaluated

    def score_validation_result(
        self,
        validation_result: ValidationResult,
    ) -> ClauseAnalysis:
        scored_findings: list[ScoredRiskFinding] = []
        total_score = 0

        for index, finding in enumerate(validation_result.findings, start=1):
            score = self._get_score_for_finding(finding)

            if score == 0:
                continue

            total_score += score

            scored_findings.append(
                ScoredRiskFinding(
                    finding_id=f"RISK-{index:03}",
                    source_finding_id=finding.finding_id,
                    field=finding.field,
                    category=self._get_category_for_finding(finding),
                    risk_tier=self._get_risk_tier(score),
                    score=score,
                    policy_rule=finding.policy_rule,
                    expected_value=finding.expected_value,
                    actual_value=finding.actual_value,
                    recommendation=finding.recommendation,
                    evidence_ref=finding.evidence_ref,
                    clause_id=finding.clause_id,
                    page_number=finding.page_number,
                    bbox=finding.bbox,
                    tolerance_band=self._get_tolerance_band_for_finding(finding),
                )
            )

        jurisdiction_finding = self._create_jurisdiction_risk_finding(
            validation_result,
        )

        if jurisdiction_finding is not None:
            total_score += jurisdiction_finding.score
            scored_findings.append(jurisdiction_finding)

        capped_score = min(total_score, 100)

        return ClauseAnalysis(
            contract_id=validation_result.contract_id,
            total_score=capped_score,
            risk_tier=self._get_risk_tier(capped_score),
            jurisdictions_evaluated=self._get_jurisdictions_evaluated(
                validation_result,
            ),
            findings=scored_findings,
        )

    def write_clause_analysis(
        self,
        run_dir: str | Path,
        clause_analysis: ClauseAnalysis,
    ) -> Path:
        output_path = Path(run_dir) / "clause_analysis.json"

        clause_analysis.to_json_file(output_path)

        return output_path