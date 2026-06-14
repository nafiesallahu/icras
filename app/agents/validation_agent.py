from app.schemas.extracted_contract import (
    ClauseType,
    ContractClause,
    ExtractedContract,
)

from app.schemas.validation_result import (
    NormalizedFields,
    ValidationFinding,
    ValidationResult,
    ValidationStatus,
    Severity,
)

from app.services.policy_loader import load_playbook


class ValidationAgent:
    def __init__(self):
        self.playbook = load_playbook()

    def _get_clause_by_type(
        self,
        contract: ExtractedContract,
        clause_type: ClauseType,
    ) -> ContractClause | None:
        for clause in contract.clauses:
            if clause.clause_type == clause_type:
                return clause
        return None

    def _get_payment_terms_days(self, contract: ExtractedContract) -> int | None:
        payment_clause = self._get_clause_by_type(
            contract,
            ClauseType.PAYMENT_TERMS,
        )

        if payment_clause is None:
            return None

        value = payment_clause.structured_fields.get("payment_terms_days")

        if isinstance(value, int):
            return value

        text = payment_clause.text.lower()

        if "ninety" in text or "90" in text:
            return 90

        if "sixty" in text or "60" in text:
            return 60

        if "thirty" in text or "30" in text:
            return 30

        return None

    def _get_governing_law(self, contract: ExtractedContract) -> str | None:
        governing_law_clause = self._get_clause_by_type(
            contract,
            ClauseType.GOVERNING_LAW,
        )

        if governing_law_clause is None:
            return None

        value = governing_law_clause.structured_fields.get("governing_law")

        if isinstance(value, str) and value.strip():
            return value

        text = governing_law_clause.text.lower()

        known_jurisdictions = [
            "Germany",
            "France",
            "Netherlands",
            "United Kingdom",
            "United States",
        ]

        for jurisdiction in known_jurisdictions:
            if jurisdiction.lower() in text:
                return jurisdiction

        return None

    def _has_clause(
        self,
        contract: ExtractedContract,
        clause_type: ClauseType,
    ) -> bool:
        return self._get_clause_by_type(contract, clause_type) is not None

    def _get_counterparty_name(self, contract: ExtractedContract) -> str | None:
        counterparty_clause = self._get_clause_by_type(
            contract,
            ClauseType.COUNTERPARTY,
        )

        if counterparty_clause is None:
            return None

        value = counterparty_clause.structured_fields.get("counterparty_name")

        if isinstance(value, str) and value.strip():
            return value

        return None

    def _check_payment_terms_rule(
        self,
        contract: ExtractedContract,
        payment_terms_days: int | None,
    ) -> ValidationFinding | None:
        if payment_terms_days is None:
            return None

        max_allowed_days = self.playbook.payment_terms.max_allowed_days

        if payment_terms_days <= max_allowed_days:
            return None

        payment_clause = self._get_clause_by_type(
            contract,
            ClauseType.PAYMENT_TERMS,
        )

        if payment_clause is None:
            return None

        return ValidationFinding(
            finding_id="VAL-001",
            field="payment_terms_days",
            severity=Severity.HIGH,
            message="Payment terms exceed the maximum allowed corporate limit.",
            evidence_ref=payment_clause.evidence_ref,
            clause_id=payment_clause.clause_id,
            page_number=payment_clause.page_number,
            bbox=payment_clause.bbox,
            policy_rule="payment_terms.max_allowed_days",
            expected_value=str(max_allowed_days),
            actual_value=str(payment_terms_days),
            recommendation=(
                f"Negotiate terms down to match maximum corporate limit "
                f"of {max_allowed_days} days."
            ),
        )

    def validate(self, contract: ExtractedContract) -> ValidationResult:
        normalized_fields = NormalizedFields(
            payment_terms_days=self._get_payment_terms_days(contract),
            governing_law=self._get_governing_law(contract),
            liability_cap_present=self._has_clause(contract, ClauseType.LIABILITY_CAP),
            gdpr_clause_present=self._has_clause(contract, ClauseType.GDPR_DATA_PROCESSING),
            auto_renewal_present=self._has_clause(contract, ClauseType.AUTO_RENEWAL),
            counterparty_name=self._get_counterparty_name(contract),
            has_low_confidence_extraction=contract.overall_confidence_score < 0.70,
        )

        findings: list[ValidationFinding] = []

        payment_terms_finding = self._check_payment_terms_rule(
            contract,
            normalized_fields.payment_terms_days,
        )

        if payment_terms_finding is not None:
            findings.append(payment_terms_finding)

        validation_status = (
            ValidationStatus.COMPLETED_WITH_FINDINGS
            if findings
            else ValidationStatus.COMPLETED
        )

        return ValidationResult(
            contract_id=contract.contract_id,
            validation_status=validation_status,
            normalized_fields=normalized_fields,
            findings=findings,
        )