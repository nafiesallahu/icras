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
    
    def _get_effective_date(self, contract: ExtractedContract) -> str | None:
        effective_date_clause = self._get_clause_by_type(
            contract,
            ClauseType.EFFECTIVE_DATE,
        )

        if effective_date_clause is None:
            return None

        value = effective_date_clause.structured_fields.get("effective_date")

        if isinstance(value, str) and value.strip():
            return value

        return None
    
    def _get_all_governing_laws(self, contract: ExtractedContract) -> list[str]:
        governing_laws: list[str] = []

        for clause in contract.clauses:
            if clause.clause_type != ClauseType.GOVERNING_LAW:
                continue

            value = clause.structured_fields.get("governing_law")

            if isinstance(value, str) and value.strip():
                governing_laws.append(value)

        return governing_laws

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

    def _contains_personal_data(self, contract: ExtractedContract) -> bool:
        for clause in contract.clauses:
            value = clause.structured_fields.get("contains_personal_data")

            if value is True:
                return True

            if "personal data" in clause.text.lower():
                return True

        return False

    def _get_opt_out_window_days(
        self,
        contract: ExtractedContract,
    ) -> int | None:
        auto_renewal_clause = self._get_clause_by_type(
            contract,
            ClauseType.AUTO_RENEWAL,
        )

        if auto_renewal_clause is None:
            return None

        value = auto_renewal_clause.structured_fields.get(
            "opt_out_window_days"
        )

        if isinstance(value, int):
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

    def _check_liability_cap_rule(
        self,
        liability_cap_present: bool | None,
    ) -> ValidationFinding | None:
        if self.playbook.liability_cap.required is False:
            return None

        if liability_cap_present is True:
            return None

        return ValidationFinding(
            finding_id="VAL-002",
            field="liability_cap_present",
            severity=Severity.HIGH,
            message="Required liability cap clause is missing from the contract.",
            evidence_ref="missing:liability_cap",
            policy_rule="liability_cap.required",
            expected_value="true",
            actual_value=str(liability_cap_present).lower(),
            recommendation="Add a liability cap clause before contract approval.",
        )

    def _check_gdpr_clause_rule(
        self,
        contract: ExtractedContract,
        contains_personal_data: bool,
        gdpr_clause_present: bool | None,
    ) -> ValidationFinding | None:
        gdpr_required = (
            self.playbook.gdpr
            .data_processing_clause_required_if
            .contains_personal_data
        )

        if not gdpr_required:
            return None

        if contains_personal_data is False:
            return None

        if gdpr_clause_present is True:
            return None

        data_clause = None

        for clause in contract.clauses:
            if clause.structured_fields.get("contains_personal_data") is True:
                data_clause = clause
                break

        return ValidationFinding(
            finding_id="VAL-003",
            field="gdpr_clause_present",
            severity=Severity.HIGH,
            message="GDPR/data processing clause is required because the contract includes personal data.",
            evidence_ref=(
                data_clause.evidence_ref
                if data_clause
                else "missing:gdpr_data_processing"
            ),
            clause_id=data_clause.clause_id if data_clause else None,
            page_number=data_clause.page_number if data_clause else None,
            bbox=data_clause.bbox if data_clause else None,
            policy_rule="gdpr.data_processing_clause_required_if.contains_personal_data",
            expected_value="true",
            actual_value="false",
            recommendation="Add a GDPR/data processing clause before contract approval.",
        )

    def _check_auto_renewal_rule(
        self,
        contract: ExtractedContract,
        auto_renewal_present: bool | None,
        opt_out_window_days: int | None,
    ) -> ValidationFinding | None:
        if auto_renewal_present is not True:
            return None

        if self.playbook.auto_renewal.opt_out_window_required is False:
            return None

        minimum_days = self.playbook.auto_renewal.minimum_opt_out_days

        if opt_out_window_days is not None and opt_out_window_days >= minimum_days:
            return None

        auto_renewal_clause = self._get_clause_by_type(
            contract,
            ClauseType.AUTO_RENEWAL,
        )

        if auto_renewal_clause is None:
            return None

        return ValidationFinding(
            finding_id="VAL-004",
            field="opt_out_window_days",
            severity=Severity.MEDIUM,
            message="Auto-renewal clause is missing a valid opt-out window.",
            evidence_ref=auto_renewal_clause.evidence_ref,
            clause_id=auto_renewal_clause.clause_id,
            page_number=auto_renewal_clause.page_number,
            bbox=auto_renewal_clause.bbox,
            policy_rule="auto_renewal.minimum_opt_out_days",
            expected_value=str(minimum_days),
            actual_value=str(opt_out_window_days),
            recommendation=(
                f"Add an opt-out window of at least {minimum_days} days "
                "for the auto-renewal clause."
            ),
        )
    
    def _check_governing_law_conflict_rule(
        self,
        contract: ExtractedContract,
    ) -> ValidationFinding | None:
        governing_laws = self._get_all_governing_laws(contract)
        unique_laws = sorted(set(governing_laws))

        if len(unique_laws) <= 1:
            return None

        return ValidationFinding(
            finding_id="VAL-005",
            field="governing_law",
            severity=Severity.HIGH,
            message="Multiple conflicting governing law clauses were found.",
            evidence_ref="; ".join(
                clause.evidence_ref
                for clause in contract.clauses
                if clause.clause_type == ClauseType.GOVERNING_LAW
            ),
            policy_rule="governing_law.must_be_consistent",
            expected_value="Single consistent governing law",
            actual_value=", ".join(unique_laws),
            recommendation="Manual legal review required for conflicting governing law clauses.",
        )
    
    def _check_low_confidence_signature_rule(
        self,
        contract: ExtractedContract,
    ) -> ValidationFinding | None:
        signature_clause = self._get_clause_by_type(
            contract,
            ClauseType.SIGNATURE_BLOCK,
        )

        if signature_clause is None:
            return None

        minimum_confidence = 0.70

        if signature_clause.confidence_score >= minimum_confidence:
            return None

        return ValidationFinding(
            finding_id="VAL-006",
            field="signature_block",
            severity=Severity.MEDIUM,
            message="Signature block has low extraction confidence.",
            evidence_ref=signature_clause.evidence_ref,
            clause_id=signature_clause.clause_id,
            page_number=signature_clause.page_number,
            bbox=signature_clause.bbox,
            policy_rule="confidence_thresholds.minimum_signature_confidence",
            expected_value=str(minimum_confidence),
            actual_value=str(signature_clause.confidence_score),
            recommendation="Manual review required to confirm the signature block.",
        )

    def _check_missing_effective_date_rule(
        self,
        effective_date: str | None,
    ) -> ValidationFinding | None:
        if effective_date is not None:
            return None

        return ValidationFinding(
            finding_id="VAL-007",
            field="effective_date",
            severity=Severity.HIGH,
            message="Effective date is missing from the contract.",
            evidence_ref="missing:effective_date",
            policy_rule="effective_date.required",
            expected_value="present",
            actual_value="missing",
            recommendation="Add an effective date before contract approval.",
        )

    def _check_missing_counterparty_rule(
        self,
        counterparty_name: str | None,
    ) -> ValidationFinding | None:
        if counterparty_name is not None:
            return None

        return ValidationFinding(
            finding_id="VAL-008",
            field="counterparty_name",
            severity=Severity.HIGH,
            message="Counterparty name is missing from the contract.",
            evidence_ref="missing:counterparty",
            policy_rule="counterparty.required",
            expected_value="present",
            actual_value="missing",
            recommendation="Add or verify the contract counterparty before approval.",
        )

    def _check_missing_governing_law_rule(
        self,
        governing_law: str | None,
    ) -> ValidationFinding | None:
        if governing_law is not None:
            return None

        return ValidationFinding(
            finding_id="VAL-009",
            field="governing_law",
            severity=Severity.HIGH,
            message="Governing law is missing from the contract.",
            evidence_ref="missing:governing_law",
            policy_rule="governing_law.required",
            expected_value="present",
            actual_value="missing",
            recommendation="Add a governing law clause before contract approval.",
        )

    def _check_missing_payment_terms_rule(
        self,
        payment_terms_days: int | None,
    ) -> ValidationFinding | None:
        if payment_terms_days is not None:
            return None

        return ValidationFinding(
            finding_id="VAL-010",
            field="payment_terms_days",
            severity=Severity.HIGH,
            message="Payment terms are missing from the contract.",
            evidence_ref="missing:payment_terms",
            policy_rule="payment_terms.required",
            expected_value="present",
            actual_value="missing",
            recommendation="Add payment terms before contract approval.",
        )


    def validate(self, contract: ExtractedContract) -> ValidationResult:
        contains_personal_data = self._contains_personal_data(contract)

        normalized_fields = NormalizedFields(
            payment_terms_days=self._get_payment_terms_days(contract),
            governing_law=self._get_governing_law(contract),
            effective_date=self._get_effective_date(contract),
            liability_cap_present=self._has_clause(contract, ClauseType.LIABILITY_CAP),
            gdpr_clause_present=self._has_clause(contract, ClauseType.GDPR_DATA_PROCESSING),
            auto_renewal_present=self._has_clause(contract, ClauseType.AUTO_RENEWAL),
            opt_out_window_days=self._get_opt_out_window_days(contract),
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

        liability_cap_finding = self._check_liability_cap_rule(
            normalized_fields.liability_cap_present,
        )

        if liability_cap_finding is not None:
            findings.append(liability_cap_finding)

        gdpr_finding = self._check_gdpr_clause_rule(
            contract,
            contains_personal_data,
            normalized_fields.gdpr_clause_present,
        )

        if gdpr_finding is not None:
            findings.append(gdpr_finding)

        auto_renewal_finding = self._check_auto_renewal_rule(
            contract,
            normalized_fields.auto_renewal_present,
            normalized_fields.opt_out_window_days,
        )

        if auto_renewal_finding is not None:
            findings.append(auto_renewal_finding)

        governing_law_conflict_finding = self._check_governing_law_conflict_rule(
            contract,
        )

        if governing_law_conflict_finding is not None:
            findings.append(governing_law_conflict_finding)

        low_confidence_signature_finding = self._check_low_confidence_signature_rule(
            contract,
        )

        if low_confidence_signature_finding is not None:
            findings.append(low_confidence_signature_finding)

        effective_date_finding = self._check_missing_effective_date_rule(
            normalized_fields.effective_date,
        )

        if effective_date_finding is not None:
            findings.append(effective_date_finding)

        counterparty_finding = self._check_missing_counterparty_rule(
            normalized_fields.counterparty_name,
        )

        if counterparty_finding is not None:
            findings.append(counterparty_finding)

        governing_law_missing_finding = self._check_missing_governing_law_rule(
            normalized_fields.governing_law,
        )

        if governing_law_missing_finding is not None:
            findings.append(governing_law_missing_finding)

        payment_terms_missing_finding = self._check_missing_payment_terms_rule(
            normalized_fields.payment_terms_days,
        )

        if payment_terms_missing_finding is not None:
            findings.append(payment_terms_missing_finding)


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