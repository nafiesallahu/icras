"""Deterministic generator for ICRAS synthetic extraction fixtures.

This script produces the 10 static, fully deterministic ``extracted_contract.json``
fixtures (Agent B output) plus the Net 90 ``validation_result`` fixture (Agent D
output) used to exercise the downstream multi-agent pipeline in an offline,
zero-LLM environment.

Every fixture is strictly validated against the real Pydantic models
(``app.schemas.extracted_contract.ExtractedContract`` and
``app.schemas.validation_result.ValidationResult``) *before* it is written, so a
fixture can never land on disk unless it actually loads in the current codebase.

------------------------------------------------------------------------------
SCHEMA / FIELD MAPPING (the Pydantic models are the source of truth)
------------------------------------------------------------------------------
    ExtractedContract: schema_version, contract_id, source_file, extraction_mode,
                       overall_confidence_score, clauses
    ContractClause:    clause_id, clause_type, text, page_number, confidence_score,
                       evidence_ref, structured_fields, bbox
    BoundingBox:       x0, top, x1, bottom         (NO "locations", NO "evidence_ids")

All models are ``extra="forbid"`` and ``frozen=True``. Unknown top-level fields are
rejected, so per-scenario metadata that has no schema field (for example the
"personal data is processed" signal used by Scenario 08 / Scenario 10) is encoded
only inside the free-form ``structured_fields`` dict, never as a new top-level key.

------------------------------------------------------------------------------
POLICY ALIGNMENT
------------------------------------------------------------------------------
Canonical risk-flag vocabulary (policies/playbook.yaml, approval_policy.yaml):
    payment_terms_exceeded, missing_required_clause, missing_liability_cap,
    high_risk_jurisdiction, missing_gdpr_clause, auto_renewal_without_opt_out,
    unknown_vendor, jurisdiction_conflict

Jurisdiction policy keys are SHORT names (policies/jurisdiction_rules.yaml), so
``structured_fields.governing_law`` is normalized to "Russia" / "Iran" while the
clause ``text`` keeps the long legal wording for traceability.

------------------------------------------------------------------------------
SCENARIO RESOLUTIONS (documented, deterministic)
------------------------------------------------------------------------------
SC09  Restored to CLEAN SERVICES AGREEMENT UNDER THRESHOLD to match the official
      scenario matrix. Scenario 09 now uses an approved known vendor present in
      policies/vendor_master.csv ("Hooli Services GmbH"), Net 30 payment terms,
      a present liability cap, Germany governing law and a GDPR clause, so it is
      clean and auto-approvable with no major findings. The Unknown Vendor case
      is retained only as a documented extension helper (UNKNOWN_VENDOR_NAME /
      build_unknown_vendor_extension) and is NOT generated as scenario 09.

SC08 / SC10  MISSING GDPR signal without breaking the schema. ExtractedContract is
      extra="forbid", so a top-level ``contains_personal_data`` field is illegal,
      and there is no context/manifest schema in the codebase yet. The
      "personal data is processed" precondition is therefore encoded inside a
      ``clause_type="other"`` scope clause via
      ``structured_fields = {"contains_personal_data": true}`` (legal, because the
      forbid rule does not apply to values inside the structured_fields dict). No
      ``gdpr_data_processing`` clause is emitted, so the missing_gdpr_clause risk
      remains derivable from the fixture alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make sure the repository root is importable so ``app`` resolves when this
# script is run directly (e.g. ``python scripts/generate_fixtures.py``).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.schemas.extracted_contract import ExtractedContract  # noqa: E402
from app.schemas.validation_result import ValidationResult  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "data" / "synthetic_fixtures"
POLICIES_DIR = REPO_ROOT / "policies"
VENDOR_MASTER_PATH = POLICIES_DIR / "vendor_master.csv"

# Deterministic trusted vendor master. Scenario 09 ("Hooli Services GmbH") is an
# approved known vendor present here, so it never triggers the unknown_vendor
# flag.
VENDOR_MASTER_ROWS = [
    ("vendor_id", "vendor_name", "country", "status", "risk_level"),
    ("V001", "Globex Industries GmbH", "Germany", "approved", "low"),
    ("V002", "Initech Components Ltd", "Germany", "approved", "low"),
    ("V003", "Acme Services GmbH", "Germany", "approved", "low"),
    ("V004", "Umbrella SaaS Inc", "United States", "approved", "medium"),
    ("V005", "Wayne Enterprises GmbH", "Germany", "approved", "low"),
    ("V006", "Stark Innovations GmbH", "Germany", "approved", "low"),
    ("V007", "Hooli Services GmbH", "Germany", "approved", "low"),
]

# Extension-only counterparty (NOT used by any of the 10 official scenarios). The
# Unknown Vendor case is retained purely as a future extension / Agent C unit-test
# helper; it is intentionally absent from VENDOR_MASTER_ROWS so that, if ever
# generated, it would trigger the unknown_vendor flag.
UNKNOWN_VENDOR_NAME = "Acme Rogue Systems LLC"


# ---------------------------------------------------------------------------
# Helpers (shared construction logic)
# ---------------------------------------------------------------------------
def bbox(x0: float, top: float, x1: float, bottom: float) -> dict:
    """Build a model-shaped bounding box (x0/top/x1/bottom)."""
    return {"x0": x0, "top": top, "x1": x1, "bottom": bottom}


def make_clause(
    clause_id: str,
    clause_type: str,
    text: str,
    page_number: int,
    confidence_score: float,
    structured_fields: dict | None = None,
    box: dict | None = None,
) -> dict:
    """Build a single clause block that satisfies ContractClause exactly."""
    return {
        "clause_id": clause_id,
        "clause_type": clause_type,
        "text": text,
        "page_number": page_number,
        "confidence_score": confidence_score,
        "evidence_ref": f"contract.pdf#page={page_number}&clause={clause_id}",
        "structured_fields": dict(structured_fields or {}),
        "bbox": box,
    }


def make_contract(
    contract_id: str,
    source_file: str,
    extraction_mode: str,
    overall_confidence_score: float,
    clauses: list[dict],
) -> dict:
    """Build the top-level ExtractedContract dictionary."""
    return {
        "schema_version": "1.0",
        "contract_id": contract_id,
        "source_file": source_file,
        "extraction_mode": extraction_mode,
        "overall_confidence_score": overall_confidence_score,
        "clauses": clauses,
    }


# ---------------------------------------------------------------------------
# Scenario builders (SC01 - SC10)
# ---------------------------------------------------------------------------
def scenario_01() -> dict:
    """SC01 - Clean Mutual NDA -> auto approve, low risk, no findings.

    A true Mutual Non-Disclosure Agreement with Globex Industries GmbH. Contains
    the NDA-appropriate required clauses (counterparty, effective_date,
    confidentiality, termination, governing_law, signature_block) plus the
    universal playbook-required clauses (liability_cap, gdpr_data_processing) as
    safe, low-risk baselines so no missing_required_clause finding is raised. No
    payment_terms clause is emitted because an NDA carries no payment terms and
    payment_terms is not a universal mandatory clause in policies/playbook.yaml.
    All confidence scores are high, so the scenario stays clean and
    auto-approvable.
    """
    return make_contract(
        contract_id="contract_001",
        source_file="globex_mutual_nda.pdf",
        extraction_mode="pdf_parser",
        overall_confidence_score=0.97,
        clauses=[
            make_clause(
                "counterparty_001",
                "counterparty",
                "This Mutual Non-Disclosure Agreement is entered into with Globex "
                "Industries GmbH, a company incorporated under the laws of Germany.",
                1,
                0.98,
                {"counterparty_name": "Globex Industries GmbH"},
                bbox(72.0, 120.5, 523.4, 158.9),
            ),
            make_clause(
                "effective_date_001",
                "effective_date",
                "This Mutual Non-Disclosure Agreement shall take effect as of "
                "01 January 2025.",
                1,
                0.96,
                {"effective_date": "2025-01-01"},
                bbox(72.0, 168.0, 410.2, 196.3),
            ),
            make_clause(
                "confidentiality_001",
                "confidentiality",
                "Each party shall keep the other party's Confidential Information "
                "in strict confidence, use it solely to evaluate the contemplated "
                "business relationship, and disclose it only to representatives "
                "with a need to know who are bound by equivalent obligations.",
                2,
                0.97,
                {"confidentiality_present": True, "confidentiality_term_years": 3},
                bbox(72.0, 210.0, 520.0, 268.0),
            ),
            make_clause(
                "liability_cap_001",
                "liability_cap",
                "The aggregate liability of either party under this Agreement "
                "shall not exceed EUR 100,000.",
                3,
                0.95,
                {
                    "liability_cap_present": True,
                    "liability_cap_amount": 100000,
                    "currency": "EUR",
                },
                bbox(72.0, 280.0, 521.9, 322.0),
            ),
            make_clause(
                "termination_001",
                "termination",
                "Either party may terminate this Agreement upon thirty (30) days "
                "written notice; the confidentiality obligations shall survive "
                "such termination.",
                3,
                0.96,
                {},
                bbox(72.0, 330.0, 510.7, 380.0),
            ),
            make_clause(
                "governing_law_001",
                "governing_law",
                "This Agreement shall be governed by and construed in accordance "
                "with the laws of Germany.",
                4,
                0.98,
                {"governing_law": "Germany"},
                bbox(72.0, 100.0, 489.3, 138.0),
            ),
            make_clause(
                "gdpr_data_processing_001",
                "gdpr_data_processing",
                "To the extent any personal data is exchanged under this "
                "Agreement, the parties shall process it in accordance with "
                "Regulation (EU) 2016/679 (GDPR), applying appropriate technical "
                "and organisational measures and deleting or returning such data "
                "upon termination.",
                4,
                0.95,
                {"gdpr_clause_present": True},
                bbox(72.0, 150.0, 525.0, 230.0),
            ),
            make_clause(
                "signature_block_001",
                "signature_block",
                "IN WITNESS WHEREOF, the parties have executed this Mutual "
                "Non-Disclosure Agreement as of the Effective Date.",
                5,
                0.96,
                {},
                bbox(72.0, 500.0, 520.0, 540.0),
            ),
        ],
    )


def scenario_02() -> dict:
    """SC02 - Missing Liability Cap -> legal escalation (missing_liability_cap).

    Every other required clause is present and safe, so the ONLY risk is the
    completely absent liability_cap clause.
    """
    return make_contract(
        contract_id="contract_002",
        source_file="initech_supply_agreement.pdf",
        extraction_mode="pdf_parser",
        overall_confidence_score=0.92,
        clauses=[
            make_clause(
                "counterparty_002",
                "counterparty",
                "This Supply Agreement is made with Initech Components Ltd.",
                1,
                0.94,
                {"counterparty_name": "Initech Components Ltd"},
                bbox(72.0, 118.0, 470.5, 150.2),
            ),
            make_clause(
                "effective_date_002",
                "effective_date",
                "This Agreement shall take effect as of 15 March 2025.",
                1,
                0.93,
                {"effective_date": "2025-03-15"},
                bbox(72.0, 160.0, 410.0, 192.0),
            ),
            make_clause(
                "payment_terms_002",
                "payment_terms",
                "Invoices are payable Net 30 from receipt of goods.",
                3,
                0.93,
                {"payment_terms_days": 30},
                bbox(72.0, 205.0, 460.1, 240.3),
            ),
            make_clause(
                "confidentiality_002",
                "confidentiality",
                "Each party shall protect the Confidential Information of the "
                "other party against unauthorised disclosure.",
                4,
                0.92,
                {"confidentiality_present": True},
                bbox(72.0, 260.0, 520.0, 300.0),
            ),
            make_clause(
                "governing_law_002",
                "governing_law",
                "This Agreement shall be governed by the laws of Germany.",
                6,
                0.93,
                {"governing_law": "Germany"},
                bbox(72.0, 400.0, 489.3, 430.4),
            ),
            make_clause(
                "gdpr_data_processing_002",
                "gdpr_data_processing",
                "Each party shall comply with applicable data protection law, "
                "including the GDPR, when processing personal data under this "
                "Agreement.",
                7,
                0.91,
                {"gdpr_clause_present": True},
                bbox(72.0, 110.0, 522.0, 175.6),
            ),
            make_clause(
                "termination_002",
                "termination",
                "Either party may terminate this Agreement for material breach "
                "upon thirty (30) days written notice.",
                8,
                0.92,
                {},
                bbox(72.0, 220.0, 510.7, 262.9),
            ),
        ],
    )


def scenario_03() -> dict:
    """SC03 - Net 90 Payment Terms -> finance escalation (payment_terms_exceeded).

    Test-locked invariants (tests/test_schema_fixtures.py):
        contract_003, synthetic_fallback, overall 0.6, clauses[0] = payment_terms
        on page 3. ALL bbox values are null (synthetic fallback).

    A counterparty clause is appended LAST so clauses[0] stays payment_terms while
    still providing a counterparty (so the validation result can legitimately
    populate counterparty_name).
    """
    return make_contract(
        contract_id="contract_003",
        source_file="contract.pdf",
        extraction_mode="synthetic_fallback",
        overall_confidence_score=0.6,
        clauses=[
            make_clause(
                "payment_terms_001",
                "payment_terms",
                "Payment shall be made within 90 days of invoice receipt.",
                3,
                0.6,
                {"payment_terms_days": 90},
                None,
            ),
            make_clause(
                "governing_law_001",
                "governing_law",
                "This Agreement shall be governed by the laws of Germany.",
                6,
                0.6,
                {"governing_law": "Germany"},
                None,
            ),
            make_clause(
                "liability_cap_001",
                "liability_cap",
                "Each party's total liability shall not exceed the total fees "
                "paid under this Agreement.",
                5,
                0.6,
                {"liability_cap_present": True},
                None,
            ),
            make_clause(
                "counterparty_001",
                "counterparty",
                "This Agreement is entered into with Acme Services GmbH.",
                1,
                0.6,
                {"counterparty_name": "Acme Services GmbH"},
                None,
            ),
        ],
    )


def scenario_04() -> dict:
    """SC04 - High-Risk Jurisdiction -> compliance escalation (high_risk_jurisdiction).

    The clause text keeps the long legal name ("The Russian Federation") for
    traceability, while structured_fields.governing_law is normalized to the short
    policy key "Russia". The counterparty is a trusted vendor so the only risk is
    the jurisdiction.
    """
    return make_contract(
        contract_id="contract_004",
        source_file="globex_eastern_operations_agreement.pdf",
        extraction_mode="pdf_parser",
        overall_confidence_score=0.9,
        clauses=[
            make_clause(
                "counterparty_004",
                "counterparty",
                "This Agreement is entered into with Globex Industries GmbH for "
                "operations conducted in the eastern region.",
                1,
                0.92,
                {"counterparty_name": "Globex Industries GmbH"},
                bbox(72.0, 116.0, 510.0, 148.5),
            ),
            make_clause(
                "effective_date_004",
                "effective_date",
                "This Agreement shall take effect as of 01 April 2025.",
                1,
                0.91,
                {"effective_date": "2025-04-01"},
                bbox(72.0, 160.0, 410.0, 192.0),
            ),
            make_clause(
                "payment_terms_004",
                "payment_terms",
                "Payment terms are Net 30 from the date of invoice.",
                3,
                0.9,
                {"payment_terms_days": 30},
                bbox(72.0, 210.0, 455.0, 244.0),
            ),
            make_clause(
                "confidentiality_004",
                "confidentiality",
                "Each party shall keep the other party's Confidential Information "
                "in strict confidence.",
                4,
                0.9,
                {"confidentiality_present": True},
                bbox(72.0, 260.0, 520.0, 300.0),
            ),
            make_clause(
                "liability_cap_004",
                "liability_cap",
                "Total liability shall not exceed the fees paid in the prior "
                "twelve (12) months.",
                5,
                0.89,
                {"liability_cap_present": True},
                bbox(72.0, 300.0, 512.0, 342.0),
            ),
            make_clause(
                "termination_004",
                "termination",
                "Either party may terminate for material breach upon thirty (30) "
                "days written notice.",
                5,
                0.9,
                {},
                bbox(72.0, 350.0, 510.0, 392.0),
            ),
            make_clause(
                "governing_law_004",
                "governing_law",
                "This Agreement shall be governed by, and the legal seat of "
                "performance shall be located in, The Russian Federation.",
                6,
                0.9,
                {"governing_law": "Russia"},
                bbox(72.0, 400.0, 525.0, 446.2),
            ),
            make_clause(
                "gdpr_data_processing_004",
                "gdpr_data_processing",
                "The parties shall process personal data in accordance with the "
                "GDPR and applicable data protection law.",
                7,
                0.89,
                {"gdpr_clause_present": True},
                bbox(72.0, 100.0, 522.0, 175.0),
            ),
        ],
    )


def scenario_05() -> dict:
    """SC05 - Auto Renewal Without Opt-Out -> legal/material risk
    (auto_renewal_without_opt_out).

    Contains auto-renewal language but no opt-out window. A safe liability_cap is
    injected (the scenario is NOT a missing-cap test), and all other required
    clauses are safe, so the only risk is the missing opt-out.
    """
    return make_contract(
        contract_id="contract_005",
        source_file="umbrella_saas_subscription.pdf",
        extraction_mode="pdf_parser",
        overall_confidence_score=0.91,
        clauses=[
            make_clause(
                "counterparty_005",
                "counterparty",
                "This Subscription Agreement is made with Umbrella SaaS Inc.",
                1,
                0.92,
                {"counterparty_name": "Umbrella SaaS Inc"},
                bbox(72.0, 120.0, 470.0, 152.0),
            ),
            make_clause(
                "effective_date_005",
                "effective_date",
                "This Agreement shall take effect as of 01 February 2025.",
                1,
                0.91,
                {"effective_date": "2025-02-01"},
                bbox(72.0, 160.0, 410.0, 192.0),
            ),
            make_clause(
                "payment_terms_005",
                "payment_terms",
                "Subscription fees are invoiced annually and due Net 30.",
                3,
                0.9,
                {"payment_terms_days": 30},
                bbox(72.0, 210.0, 460.0, 246.0),
            ),
            make_clause(
                "confidentiality_005",
                "confidentiality",
                "Each party shall protect the other party's Confidential "
                "Information from unauthorised disclosure.",
                3,
                0.9,
                {"confidentiality_present": True},
                bbox(72.0, 255.0, 520.0, 295.0),
            ),
            make_clause(
                "auto_renewal_005",
                "auto_renewal",
                "This Agreement shall automatically renew for successive twelve "
                "(12) month terms at the end of each then-current term.",
                4,
                0.9,
                {"auto_renewal_present": True, "opt_out_window_days": None},
                bbox(72.0, 300.0, 522.0, 348.0),
            ),
            make_clause(
                "liability_cap_005",
                "liability_cap",
                "Aggregate liability shall not exceed the total fees paid in the "
                "preceding twelve (12) months.",
                5,
                0.9,
                {"liability_cap_present": True},
                bbox(72.0, 360.0, 512.0, 402.0),
            ),
            make_clause(
                "termination_005",
                "termination",
                "Either party may terminate for material breach upon thirty (30) "
                "days written notice.",
                5,
                0.9,
                {},
                bbox(72.0, 410.0, 510.0, 452.0),
            ),
            make_clause(
                "governing_law_005",
                "governing_law",
                "This Agreement shall be governed by the laws of Germany.",
                6,
                0.9,
                {"governing_law": "Germany"},
                bbox(72.0, 460.0, 489.0, 490.0),
            ),
            make_clause(
                "gdpr_data_processing_005",
                "gdpr_data_processing",
                "The parties shall process personal data in accordance with the "
                "GDPR and applicable data protection law.",
                7,
                0.9,
                {"gdpr_clause_present": True},
                bbox(72.0, 100.0, 522.0, 175.0),
            ),
        ],
    )


def scenario_06() -> dict:
    """SC06 - Conflicting Governing Law -> legal escalation (jurisdiction_conflict).

    Two governing_law clauses co-exist on the SAME page (page 6) so the scenario
    exercises same-page conflict detection: one normalizes to Germany, the other
    to New York.
    """
    return make_contract(
        contract_id="contract_006",
        source_file="wayne_enterprises_agreement.pdf",
        extraction_mode="pdf_parser",
        overall_confidence_score=0.88,
        clauses=[
            make_clause(
                "counterparty_006",
                "counterparty",
                "This Agreement is entered into with Wayne Enterprises GmbH.",
                1,
                0.9,
                {"counterparty_name": "Wayne Enterprises GmbH"},
                bbox(72.0, 118.0, 478.0, 150.0),
            ),
            make_clause(
                "effective_date_006",
                "effective_date",
                "This Agreement shall take effect as of 01 May 2025.",
                1,
                0.9,
                {"effective_date": "2025-05-01"},
                bbox(72.0, 160.0, 410.0, 192.0),
            ),
            make_clause(
                "payment_terms_006",
                "payment_terms",
                "Payment is due Net 30 from invoice date.",
                3,
                0.9,
                {"payment_terms_days": 30},
                bbox(72.0, 210.0, 450.0, 244.0),
            ),
            make_clause(
                "confidentiality_006",
                "confidentiality",
                "Each party shall keep the other party's Confidential Information "
                "in strict confidence.",
                4,
                0.9,
                {"confidentiality_present": True},
                bbox(72.0, 260.0, 520.0, 300.0),
            ),
            make_clause(
                "liability_cap_006",
                "liability_cap",
                "Aggregate liability shall not exceed the total fees paid under "
                "this Agreement.",
                5,
                0.89,
                {"liability_cap_present": True},
                bbox(72.0, 300.0, 512.0, 342.0),
            ),
            make_clause(
                "termination_006",
                "termination",
                "Either party may terminate for material breach upon thirty (30) "
                "days written notice.",
                5,
                0.9,
                {},
                bbox(72.0, 350.0, 510.0, 392.0),
            ),
            make_clause(
                "governing_law_006_a",
                "governing_law",
                "The Courts of Frankfurt, Germany shall have exclusive "
                "jurisdiction over any dispute arising under this Agreement.",
                6,
                0.86,
                {"governing_law": "Germany", "forum": "Frankfurt, Germany"},
                bbox(72.0, 300.0, 525.0, 346.0),
            ),
            make_clause(
                "governing_law_006_b",
                "governing_law",
                "Notwithstanding the foregoing, New York Law applies to the "
                "interpretation and enforcement of this Agreement.",
                6,
                0.85,
                {"governing_law": "New York"},
                bbox(72.0, 360.0, 520.0, 406.0),
            ),
            make_clause(
                "gdpr_data_processing_006",
                "gdpr_data_processing",
                "The parties shall process personal data in accordance with the "
                "GDPR and applicable data protection law.",
                7,
                0.89,
                {"gdpr_clause_present": True},
                bbox(72.0, 100.0, 522.0, 175.0),
            ),
        ],
    )


def scenario_07() -> dict:
    """SC07 - Low Confidence Signature -> manual review (medium risk).

    Mutual NDA structure. The signature block confidence is deliberately low
    (0.55) and the overall confidence is below threshold (0.68) so the scenario
    truly tests low-confidence routing. These low values must not be raised.
    """
    return make_contract(
        contract_id="contract_007",
        source_file="mutual_nda.pdf",
        extraction_mode="pdf_parser",
        overall_confidence_score=0.68,
        clauses=[
            make_clause(
                "counterparty_007",
                "counterparty",
                "This Mutual Non-Disclosure Agreement is entered into with "
                "Stark Innovations GmbH.",
                1,
                0.9,
                {"counterparty_name": "Stark Innovations GmbH"},
                bbox(72.0, 120.0, 500.0, 152.0),
            ),
            make_clause(
                "confidentiality_007",
                "confidentiality",
                "Each party agrees to hold the other party's Confidential "
                "Information in strict confidence and to use it solely for the "
                "purpose of evaluating a potential business relationship.",
                2,
                0.9,
                {"confidentiality_present": True, "confidentiality_term_years": 5},
                bbox(72.0, 200.0, 525.0, 260.0),
            ),
            make_clause(
                "termination_007",
                "termination",
                "The confidentiality obligations shall survive for five (5) "
                "years following termination of this Agreement.",
                3,
                0.9,
                {},
                bbox(72.0, 300.0, 515.0, 344.0),
            ),
            make_clause(
                "governing_law_007",
                "governing_law",
                "This Agreement shall be governed by the laws of Germany.",
                4,
                0.9,
                {"governing_law": "Germany"},
                bbox(72.0, 400.0, 489.0, 430.0),
            ),
            make_clause(
                "signature_block_007",
                "signature_block",
                "IN WITNESS WHEREOF, the parties have executed this Agreement as "
                "of the Effective Date.",
                5,
                0.55,
                {},
                bbox(72.0, 500.0, 520.0, 540.0),
            ),
        ],
    )


def scenario_08() -> dict:
    """SC08 - Missing GDPR Clause -> compliance escalation (missing_gdpr_clause).

    Personal data IS processed (encoded via an "other" scope clause carrying
    structured_fields.contains_personal_data = true), but NO gdpr_data_processing
    clause exists. All other clauses are safe so the only risk is the missing GDPR
    clause.
    """
    return make_contract(
        contract_id="contract_008",
        source_file="hooli_services_agreement.pdf",
        extraction_mode="pdf_parser",
        overall_confidence_score=0.91,
        clauses=[
            make_clause(
                "counterparty_008",
                "counterparty",
                "This Services Agreement is entered into with Hooli Services GmbH.",
                1,
                0.92,
                {"counterparty_name": "Hooli Services GmbH"},
                bbox(72.0, 118.0, 485.0, 150.0),
            ),
            make_clause(
                "effective_date_008",
                "effective_date",
                "This Agreement shall take effect as of 01 June 2025.",
                1,
                0.91,
                {"effective_date": "2025-06-01"},
                bbox(72.0, 160.0, 410.0, 192.0),
            ),
            make_clause(
                "data_scope_008",
                "other",
                "Under this Agreement the Provider processes personal data of the "
                "Customer's employees and end users, including names, contact "
                "details and usage records, in order to deliver the services.",
                2,
                0.9,
                {"contains_personal_data": True},
                bbox(72.0, 200.0, 525.0, 260.0),
            ),
            make_clause(
                "payment_terms_008",
                "payment_terms",
                "Payment is due Net 30 from the date of invoice.",
                3,
                0.91,
                {"payment_terms_days": 30},
                bbox(72.0, 280.0, 450.0, 314.0),
            ),
            make_clause(
                "confidentiality_008",
                "confidentiality",
                "Each party shall keep the other party's Confidential Information "
                "in strict confidence.",
                4,
                0.9,
                {"confidentiality_present": True},
                bbox(72.0, 330.0, 520.0, 370.0),
            ),
            make_clause(
                "liability_cap_008",
                "liability_cap",
                "Aggregate liability shall not exceed the total fees paid under "
                "this Agreement.",
                5,
                0.9,
                {"liability_cap_present": True},
                bbox(72.0, 380.0, 512.0, 422.0),
            ),
            make_clause(
                "governing_law_008",
                "governing_law",
                "This Agreement shall be governed by the laws of Germany.",
                6,
                0.9,
                {"governing_law": "Germany"},
                bbox(72.0, 430.0, 489.0, 460.0),
            ),
            make_clause(
                "termination_008",
                "termination",
                "Either party may terminate for convenience upon sixty (60) "
                "days written notice.",
                8,
                0.9,
                {},
                bbox(72.0, 220.0, 510.0, 262.0),
            ),
        ],
    )


def scenario_09() -> dict:
    """SC09 - Clean Standard Services Agreement Under Threshold -> auto approve.

    Restored to the official Clean Services interpretation (see module docstring).
    The counterparty is "Hooli Services GmbH", an approved known vendor present in
    policies/vendor_master.csv. Payment terms are Net 30, the liability cap is
    present, governing law is Germany, and a compliant GDPR clause is included, so
    the contract is low-risk with no major findings and is auto-approvable. It must
    NOT trigger unknown_vendor, missing_liability_cap, payment_terms_exceeded,
    high_risk_jurisdiction, missing_gdpr_clause, auto_renewal_without_opt_out or
    jurisdiction_conflict.
    """
    return make_contract(
        contract_id="contract_009",
        source_file="hooli_clean_services_agreement.pdf",
        extraction_mode="pdf_parser",
        overall_confidence_score=0.92,
        clauses=[
            make_clause(
                "counterparty_009",
                "counterparty",
                "This Standard Services Agreement is entered into with Hooli "
                "Services GmbH, a company incorporated under the laws of Germany.",
                1,
                0.95,
                {"counterparty_name": "Hooli Services GmbH"},
                bbox(72.0, 120.0, 490.0, 152.0),
            ),
            make_clause(
                "effective_date_009",
                "effective_date",
                "This Agreement shall take effect as of 01 July 2025.",
                1,
                0.92,
                {"effective_date": "2025-07-01"},
                bbox(72.0, 160.0, 410.0, 192.0),
            ),
            make_clause(
                "payment_terms_009",
                "payment_terms",
                "Payment is due Net 30 from invoice receipt.",
                3,
                0.91,
                {"payment_terms_days": 30},
                bbox(72.0, 210.0, 450.0, 244.0),
            ),
            make_clause(
                "confidentiality_009",
                "confidentiality",
                "Each party shall keep the other party's Confidential Information "
                "in strict confidence.",
                4,
                0.9,
                {"confidentiality_present": True},
                bbox(72.0, 260.0, 520.0, 300.0),
            ),
            make_clause(
                "liability_cap_009",
                "liability_cap",
                "The aggregate liability of either party shall not exceed the "
                "total fees paid under this Agreement.",
                5,
                0.9,
                {"liability_cap_present": True},
                bbox(72.0, 300.0, 512.0, 342.0),
            ),
            make_clause(
                "termination_009",
                "termination",
                "Either party may terminate for material breach upon thirty (30) "
                "days written notice.",
                5,
                0.9,
                {},
                bbox(72.0, 350.0, 510.0, 392.0),
            ),
            make_clause(
                "governing_law_009",
                "governing_law",
                "This Agreement shall be governed by the laws of Germany.",
                6,
                0.9,
                {"governing_law": "Germany"},
                bbox(72.0, 400.0, 489.0, 430.0),
            ),
            make_clause(
                "gdpr_data_processing_009",
                "gdpr_data_processing",
                "The parties shall process personal data in accordance with the "
                "GDPR and applicable data protection law.",
                7,
                0.9,
                {"gdpr_clause_present": True},
                bbox(72.0, 100.0, 522.0, 175.0),
            ),
            make_clause(
                "signature_block_009",
                "signature_block",
                "IN WITNESS WHEREOF, the parties have executed this Standard "
                "Services Agreement as of the Effective Date.",
                8,
                0.93,
                {},
                bbox(72.0, 500.0, 520.0, 540.0),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Extension-only helper (NOT one of the 10 official scenario fixtures)
# ---------------------------------------------------------------------------
def build_unknown_vendor_extension() -> dict:
    """Unknown Vendor Master Match case -> procurement (unknown_vendor).

    EXTENSION ONLY. This is intentionally NOT generated as scenario 09 (which is
    now the Clean Services Agreement Under Threshold). It is retained as a future
    extension / Agent C unit-test helper. The counterparty "Acme Rogue Systems
    LLC" is absent from policies/vendor_master.csv, so it would trigger the
    unknown_vendor flag. It is not registered in SCENARIOS and never written to
    data/synthetic_fixtures/.
    """
    return make_contract(
        contract_id="contract_unknown_vendor",
        source_file="acme_rogue_systems_agreement.pdf",
        extraction_mode="pdf_parser",
        overall_confidence_score=0.92,
        clauses=[
            make_clause(
                "counterparty_ext",
                "counterparty",
                f"This Agreement is entered into with {UNKNOWN_VENDOR_NAME}.",
                1,
                0.95,
                {"counterparty_name": UNKNOWN_VENDOR_NAME},
                bbox(72.0, 120.0, 490.0, 152.0),
            ),
            make_clause(
                "payment_terms_ext",
                "payment_terms",
                "Payment is due Net 30 from invoice receipt.",
                3,
                0.91,
                {"payment_terms_days": 30},
                bbox(72.0, 210.0, 450.0, 244.0),
            ),
            make_clause(
                "confidentiality_ext",
                "confidentiality",
                "Each party shall keep the other party's Confidential Information "
                "in strict confidence.",
                4,
                0.9,
                {"confidentiality_present": True},
                bbox(72.0, 260.0, 520.0, 300.0),
            ),
            make_clause(
                "liability_cap_ext",
                "liability_cap",
                "Total liability shall not exceed the fees paid under this "
                "Agreement.",
                5,
                0.9,
                {"liability_cap_present": True},
                bbox(72.0, 300.0, 512.0, 342.0),
            ),
            make_clause(
                "termination_ext",
                "termination",
                "Either party may terminate for material breach upon thirty (30) "
                "days written notice.",
                5,
                0.9,
                {},
                bbox(72.0, 350.0, 510.0, 392.0),
            ),
            make_clause(
                "governing_law_ext",
                "governing_law",
                "This Agreement shall be governed by the laws of Germany.",
                6,
                0.9,
                {"governing_law": "Germany"},
                bbox(72.0, 400.0, 489.0, 430.0),
            ),
            make_clause(
                "gdpr_data_processing_ext",
                "gdpr_data_processing",
                "The parties shall process personal data in accordance with the "
                "GDPR and applicable data protection law.",
                7,
                0.9,
                {"gdpr_clause_present": True},
                bbox(72.0, 100.0, 522.0, 175.0),
            ),
        ],
    )


def scenario_10() -> dict:
    """SC10 - Multiple High Risks / Stress Test -> compliance review required.

    Combines several simultaneous risks so decision-priority logic can be tested:
        - payment terms above policy (120 days)  -> payment_terms_exceeded
        - missing liability cap                   -> missing_liability_cap
        - high-risk jurisdiction (Iran)           -> high_risk_jurisdiction
        - missing GDPR clause (personal data set) -> missing_gdpr_clause
        - auto-renewal without opt-out            -> auto_renewal_without_opt_out

    extraction_mode = synthetic_fallback, so ALL bbox values are null. The
    governing law text keeps the long legal name while structured_fields is
    normalized to "Iran". No liability_cap clause and no gdpr_data_processing
    clause are emitted.
    """
    return make_contract(
        contract_id="contract_010",
        source_file="oscorp_offshore_master_agreement.pdf",
        extraction_mode="synthetic_fallback",
        overall_confidence_score=0.45,
        clauses=[
            make_clause(
                "counterparty_010",
                "counterparty",
                "This Master Agreement is entered into with Oscorp Offshore LLC.",
                1,
                0.5,
                {"counterparty_name": "Oscorp Offshore LLC"},
                None,
            ),
            make_clause(
                "data_scope_010",
                "other",
                "Under this Agreement Oscorp processes personal data of data "
                "subjects located in the European Union, including identifiers and "
                "financial records.",
                2,
                0.46,
                {"contains_personal_data": True},
                None,
            ),
            make_clause(
                "payment_terms_010",
                "payment_terms",
                "Payment shall be made within 120 days of invoice receipt.",
                3,
                0.44,
                {"payment_terms_days": 120},
                None,
            ),
            make_clause(
                "auto_renewal_010",
                "auto_renewal",
                "This Agreement shall automatically renew for successive annual "
                "terms.",
                4,
                0.42,
                {"auto_renewal_present": True, "opt_out_window_days": None},
                None,
            ),
            make_clause(
                "governing_law_010",
                "governing_law",
                "This Agreement shall be governed by the laws of, and the legal "
                "seat of performance shall be in, The Islamic Republic of Iran.",
                6,
                0.43,
                {"governing_law": "Iran"},
                None,
            ),
        ],
    )


# Ordered registry of scenario builders (exactly 10 files).
SCENARIOS = [
    scenario_01,
    scenario_02,
    scenario_03,
    scenario_04,
    scenario_05,
    scenario_06,
    scenario_07,
    scenario_08,
    scenario_09,
    scenario_10,
]


# ---------------------------------------------------------------------------
# Validation-result fixture (Agent D) for the Net 90 scenario
# ---------------------------------------------------------------------------
def build_scenario_03_validation_result() -> dict:
    """Net 90 validation result aligned with scenario_03 extraction.

    Only fields actually supported by the scenario_03 extraction are populated:
    payment_terms_days, governing_law, liability_cap_present, counterparty_name.
    effective_date / gdpr_clause_present / auto_renewal_present / opt_out_window_days
    are left null because scenario_03 has no corresponding clause/value. Because
    the extraction overall_confidence_score is 0.6, has_low_confidence_extraction
    is true (it must not be reported as false).
    """
    return {
        "schema_version": "1.0",
        "contract_id": "contract_003",
        "validation_status": "completed_with_findings",
        "normalized_fields": {
            "payment_terms_days": 90,
            "governing_law": "Germany",
            "effective_date": None,
            "liability_cap_present": True,
            "gdpr_clause_present": None,
            "auto_renewal_present": None,
            "opt_out_window_days": None,
            "counterparty_name": "Acme Services GmbH",
            "has_low_confidence_extraction": True,
        },
        "findings": [
            {
                "finding_id": "VAL-001",
                "field": "payment_terms_days",
                "severity": "high",
                "message": "Payment terms normalized to 90 days, which exceeds the "
                "maximum allowed Net 30 policy.",
                "evidence_ref": "contract.pdf#page=3&clause=payment_terms_001",
                "clause_id": "payment_terms_001",
                "page_number": 3,
                "policy_rule": "payment_terms_exceeded",
                "expected_value": "30",
                "actual_value": "90",
                "recommendation": "Route to finance for payment-terms review.",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Deterministic custom assertions (raise clear ValueError on failure)
# ---------------------------------------------------------------------------
def _clause_types(contract: dict) -> list[str]:
    return [c["clause_type"] for c in contract["clauses"]]


def _governing_law_values(contract: dict) -> list[str]:
    return [
        c["structured_fields"].get("governing_law")
        for c in contract["clauses"]
        if c["clause_type"] == "governing_law"
    ]


def assert_scenario_02_has_no_liability_cap(contract: dict) -> None:
    if "liability_cap" in _clause_types(contract):
        raise ValueError("Scenario 02 must NOT include a liability_cap clause.")


def assert_scenario_03_all_bboxes_are_null(contract: dict) -> None:
    if any(c["bbox"] is not None for c in contract["clauses"]):
        raise ValueError("Scenario 03 must keep ALL bbox values null.")


def assert_scenario_04_governing_law_is_russia(contract: dict) -> None:
    values = _governing_law_values(contract)
    if values != ["Russia"]:
        raise ValueError(
            "Scenario 04 normalized governing_law must equal exactly 'Russia', "
            f"got {values!r}."
        )


def assert_scenario_05_has_auto_renewal_without_opt_out(contract: dict) -> None:
    auto = [c for c in contract["clauses"] if c["clause_type"] == "auto_renewal"]
    if not auto:
        raise ValueError("Scenario 05 must include an auto_renewal clause.")
    sf = auto[0]["structured_fields"]
    if sf.get("auto_renewal_present") is not True:
        raise ValueError("Scenario 05 auto_renewal_present must be true.")
    if sf.get("opt_out_window_days") is not None:
        raise ValueError("Scenario 05 opt_out_window_days must be null (no opt-out).")


def assert_scenario_06_has_two_conflicting_governing_laws(contract: dict) -> None:
    gl = [c for c in contract["clauses"] if c["clause_type"] == "governing_law"]
    if len(gl) != 2:
        raise ValueError("Scenario 06 must contain exactly two governing_law clauses.")
    normalized = {c["structured_fields"].get("governing_law") for c in gl}
    if normalized != {"Germany", "New York"}:
        raise ValueError(
            "Scenario 06 governing_law clauses must normalize to Germany and "
            f"New York, got {normalized!r}."
        )
    pages = {c["page_number"] for c in gl}
    if len(pages) != 1:
        raise ValueError(
            "Scenario 06 conflicting governing_law clauses must share the same "
            f"page_number, got pages {sorted(pages)!r}."
        )
    ids = {c["clause_id"] for c in gl}
    if ids != {"governing_law_006_a", "governing_law_006_b"}:
        raise ValueError(
            "Scenario 06 must use clause IDs governing_law_006_a and "
            f"governing_law_006_b, got {sorted(ids)!r}."
        )


def assert_scenario_07_signature_confidence_is_low(contract: dict) -> None:
    sig = [c for c in contract["clauses"] if c["clause_type"] == "signature_block"]
    if not sig:
        raise ValueError("Scenario 07 must include a signature_block clause.")
    if sig[0]["confidence_score"] >= 0.6:
        raise ValueError(
            "Scenario 07 signature_block confidence must be low (< 0.6), got "
            f"{sig[0]['confidence_score']!r}."
        )


def assert_scenario_08_has_no_gdpr_clause(contract: dict) -> None:
    if "gdpr_data_processing" in _clause_types(contract):
        raise ValueError("Scenario 08 must NOT include a gdpr_data_processing clause.")
    has_personal_data = any(
        c["structured_fields"].get("contains_personal_data") is True
        for c in contract["clauses"]
    )
    if not has_personal_data:
        raise ValueError(
            "Scenario 08 must signal contains_personal_data=true (via an 'other' "
            "scope clause) so missing_gdpr_clause is derivable."
        )


def assert_scenario_09_is_clean_services_under_threshold(contract: dict) -> None:
    # Scenario 09 is the official Clean Services Agreement Under Threshold case.
    if contract["contract_id"] != "contract_009":
        raise ValueError("Scenario 09 must use contract_id 'contract_009'.")

    counterparties = [
        c["structured_fields"].get("counterparty_name")
        for c in contract["clauses"]
        if c["clause_type"] == "counterparty"
    ]
    if counterparties != ["Hooli Services GmbH"]:
        raise ValueError(
            "Scenario 09 must use an approved known vendor, preferably "
            f"'Hooli Services GmbH', got {counterparties!r}."
        )

    if any(
        c["structured_fields"].get("counterparty_name") == UNKNOWN_VENDOR_NAME
        for c in contract["clauses"]
    ):
        raise ValueError(
            f"Scenario 09 must not use {UNKNOWN_VENDOR_NAME!r}."
        )

    # Counterparty must be an approved vendor present in vendor_master.csv.
    known_vendor_names = {row[1] for row in VENDOR_MASTER_ROWS[1:]}
    if "Hooli Services GmbH" not in known_vendor_names:
        raise ValueError(
            "Scenario 09 counterparty must exist in vendor_master.csv "
            "(approved known vendor)."
        )

    payment_terms = [
        c for c in contract["clauses"] if c["clause_type"] == "payment_terms"
    ]
    if (
        not payment_terms
        or payment_terms[0]["structured_fields"].get("payment_terms_days") != 30
    ):
        raise ValueError("Scenario 09 must have Net 30 payment terms.")

    if "liability_cap" not in _clause_types(contract):
        raise ValueError("Scenario 09 must include a liability_cap clause.")

    if _governing_law_values(contract) != ["Germany"]:
        raise ValueError("Scenario 09 must use Germany as governing law.")

    if "gdpr_data_processing" not in _clause_types(contract):
        raise ValueError("Scenario 09 must include a gdpr_data_processing clause.")

    auto = [c for c in contract["clauses"] if c["clause_type"] == "auto_renewal"]
    if auto:
        raise ValueError(
            "Scenario 09 must not include an auto_renewal clause "
            "(no auto_renewal_without_opt_out risk)."
        )


def assert_scenario_10_has_multiple_high_risks(contract: dict) -> None:
    types = _clause_types(contract)
    payment = [c for c in contract["clauses"] if c["clause_type"] == "payment_terms"]
    if not payment or payment[0]["structured_fields"].get("payment_terms_days") != 120:
        raise ValueError("Scenario 10 must include payment terms of 120 days.")
    if "liability_cap" in types:
        raise ValueError("Scenario 10 must NOT include a liability_cap clause.")
    if "gdpr_data_processing" in types:
        raise ValueError("Scenario 10 must NOT include a gdpr_data_processing clause.")
    if _governing_law_values(contract) != ["Iran"]:
        raise ValueError("Scenario 10 normalized governing_law must equal 'Iran'.")
    if not any(
        c["structured_fields"].get("contains_personal_data") is True
        for c in contract["clauses"]
    ):
        raise ValueError(
            "Scenario 10 must signal contains_personal_data=true so "
            "missing_gdpr_clause is derivable."
        )


def assert_scenario_10_all_bboxes_are_null(contract: dict) -> None:
    if any(c["bbox"] is not None for c in contract["clauses"]):
        raise ValueError("Scenario 10 must keep ALL bbox values null.")


# Map 1-based scenario index -> custom assertions that apply to it.
SCENARIO_ASSERTIONS = {
    2: [assert_scenario_02_has_no_liability_cap],
    3: [assert_scenario_03_all_bboxes_are_null],
    4: [assert_scenario_04_governing_law_is_russia],
    5: [assert_scenario_05_has_auto_renewal_without_opt_out],
    6: [assert_scenario_06_has_two_conflicting_governing_laws],
    7: [assert_scenario_07_signature_confidence_is_low],
    8: [assert_scenario_08_has_no_gdpr_clause],
    9: [assert_scenario_09_is_clean_services_under_threshold],
    10: [
        assert_scenario_10_has_multiple_high_risks,
        assert_scenario_10_all_bboxes_are_null,
    ],
}


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_vendor_master() -> None:
    """Write the deterministic trusted vendor master CSV."""
    POLICIES_DIR.mkdir(parents=True, exist_ok=True)
    lines = [",".join(row) for row in VENDOR_MASTER_ROWS]
    VENDOR_MASTER_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Generate, strictly validate, and serialize all fixtures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if len(SCENARIOS) != 10:
        raise RuntimeError(f"Expected exactly 10 scenarios, found {len(SCENARIOS)}.")

    written: list[str] = []

    # 1) Deterministic policy data.
    write_vendor_master()
    written.append(str(VENDOR_MASTER_PATH.relative_to(REPO_ROOT)))

    # 2) Extraction fixtures (Agent B).
    for index, builder in enumerate(SCENARIOS, start=1):
        data = builder()

        # Strict schema validation against the real Pydantic model.
        try:
            ExtractedContract.model_validate(data)
        except Exception as exc:  # noqa: BLE001 - re-raised with scenario context
            raise ValueError(
                f"Scenario {index:02d} failed ExtractedContract validation: {exc}"
            ) from exc

        # Scenario-specific deterministic invariants.
        for check in SCENARIO_ASSERTIONS.get(index, []):
            check(data)

        file_path = OUTPUT_DIR / f"scenario_{index:02d}_extracted_contract.json"
        write_json(file_path, data)
        written.append(str(file_path.relative_to(REPO_ROOT)))
        print(f"[OK] wrote and validated {file_path.relative_to(REPO_ROOT)}")

    # 3) Validation-result fixture (Agent D) for the Net 90 scenario.
    validation_payload = build_scenario_03_validation_result()
    try:
        ValidationResult.model_validate(validation_payload)
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise ValueError(
            f"scenario_03 validation result failed ValidationResult validation: {exc}"
        ) from exc

    canonical_validation = OUTPUT_DIR / "scenario_03_validation_result.json"
    compat_validation = OUTPUT_DIR / "validation_result_net90.json"
    write_json(canonical_validation, validation_payload)
    write_json(compat_validation, validation_payload)
    written.append(str(canonical_validation.relative_to(REPO_ROOT)))
    written.append(str(compat_validation.relative_to(REPO_ROOT)))
    print(f"[OK] wrote and validated {canonical_validation.relative_to(REPO_ROOT)}")
    print(
        f"[OK] wrote compatibility copy {compat_validation.relative_to(REPO_ROOT)} "
        "(read by tests/test_schema_fixtures.py)"
    )

    # 4) Success report.
    print("\n" + "=" * 70)
    print("ICRAS fixture generation complete.")
    print("=" * 70)
    print(f"Files written ({len(written)}):")
    for item in written:
        print(f"  - {item}")
    print(
        "\nScenario 09 resolution: restored to Clean Services Agreement Under "
        "Threshold to match the official scenario matrix (counterparty 'Hooli "
        "Services GmbH', an approved known vendor in vendor_master.csv; Net 30, "
        "liability cap present, Germany governing law, GDPR clause present). The "
        f"Unknown Vendor case (counterparty '{UNKNOWN_VENDOR_NAME}') is retained "
        "only as a documented extension helper (build_unknown_vendor_extension) "
        "and is NOT generated as scenario 09."
    )
    print(
        "Scenario 08 / Scenario 10 missing-GDPR handling: personal-data processing "
        "is encoded inside an 'other' scope clause via "
        "structured_fields.contains_personal_data=true (schema-safe, since "
        "extra=\"forbid\" applies to top-level fields, not structured_fields "
        "values). No gdpr_data_processing clause is emitted, and no unknown "
        "top-level field is added. This represents the upstream context/manifest "
        "signal that no schema currently models."
    )


if __name__ == "__main__":
    main()
