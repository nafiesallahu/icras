"""Deterministic generator for Agent B (Extraction Agent) JSON fixtures.

This script produces 10 static, fully deterministic ``extracted_contract.json``
fixtures that mock the production output of Agent B. Every fixture strictly
validates against ``app.schemas.extracted_contract.ExtractedContract`` so that
downstream validation and risk-rules engines can be exercised in an offline,
zero-LLM environment.

------------------------------------------------------------------------------
SPEC-TO-MODEL FIELD MAPPING
------------------------------------------------------------------------------
The original specification used generic clause field names. They are mapped to
the real Pydantic model as follows (the model is the source of truth):

    spec ``clause_text``                    -> model ``text``
    spec ``bbox.x_min/y_min/x_max/y_max``   -> model ``bbox.x0/top/x1/bottom``

Each clause additionally carries the model-required ``clause_id``,
``confidence_score``, ``evidence_ref`` and ``structured_fields``.

------------------------------------------------------------------------------
REQUIREMENT TRACEABILITY MATRIX
------------------------------------------------------------------------------
Technical constraints:
    TC1  frozen=True / extra="forbid"  -> every fixture is validated with
         ExtractedContract.model_validate(...) before it is written; fixtures
         only ever contain allowed keys.                  [main(), make_clause]
    TC2  clause fields present         -> make_clause() always emits clause_type,
         text (=spec clause_text), page_number and bbox (=spec bbox), plus the
         required clause_id/confidence_score/evidence_ref/structured_fields.
                                                                  [make_clause]
    TC3  exactly 10 files at
         data/synthetic_fixtures/scenario_[01-10]_extracted_contract.json
                                                              [SCENARIOS, main]

Scenario data contracts (one builder per scenario / file):
    SC01 Clean Corporate Baseline      -> scenario_01()
    SC02 Missing Liability Cap         -> scenario_02()
    SC03 Net 90 (test-compatible)      -> scenario_03()
    SC04 High-Risk Jurisdiction        -> scenario_04()
    SC05 Auto Renewal No Opt-Out       -> scenario_05()
    SC06 Conflicting Governing Law     -> scenario_06()
    SC07 Standard NDA                  -> scenario_07()
    SC08 Missing GDPR Clause           -> scenario_08()
    SC09 Unknown Vendor Master Match   -> scenario_09()
    SC10 Compound Critical             -> scenario_10()

Coverage check: TC1-TC3 are each covered once by shared helpers / the write
loop; SC01-SC10 are each covered by exactly one builder and one output file.
No requirement is implemented more than once.
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

OUTPUT_DIR = REPO_ROOT / "data" / "synthetic_fixtures"


# ---------------------------------------------------------------------------
# Helpers (shared construction logic -> guarantees TC1 and TC2)
# ---------------------------------------------------------------------------
def bbox(x0: float, top: float, x1: float, bottom: float) -> dict:
    """Build a model-shaped bounding box (spec x_min/y_min/x_max/y_max)."""
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
    """SC01 - Clean Corporate Baseline.

    Net 30 payment terms, a liability cap matching the contract value,
    governing law = Germany, and a fully articulated GDPR clause.
    """
    return make_contract(
        contract_id="contract_001",
        source_file="acme_master_services_agreement.pdf",
        extraction_mode="pdf_parser",
        overall_confidence_score=0.97,
        clauses=[
            make_clause(
                "counterparty_001",
                "counterparty",
                "This Master Services Agreement is entered into with Globex "
                "Industries GmbH, a company incorporated under the laws of Germany.",
                1,
                0.98,
                {"counterparty_name": "Globex Industries GmbH"},
                bbox(72.0, 120.5, 523.4, 158.9),
            ),
            make_clause(
                "effective_date_001",
                "effective_date",
                "This Agreement shall take effect as of 01 January 2025.",
                1,
                0.96,
                {"effective_date": "2025-01-01"},
                bbox(72.0, 168.0, 410.2, 196.3),
            ),
            make_clause(
                "payment_terms_001",
                "payment_terms",
                "Payment shall be due Net 30 from the date of a valid invoice.",
                4,
                0.97,
                {"payment_terms_days": 30},
                bbox(72.0, 210.0, 498.7, 248.1),
            ),
            make_clause(
                "liability_cap_001",
                "liability_cap",
                "The aggregate liability of either party shall not exceed the "
                "total Contract Value of EUR 500,000 paid under this Agreement.",
                5,
                0.95,
                {
                    "liability_cap_present": True,
                    "liability_cap_amount": 500000,
                    "contract_value": 500000,
                    "currency": "EUR",
                },
                bbox(72.0, 300.4, 521.9, 352.6),
            ),
            make_clause(
                "governing_law_001",
                "governing_law",
                "This Agreement shall be governed by and construed in accordance "
                "with the laws of Germany.",
                6,
                0.98,
                {"governing_law": "Germany"},
                bbox(72.0, 410.0, 489.3, 438.7),
            ),
            make_clause(
                "gdpr_data_processing_001",
                "gdpr_data_processing",
                "The parties shall process personal data in accordance with "
                "Regulation (EU) 2016/679 (GDPR). The Processor shall act only on "
                "documented instructions from the Controller, ensure appropriate "
                "technical and organisational measures under Article 32, support "
                "data subject rights, notify personal data breaches without undue "
                "delay, and delete or return all personal data upon termination.",
                7,
                0.94,
                {"gdpr_clause_present": True},
                bbox(72.0, 100.0, 525.0, 240.8),
            ),
        ],
    )


def scenario_02() -> dict:
    """SC02 - Missing Liability Cap.

    Standard clauses are present, but the liability cap clause is omitted
    entirely so downstream rules can flag the missing cap.
    """
    return make_contract(
        contract_id="contract_002",
        source_file="initech_supply_agreement.pdf",
        extraction_mode="pdf_parser",
        overall_confidence_score=0.9,
        clauses=[
            make_clause(
                "counterparty_002",
                "counterparty",
                "This Supply Agreement is made with Initech Components Ltd.",
                1,
                0.93,
                {"counterparty_name": "Initech Components Ltd"},
                bbox(72.0, 118.0, 470.5, 150.2),
            ),
            make_clause(
                "payment_terms_002",
                "payment_terms",
                "Invoices are payable Net 45 from receipt of goods.",
                3,
                0.91,
                {"payment_terms_days": 45},
                bbox(72.0, 205.0, 460.1, 240.3),
            ),
            make_clause(
                "governing_law_002",
                "governing_law",
                "This Agreement shall be governed by the laws of Germany.",
                6,
                0.92,
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
                0.89,
                {"gdpr_clause_present": True},
                bbox(72.0, 110.0, 522.0, 175.6),
            ),
            make_clause(
                "termination_002",
                "termination",
                "Either party may terminate this Agreement for material breach "
                "upon thirty (30) days written notice.",
                8,
                0.9,
                {},
                bbox(72.0, 220.0, 510.7, 262.9),
            ),
        ],
    )


def scenario_03() -> dict:
    """SC03 - Net 90 Payment Terms (kept compatible with the existing test).

    The existing test asserts contract_003, synthetic_fallback, overall score
    0.6, and clauses[0] = payment_terms on page 3. The payment text explicitly
    contains "within 90 days of invoice receipt" to exercise text-to-integer
    normalizers.
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
        ],
    )


def scenario_04() -> dict:
    """SC04 - High-Risk Jurisdiction.

    The legal-seat text names a high-risk jurisdiction ("The Russian
    Federation"). The alternative phrasing "The Islamic Republic of Iran" is
    documented here as an equivalent trigger.
    """
    return make_contract(
        contract_id="contract_004",
        source_file="vostok_energy_services_agreement.pdf",
        extraction_mode="pdf_parser",
        overall_confidence_score=0.88,
        clauses=[
            make_clause(
                "counterparty_004",
                "counterparty",
                "This Agreement is entered into with Vostok Energy Services OJSC.",
                1,
                0.9,
                {"counterparty_name": "Vostok Energy Services OJSC"},
                bbox(72.0, 116.0, 480.0, 148.5),
            ),
            make_clause(
                "payment_terms_004",
                "payment_terms",
                "Payment terms are Net 30 from the date of invoice.",
                3,
                0.89,
                {"payment_terms_days": 30},
                bbox(72.0, 210.0, 455.0, 244.0),
            ),
            make_clause(
                "governing_law_004",
                "governing_law",
                "This Agreement shall be governed by, and the legal seat of "
                "performance shall be located in, The Russian Federation.",
                6,
                0.87,
                {"governing_law": "The Russian Federation"},
                bbox(72.0, 400.0, 525.0, 446.2),
            ),
            make_clause(
                "liability_cap_004",
                "liability_cap",
                "Total liability shall not exceed the fees paid in the prior "
                "twelve (12) months.",
                5,
                0.86,
                {"liability_cap_present": True},
                bbox(72.0, 300.0, 512.0, 342.0),
            ),
        ],
    )


def scenario_05() -> dict:
    """SC05 - Auto Renewal, No Opt-Out.

    Contains auto-renewal language but deliberately omits any 30-day opt-out
    notice window.
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
                "payment_terms_005",
                "payment_terms",
                "Subscription fees are invoiced annually and due Net 30.",
                3,
                0.9,
                {"payment_terms_days": 30},
                bbox(72.0, 210.0, 460.0, 246.0),
            ),
            make_clause(
                "auto_renewal_005",
                "auto_renewal",
                "This Agreement shall automatically renew for successive twelve "
                "(12) month terms at the end of each then-current term.",
                4,
                0.9,
                {"auto_renewal_present": True},
                bbox(72.0, 300.0, 522.0, 348.0),
            ),
            make_clause(
                "governing_law_005",
                "governing_law",
                "This Agreement shall be governed by the laws of Germany.",
                6,
                0.9,
                {"governing_law": "Germany"},
                bbox(72.0, 410.0, 489.0, 440.0),
            ),
        ],
    )


def scenario_06() -> dict:
    """SC06 - Conflicting Governing Law.

    Contains an explicit data contradiction: one clause names the Courts of
    Frankfurt, Germany, while another states that New York law applies.
    """
    return make_contract(
        contract_id="contract_006",
        source_file="wayne_enterprises_agreement.pdf",
        extraction_mode="pdf_parser",
        overall_confidence_score=0.84,
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
                "governing_law_006_a",
                "governing_law",
                "The Courts of Frankfurt, Germany shall have exclusive "
                "jurisdiction over any dispute arising under this Agreement.",
                6,
                0.83,
                {"governing_law": "Germany", "forum": "Frankfurt, Germany"},
                bbox(72.0, 300.0, 525.0, 346.0),
            ),
            make_clause(
                "governing_law_006_b",
                "governing_law",
                "Notwithstanding the foregoing, New York Law applies to the "
                "interpretation and enforcement of this Agreement.",
                7,
                0.82,
                {"governing_law": "New York"},
                bbox(72.0, 110.0, 520.0, 156.0),
            ),
            make_clause(
                "payment_terms_006",
                "payment_terms",
                "Payment is due Net 30 from invoice date.",
                3,
                0.85,
                {"payment_terms_days": 30},
                bbox(72.0, 210.0, 450.0, 244.0),
            ),
        ],
    )


def scenario_07() -> dict:
    """SC07 - Standard NDA.

    Mocks standard NDA parameters and gracefully omits vendor/payment-specific
    structures without breaking the base schema constraints.
    """
    return make_contract(
        contract_id="contract_007",
        source_file="mutual_nda.pdf",
        extraction_mode="pdf_parser",
        overall_confidence_score=0.93,
        clauses=[
            make_clause(
                "counterparty_007",
                "counterparty",
                "This Mutual Non-Disclosure Agreement is entered into with "
                "Stark Innovations GmbH.",
                1,
                0.94,
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
                0.93,
                {"confidentiality_term_years": 5},
                bbox(72.0, 200.0, 525.0, 260.0),
            ),
            make_clause(
                "termination_007",
                "termination",
                "The confidentiality obligations shall survive for five (5) "
                "years following termination of this Agreement.",
                3,
                0.92,
                {},
                bbox(72.0, 300.0, 515.0, 344.0),
            ),
            make_clause(
                "governing_law_007",
                "governing_law",
                "This Agreement shall be governed by the laws of Germany.",
                4,
                0.93,
                {"governing_law": "Germany"},
                bbox(72.0, 400.0, 489.0, 430.0),
            ),
            make_clause(
                "signature_block_007",
                "signature_block",
                "IN WITNESS WHEREOF, the parties have executed this Agreement as "
                "of the Effective Date.",
                5,
                0.91,
                {},
                bbox(72.0, 500.0, 520.0, 540.0),
            ),
        ],
    )


def scenario_08() -> dict:
    """SC08 - Missing GDPR Clause.

    Typical commercial operations, but the GDPR_DATA_PROCESSING clause entry is
    completely omitted.
    """
    return make_contract(
        contract_id="contract_008",
        source_file="hooli_services_agreement.pdf",
        extraction_mode="pdf_parser",
        overall_confidence_score=0.9,
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
                "payment_terms_008",
                "payment_terms",
                "Payment is due Net 30 from the date of invoice.",
                3,
                0.91,
                {"payment_terms_days": 30},
                bbox(72.0, 210.0, 450.0, 244.0),
            ),
            make_clause(
                "liability_cap_008",
                "liability_cap",
                "Aggregate liability shall not exceed the total fees paid under "
                "this Agreement.",
                5,
                0.9,
                {"liability_cap_present": True},
                bbox(72.0, 300.0, 512.0, 342.0),
            ),
            make_clause(
                "governing_law_008",
                "governing_law",
                "This Agreement shall be governed by the laws of Germany.",
                6,
                0.9,
                {"governing_law": "Germany"},
                bbox(72.0, 400.0, 489.0, 430.0),
            ),
            make_clause(
                "termination_008",
                "termination",
                "Either party may terminate for convenience upon sixty (60) "
                "days written notice.",
                8,
                0.89,
                {},
                bbox(72.0, 220.0, 510.0, 262.0),
            ),
        ],
    )


def scenario_09() -> dict:
    """SC09 - Unknown Vendor Master Match.

    The counterparty name reads exactly "Acme Rogue Systems LLC" to
    intentionally fail downstream CSV vendor-master matching filters.
    """
    return make_contract(
        contract_id="contract_009",
        source_file="acme_rogue_systems_agreement.pdf",
        extraction_mode="pdf_parser",
        overall_confidence_score=0.9,
        clauses=[
            make_clause(
                "counterparty_009",
                "counterparty",
                "This Agreement is entered into with Acme Rogue Systems LLC.",
                1,
                0.95,
                {"counterparty_name": "Acme Rogue Systems LLC"},
                bbox(72.0, 120.0, 490.0, 152.0),
            ),
            make_clause(
                "payment_terms_009",
                "payment_terms",
                "Payment is due Net 30 from invoice receipt.",
                3,
                0.9,
                {"payment_terms_days": 30},
                bbox(72.0, 210.0, 450.0, 244.0),
            ),
            make_clause(
                "liability_cap_009",
                "liability_cap",
                "Total liability shall not exceed the fees paid under this "
                "Agreement.",
                5,
                0.89,
                {"liability_cap_present": True},
                bbox(72.0, 300.0, 512.0, 342.0),
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
        ],
    )


def scenario_10() -> dict:
    """SC10 - Multiple High Risks and Critical Cumulative Score.

    A compound worst-case file combining: 120 day payment terms, an untrusted
    jurisdiction, no liability cap, and no GDPR clause.
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
                "payment_terms_010",
                "payment_terms",
                "Payment shall be made within 120 days of invoice receipt.",
                3,
                0.44,
                {"payment_terms_days": 120},
                None,
            ),
            make_clause(
                "governing_law_010",
                "governing_law",
                "This Agreement shall be governed by the laws of, and the legal "
                "seat of performance shall be in, The Islamic Republic of Iran.",
                6,
                0.43,
                {"governing_law": "The Islamic Republic of Iran"},
                None,
            ),
            make_clause(
                "auto_renewal_010",
                "auto_renewal",
                "This Agreement shall automatically renew for successive annual "
                "terms.",
                7,
                0.42,
                {"auto_renewal_present": True},
                None,
            ),
        ],
    )


# Ordered registry of scenario builders (TC3: exactly 10 files).
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


def main() -> None:
    """Generate, strictly validate, and serialize all 10 fixtures."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if len(SCENARIOS) != 10:
        raise RuntimeError(f"Expected exactly 10 scenarios, found {len(SCENARIOS)}.")

    for index, builder in enumerate(SCENARIOS, start=1):
        data = builder()

        # TC1: prove the dict strictly validates before it is ever written.
        ExtractedContract.model_validate(data)

        file_path = OUTPUT_DIR / f"scenario_{index:02d}_extracted_contract.json"
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=4)
            handle.write("\n")

        print(f"[OK] wrote and validated {file_path.relative_to(REPO_ROOT)}")

    print(f"Generated {len(SCENARIOS)} fixtures in {OUTPUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
