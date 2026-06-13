from pathlib import Path

from pydantic import ValidationError

from app.schemas.extracted_contract import ExtractedContract
from app.schemas.validation_result import ValidationResult

# This test file proves that the mock JSON fixtures match the Pydantic schemas.
# It also proves that strict validation rejects unexpected fields.

FIXTURES_DIR = Path("data/synthetic_fixtures")


def test_extracted_contract_fixture_is_valid() -> None:
    # Load the synthetic Agent B output fixture.
    fixture_path = FIXTURES_DIR / "scenario_03_extracted_contract.json"

    # Pydantic validates the full JSON structure here.
    extracted_contract = ExtractedContract.from_json_file(fixture_path)

    # These assertions confirm that important extracted fields are readable.
    assert extracted_contract.contract_id == "contract_003"
    assert extracted_contract.extraction_mode == "synthetic_fallback"
    assert extracted_contract.overall_confidence_score == 0.6
    assert extracted_contract.clauses[0].clause_type == "payment_terms"
    assert extracted_contract.clauses[0].page_number == 3


def test_validation_result_fixture_is_valid() -> None:
    # Load the synthetic Agent D output fixture.
    fixture_path = FIXTURES_DIR / "validation_result_net90.json"

    # Pydantic validates normalized fields and validation findings.
    validation_result = ValidationResult.from_json_file(fixture_path)

    # These assertions confirm that downstream agents can access normalized data.
    assert validation_result.contract_id == "contract_003"
    assert validation_result.validation_status == "completed_with_findings"
    assert validation_result.normalized_fields.payment_terms_days == 90
    assert validation_result.findings[0].finding_id == "VAL-001"


def test_extracted_contract_rejects_extra_fields() -> None:
    # This payload intentionally includes an unexpected field inside a clause.
    # Because the schema uses extra="forbid", validation must fail.
    invalid_payload = {
        "schema_version": "1.0",
        "contract_id": "contract_003",
        "source_file": "contract.pdf",
        "extraction_mode": "synthetic_fallback",
        "overall_confidence_score": 0.6,
        "clauses": [
            {
                "clause_id": "payment_terms_001",
                "clause_type": "payment_terms",
                "text": "Payment shall be made within ninety (90) days.",
                "page_number": 3,
                "confidence_score": 0.6,
                "evidence_ref": "contract.pdf#page=3&clause=payment_terms_001",
                "unexpected_field": "not allowed",
            }
        ],
    }

    try:
        ExtractedContract.model_validate(invalid_payload)
    except ValidationError:
        # Expected result: Pydantic rejects the unexpected field.
        return

    raise AssertionError("Extra fields should be rejected.")
