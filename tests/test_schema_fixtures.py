from pathlib import Path

from pydantic import ValidationError

from app.schemas.extracted_contract import ExtractedContract
from app.schemas.validation_result import ValidationResult

FIXTURES_DIR = Path("tests/fixtures")


def test_extracted_contract_fixture_is_valid() -> None:
    fixture_path = FIXTURES_DIR / "extracted_contract_net90.json"

    extracted_contract = ExtractedContract.from_json_file(fixture_path)

    assert extracted_contract.contract_id == "contract_003"
    assert extracted_contract.extraction_mode == "synthetic_fallback"
    assert extracted_contract.overall_confidence_score == 0.6
    assert extracted_contract.clauses[0].clause_type == "payment_terms"
    assert extracted_contract.clauses[0].page_number == 3


def test_validation_result_fixture_is_valid() -> None:
    fixture_path = FIXTURES_DIR / "validation_result_net90.json"

    validation_result = ValidationResult.from_json_file(fixture_path)

    assert validation_result.contract_id == "contract_003"
    assert validation_result.validation_status == "completed_with_findings"
    assert validation_result.normalized_fields.payment_terms_days == 90
    assert validation_result.findings[0].finding_id == "VAL-001"


def test_extracted_contract_rejects_extra_fields() -> None:
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
        return

    raise AssertionError("Extra fields should be rejected.")
