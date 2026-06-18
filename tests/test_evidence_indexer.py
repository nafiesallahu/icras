from app.schemas.evidence import EvidenceIndex, EvidenceItem
from app.schemas.extracted_contract import BoundingBox
from app.services.evidence_indexer import (
    EvidenceIndexer,
    EvidenceLookupError,
)


def test_find_by_clause_id():
    evidence_index = EvidenceIndex(
        evidence_items=[
            EvidenceItem(
                evidence_id="EV-001",
                document_id="DOC-001",
                filename="contract.pdf",
                page=1,
                clause_id="payment_terms_001",
                text_excerpt="Payment terms shall be 90 days.",
            )
        ]
    )

    indexer = EvidenceIndexer(evidence_index)

    result = indexer.find_by_clause_id(
        "payment_terms_001"
    )

    assert result.evidence_id == "EV-001"


def test_find_by_text():
    evidence_index = EvidenceIndex(
        evidence_items=[
            EvidenceItem(
                evidence_id="EV-001",
                document_id="DOC-001",
                filename="contract.pdf",
                page=1,
                clause_id="payment_terms_001",
                text_excerpt="Payment terms shall be 90 days.",
            )
        ]
    )

    indexer = EvidenceIndexer(evidence_index)

    result = indexer.find_by_text(
        "payment terms"
    )

    assert result.clause_id == "payment_terms_001"


def test_missing_clause_id_raises():
    evidence_index = EvidenceIndex()

    indexer = EvidenceIndexer(evidence_index)

    try:
        indexer.find_by_clause_id(
            "unknown_clause"
        )
        assert False
    except EvidenceLookupError:
        assert True

def test_returns_page_and_bbox_information():
    evidence_index = EvidenceIndex(
        evidence_items=[
            EvidenceItem(
                evidence_id="EV-001",
                document_id="DOC-001",
                filename="contract.pdf",
                page=4,
                clause_id="payment_terms_001",
                bbox=BoundingBox(
                    x0=10,
                    top=20,
                    x1=100,
                    bottom=40,
                ),
                text_excerpt="Payment terms shall be 90 days.",
            )
        ]
    )

    indexer = EvidenceIndexer(evidence_index)

    result = indexer.find_by_clause_id(
        "payment_terms_001"
    )

    assert result.page == 4

    assert result.bbox is not None

    assert result.bbox.x0 == 10
    assert result.bbox.top == 20


def test_get_location_for_clause():
    evidence_index = EvidenceIndex(
        evidence_items=[
            EvidenceItem(
                evidence_id="EV-001",
                document_id="DOC-001",
                filename="contract.pdf",
                page=5,
                clause_id="gdpr_001",
                bbox=BoundingBox(
                    x0=10,
                    top=20,
                    x1=100,
                    bottom=40,
                ),
                text_excerpt="GDPR clause",
            )
        ]
    )

    indexer = EvidenceIndexer(evidence_index)

    location = indexer.get_location_for_clause(
        "gdpr_001"
    )

    assert location.page_number == 5
    assert location.evidence_ref == "EV-001"
    assert location.bbox is not None

from app.schemas.validation_result import (
    Severity,
    ValidationFinding,
)


def test_enrich_finding_location_from_clause_id():
    evidence_index = EvidenceIndex(
        evidence_items=[
            EvidenceItem(
                evidence_id="EV-001",
                document_id="DOC-001",
                filename="contract.pdf",
                page=2,
                clause_id="liability_cap_001",
                bbox=BoundingBox(
                    x0=10,
                    top=20,
                    x1=100,
                    bottom=40,
                ),
                text_excerpt="Liability cap clause",
            )
        ]
    )

    finding = ValidationFinding(
        finding_id="VAL-001",
        field="liability_cap_present",
        severity=Severity.HIGH,
        message="Liability cap is missing.",
        evidence_ref="missing:liability_cap",
        clause_id="liability_cap_001",
        policy_rule="liability_cap.required",
        expected_value="present",
        actual_value="missing",
        recommendation="Add liability cap.",
    )

    enriched = EvidenceIndexer(evidence_index).enrich_finding_location(
        finding
    )

    assert enriched.page_number == 2
    assert enriched.bbox is not None
    assert enriched.evidence_ref == "EV-001"


def test_enrich_finding_location_from_evidence_index_file(tmp_path):
    evidence_index = EvidenceIndex(
        evidence_items=[
            EvidenceItem(
                evidence_id="EV-001",
                document_id="DOC-001",
                filename="contract.pdf",
                page=3,
                clause_id="payment_terms_001",
                bbox=BoundingBox(
                    x0=10,
                    top=20,
                    x1=100,
                    bottom=40,
                ),
                text_excerpt="Payment terms shall be 90 days.",
            )
        ]
    )

    evidence_file = tmp_path / "evidence_index.json"
    evidence_index.to_json_file(evidence_file)

    loaded_index = EvidenceIndex.from_json_file(evidence_file)

    finding = ValidationFinding(
        finding_id="VAL-001",
        field="payment_terms_days",
        severity=Severity.HIGH,
        message="Payment terms exceed policy.",
        evidence_ref="contract.pdf#page=3&clause=payment_terms_001",
        clause_id="payment_terms_001",
        policy_rule="payment_terms.max_allowed_days",
        expected_value="30",
        actual_value="90",
        recommendation="Negotiate payment terms down to 30 days.",
    )

    enriched = EvidenceIndexer(loaded_index).enrich_finding_location(
        finding
    )

    assert enriched.page_number == 3
    assert enriched.bbox is not None
    assert enriched.bbox.top == 20
    assert enriched.evidence_ref == "EV-001"