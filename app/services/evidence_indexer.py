from dataclasses import dataclass

from app.schemas.evidence import EvidenceIndex, EvidenceItem
from app.schemas.extracted_contract import BoundingBox


class EvidenceLookupError(Exception):
    """Raised when evidence cannot be matched to a query."""


@dataclass(frozen=True)
class EvidenceLocation:
    page_number: int
    bbox: BoundingBox | None
    evidence_ref: str


class EvidenceIndexer:
    def __init__(self, evidence_index: EvidenceIndex):
        self.evidence_index = evidence_index

    def find_by_clause_id(self, clause_id: str) -> EvidenceItem:
        for item in self.evidence_index.evidence_items:
            if item.clause_id == clause_id:
                return item

        raise EvidenceLookupError(
            f"No evidence item found for clause_id: {clause_id}"
        )

    def find_by_text(self, query_text: str) -> EvidenceItem:
        normalized_query = self._normalize_text(query_text)

        for item in self.evidence_index.evidence_items:
            normalized_excerpt = self._normalize_text(item.text_excerpt)

            if normalized_query in normalized_excerpt:
                return item

        raise EvidenceLookupError(
            f"No evidence item found for text query: {query_text}"
        )

    def get_location_for_clause(
        self,
        clause_id: str,
    ) -> EvidenceLocation:
        item = self.find_by_clause_id(clause_id)

        return EvidenceLocation(
            page_number=item.page,
            bbox=item.bbox,
            evidence_ref=item.evidence_id,
        )
    
    def enrich_finding_location(
        self,
        finding,
    ):
        if getattr(finding, "clause_id", None) is None:
            return finding

        location = self.get_location_for_clause(
            finding.clause_id,
        )

        return finding.model_copy(
            update={
                "page_number": location.page_number,
                "bbox": location.bbox,
                "evidence_ref": location.evidence_ref,
            }
        )

    @staticmethod
    def _normalize_text(value: str) -> str:
        return " ".join(value.lower().split())

    