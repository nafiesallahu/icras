from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.finding import UnifiedFinding

# These are the only counterparty classifications Agent C is allowed to produce.
#
# approved:
# The vendor exists in vendor_master.csv and is approved.
#
# new:
# The vendor exists in vendor_master.csv but is classified as new.
#
# unknown:
# No acceptable vendor match was found.
#
# high_risk:
# The vendor exists, but its vendor master record identifies it as high risk.
CounterpartyStatus = Literal[
    "approved",
    "new",
    "unknown",
    "high_risk",
]


# These values are used by downstream validation, risk scoring,
# exception triage, and approval-routing agents.
RiskLevel = Literal[
    "low",
    "medium",
    "high",
]


class NormalizedCounterparty(BaseModel):
    """
    Validated output produced by Agent C.

    Agent C will create one instance of this model and serialize it into:

        runs/<run_id>/normalized_counterparty.json

    The model records:
    - which counterparty names were received;
    - which official vendor was matched;
    - the fuzzy-match score;
    - the final counterparty classification;
    - risk flags;
    - standardized findings requiring downstream review.
    """

    model_config = ConfigDict(
        # Reject fields that are not explicitly defined in this schema.
        # This prevents accidental or inconsistent JSON keys.
        extra="forbid",
        # Prevent model attributes from being reassigned after creation.
        # This protects the normalized result from accidental modification.
        frozen=True,
        # Prevent unsafe automatic conversions such as "97" into integer 97.
        # The service must provide values using their correct data types.
        strict=True,
        # Remove unnecessary whitespace from string values.
        # For example, "  Acme GmbH  " becomes "Acme GmbH".
        str_strip_whitespace=True,
    )

    # The name Agent C selected as the main value to resolve.
    # This will normally be based on the extracted or manifest counterparty.
    input_counterparty_name: str = Field(
        ...,
        min_length=1,
        description="Counterparty name submitted to the vendor-matching service.",
    )

    # The counterparty declared in context_packet.json,
    # originally supplied through manifest.yaml.
    #
    # It is nullable because some contract bundles might not provide it.
    manifest_counterparty_name: str | None = Field(
        default=None,
        min_length=1,
        description="Counterparty name declared in the contract manifest.",
    )

    # The counterparty detected by Agent B inside extracted_contract.json.
    #
    # It is nullable because extraction may fail or the contract may not
    # clearly identify the counterparty.
    extracted_counterparty_name: str | None = Field(
        default=None,
        min_length=1,
        description="Counterparty name extracted from the contract.",
    )

    # Official vendor identifier from vendor_master.csv.
    #
    # This is null when no acceptable vendor match was found.
    matched_vendor_id: str | None = Field(
        default=None,
        min_length=1,
        description="Official identifier of the matched vendor.",
    )

    # Official vendor name from vendor_master.csv.
    #
    # This is null when the counterparty could not be resolved.
    matched_vendor_name: str | None = Field(
        default=None,
        min_length=1,
        description="Official name of the matched vendor.",
    )

    # Similarity score produced by the fuzzy-matching service.
    #
    # null:
    # Matching could not be performed.
    #
    # 0:
    # No similarity.
    #
    # 100:
    # Exact or fully normalized match.
    match_score: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Fuzzy-match score between 0 and 100.",
    )

    # Final business classification assigned to the counterparty.
    status: CounterpartyStatus = Field(
        ...,
        description="Resolved counterparty status.",
    )

    # Risk level passed to Agents D, E, and H.
    risk_level: RiskLevel = Field(
        ...,
        description="Counterparty risk classification.",
    )

    # Machine-readable signals for downstream deterministic rules.
    #
    # Examples:
    # - counterparty_mismatch
    # - counterparty_not_found
    # - high_risk_counterparty
    # - manual_review_required
    #
    # default_factory creates a separate empty list for every model instance.
    flags: list[str] = Field(
        default_factory=list,
        description="Machine-readable risk and workflow flags.",
    )

    # Findings use the unified UnifiedFinding shared by Agents C, D, and E.
    #
    # Pydantic accepts either:
    # 1. an existing UnifiedFinding object; or
    # 2. a dictionary containing the UnifiedFinding fields.
    #
    # A dictionary is automatically validated and converted into a
    # UnifiedFinding object.
    findings: list[UnifiedFinding] = Field(
        default_factory=list,
        description="Standardized findings generated by Agent C.",
    )
