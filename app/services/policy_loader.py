from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr


POLICIES_DIR = Path("policies")

NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
RiskScore = Annotated[StrictInt, Field(ge=0, le=100)]


class PolicyLoaderError(Exception):
    """Raised when a policy file is missing or invalid."""
    pass


class StrictPolicyModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class RiskThreshold(StrictPolicyModel):
    min_score: RiskScore
    max_score: RiskScore


class PaymentTermsPolicy(StrictPolicyModel):
    max_allowed_days: NonNegativeInt
    medium_risk_days: NonNegativeInt | None = None
    high_risk_days: NonNegativeInt | None = None
    risk_if_exceeds: StrictStr
    approver: StrictStr


class LiabilityCapPolicy(StrictPolicyModel):
    required: StrictBool
    minimum_cap_type: StrictStr
    risk_if_missing: StrictStr
    approver: StrictStr


class AutoRenewalPolicy(StrictPolicyModel):
    allowed: StrictBool
    opt_out_window_required: StrictBool
    minimum_opt_out_days: NonNegativeInt
    risk_if_missing: StrictStr
    approver: StrictStr


class GdprCondition(StrictPolicyModel):
    contains_personal_data: StrictBool


class GdprPolicy(StrictPolicyModel):
    data_processing_clause_required_if: GdprCondition
    risk_if_missing: StrictStr
    approver: StrictStr


class PlaybookPolicy(StrictPolicyModel):
    version: StrictStr
    owner: StrictStr
    description: StrictStr

    risk_thresholds: dict[str, RiskThreshold]
    payment_terms: PaymentTermsPolicy
    liability_cap: LiabilityCapPolicy
    auto_renewal: AutoRenewalPolicy
    gdpr: GdprPolicy

    required_clauses: list[StrictStr]
    risk_scores: dict[str, RiskScore]

class JurisdictionProfile(StrictPolicyModel):
    risk_level: StrictStr
    risk_score: RiskScore
    compliance_review_required: StrictBool


class JurisdictionPolicy(StrictPolicyModel):
    version: StrictStr
    owner: StrictStr
    description: StrictStr

    jurisdictions: dict[str, JurisdictionProfile]
    default_jurisdiction: JurisdictionProfile


class ApprovalRule(StrictPolicyModel):
    risk_level: StrictStr
    action: StrictStr
    approver: StrictStr
    category: StrictStr | None = None


class DepartmentRouting(StrictPolicyModel):
    triggers: list[StrictStr]


class ApprovalPolicy(StrictPolicyModel):
    version: StrictStr
    owner: StrictStr
    description: StrictStr

    approval_rules: list[ApprovalRule]
    department_routing: dict[str, DepartmentRouting]
    decision_priority: dict[str, RiskScore]

def _load_yaml_file(file_path: Path) -> dict:
    if not file_path.exists():
        raise PolicyLoaderError(f"Policy file not found: {file_path}")

    try:
        with file_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as error:
        raise PolicyLoaderError(f"Invalid YAML format in {file_path}: {error}") from error

    if not isinstance(data, dict):
        raise PolicyLoaderError(f"Policy file must contain a YAML dictionary: {file_path}")

    return data


def load_playbook(path: str | Path = POLICIES_DIR / "playbook.yaml") -> PlaybookPolicy:
    data = _load_yaml_file(Path(path))

    try:
        return PlaybookPolicy.model_validate(data)
    except Exception as error:
        raise PolicyLoaderError(f"Invalid playbook policy configuration: {error}") from error

    
def load_jurisdictions(
    path: str | Path = POLICIES_DIR / "jurisdiction_rules.yaml",
) -> JurisdictionPolicy:
    data = _load_yaml_file(Path(path))

    try:
        return JurisdictionPolicy.model_validate(data)
    except Exception as error:
        raise PolicyLoaderError(
            f"Invalid jurisdiction policy configuration: {error}"
        ) from error


def load_approval_policy(
    path: str | Path = POLICIES_DIR / "approval_policy.yaml",
) -> ApprovalPolicy:
    data = _load_yaml_file(Path(path))

    try:
        return ApprovalPolicy.model_validate(data)
    except Exception as error:
        raise PolicyLoaderError(
            f"Invalid approval policy configuration: {error}"
        ) from error