from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from app.schemas.extracted_contract import StrictSchemaModel

PolicyRiskLevel = Literal[
    "low",
    "medium",
    "high",
    "critical",
]

PriorityValue = Annotated[
    StrictInt,
    Field(ge=0),
]

ACTION_PRIORITY_ALIASES: dict[str, str] = {
    "legal_review": "legal_review_required",
    "finance_review": "finance_review_required",
    "compliance_review": "compliance_review_required",
    # Compatibility with existing scenario policies.
    # A direct executive_review priority will still take precedence
    # when a future policy defines one.
    "executive_review": "reject_or_block",
}


def resolve_priority_key(
    action: str,
    decision_priority: dict[str, int],
) -> str:
    """
    Return the priority key associated with an approval action.

    Direct priority entries take precedence over compatibility aliases.
    """

    if action in decision_priority:
        return action

    return ACTION_PRIORITY_ALIASES.get(
        action,
        action,
    )


class ApprovalRule(StrictSchemaModel):
    """
    One approval rule from approval_policy.yaml.

    The rule describes which action and approver apply to a given
    risk level and optional finding category.
    """

    risk_level: PolicyRiskLevel

    category: StrictStr | None = Field(
        default=None,
        min_length=1,
    )

    action: StrictStr = Field(
        ...,
        min_length=1,
    )

    approver: StrictStr = Field(
        ...,
        min_length=1,
    )


class DepartmentRouting(StrictSchemaModel):
    """
    Finding types that belong to one departmental route.
    """

    triggers: list[StrictStr] = Field(
        default_factory=list,
    )

    @field_validator("triggers")
    @classmethod
    def validate_triggers(
        cls,
        values: list[str],
    ) -> list[str]:
        """
        Reject blank triggers and remove duplicates while preserving
        the order defined in YAML.
        """

        normalized: list[str] = []

        for value in values:
            cleaned = value.strip()

            if not cleaned:
                raise ValueError("Department-routing triggers cannot be blank.")

            if cleaned not in normalized:
                normalized.append(cleaned)

        return normalized


class ApprovalPolicy(StrictSchemaModel):
    """
    Validated approval policy consumed by Agent H.

    The loader supports both:

    1. decision_priority as a top-level YAML section;
    2. the current scenario format, where decision_priority is
       nested inside department_routing.
    """

    version: StrictStr = Field(
        ...,
        min_length=1,
    )

    owner: StrictStr = Field(
        ...,
        min_length=1,
    )

    description: StrictStr = Field(
        ...,
        min_length=1,
    )

    approval_rules: list[ApprovalRule] = Field(
        ...,
        min_length=1,
    )

    department_routing: dict[StrictStr, DepartmentRouting] = Field(
        default_factory=dict,
    )

    decision_priority: dict[StrictStr, PriorityValue] = Field(
        ...,
        min_length=1,
    )

    @field_validator("decision_priority")
    @classmethod
    def validate_decision_priority(
        cls,
        values: dict[str, int],
    ) -> dict[str, int]:
        """
        Reject blank action names and ensure that auto-approval has
        a configured priority.
        """

        normalized: dict[str, int] = {}

        for action, priority in values.items():
            cleaned_action = action.strip()

            if not cleaned_action:
                raise ValueError("Decision-priority action names cannot be blank.")

            normalized[cleaned_action] = priority

        if "auto_approve" not in normalized:
            raise ValueError("decision_priority must define auto_approve.")

        return normalized

    @model_validator(mode="after")
    def validate_policy_rules(self) -> "ApprovalPolicy":
        """
        Validate approval-rule uniqueness and decision priorities.

        Every rule action must resolve either directly or through a known
        compatibility alias to a key in decision_priority.
        """

        seen_rules: set[tuple[str, str | None, str, str]] = set()

        missing_priorities: list[str] = []

        for rule in self.approval_rules:
            rule_key = (
                rule.risk_level,
                rule.category,
                rule.action,
                rule.approver,
            )

            if rule_key in seen_rules:
                raise ValueError(
                    "Duplicate approval rule detected for "
                    f"risk_level={rule.risk_level!r}, "
                    f"category={rule.category!r}, "
                    f"action={rule.action!r}, "
                    f"approver={rule.approver!r}."
                )

            seen_rules.add(rule_key)

            priority_key = resolve_priority_key(
                rule.action,
                self.decision_priority,
            )

            if priority_key not in self.decision_priority:
                missing_priorities.append(f"{rule.action} -> {priority_key}")

        if missing_priorities:
            raise ValueError(
                "Approval actions without decision priorities: "
                + ", ".join(sorted(set(missing_priorities)))
            )

        return self

    def priority_for_action(
        self,
        action: str,
    ) -> int:
        """
        Return the configured priority for an approval action.
        """

        cleaned_action = action.strip()

        if not cleaned_action:
            raise ValueError("Approval action cannot be blank.")

        priority_key = resolve_priority_key(
            cleaned_action,
            self.decision_priority,
        )

        if priority_key not in self.decision_priority:
            raise ValueError(
                "No decision priority is configured for "
                f"action {cleaned_action!r}. "
                f"Expected priority key {priority_key!r}."
            )

        return self.decision_priority[priority_key]

    @classmethod
    def from_yaml_file(
        cls,
        path: str | Path,
    ) -> "ApprovalPolicy":
        """
        Load and validate approval_policy.yaml.

        The existing scenario files place decision_priority under
        department_routing. This method moves it internally into the
        top-level structure expected by the Pydantic model.

        The original YAML file is not modified.
        """

        file_path = Path(path)

        try:
            raw_text = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"Could not read approval policy: {file_path}") from exc

        try:
            raw_data: Any = yaml.safe_load(raw_text)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in approval policy: {file_path}") from exc

        if not isinstance(raw_data, dict):
            raise ValueError("approval_policy.yaml must contain a YAML mapping.")

        normalized_data = dict(raw_data)

        raw_department_routing = normalized_data.get(
            "department_routing",
            {},
        )

        if not isinstance(raw_department_routing, dict):
            raise ValueError("department_routing must contain a YAML mapping.")

        normalized_department_routing = dict(raw_department_routing)

        # Support current scenario files where decision_priority is
        # nested under department_routing because of YAML indentation.
        nested_priority = normalized_department_routing.pop(
            "decision_priority",
            None,
        )

        top_level_priority = normalized_data.get("decision_priority")

        # Reject ambiguous policies containing two different priority
        # definitions.
        if (
            top_level_priority is not None
            and nested_priority is not None
            and top_level_priority != nested_priority
        ):
            raise ValueError(
                "Conflicting top-level and nested " "decision_priority sections."
            )

        if top_level_priority is None:
            if nested_priority is None:
                raise ValueError(
                    "approval_policy.yaml must define " "decision_priority."
                )

            normalized_data["decision_priority"] = nested_priority

        normalized_data["department_routing"] = normalized_department_routing

        return cls.model_validate(normalized_data)
