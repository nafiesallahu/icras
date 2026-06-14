import pytest
import yaml

from app.services.policy_loader import (
    PolicyLoaderError,
    load_approval_policy,
    load_jurisdictions,
    load_playbook,
)


def write_yaml(tmp_path, filename, data):
    file_path = tmp_path / filename
    file_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return file_path


def test_load_playbook_valid_file():
    playbook = load_playbook()

    assert playbook.payment_terms.max_allowed_days == 30
    assert "payment_terms_exceed_policy" in playbook.risk_scores


def test_load_jurisdictions_valid_file():
    jurisdictions = load_jurisdictions()

    assert "Germany" in jurisdictions.jurisdictions
    assert jurisdictions.default_jurisdiction.risk_score == 40


def test_load_approval_policy_valid_file():
    approval_policy = load_approval_policy()

    assert "finance" in approval_policy.department_routing
    assert approval_policy.decision_priority["auto_approve"] == 10


def test_load_playbook_fails_when_required_section_missing(tmp_path):
    invalid_playbook = {
        "version": "1.0",
        "owner": "ICRAS Team",
        "description": "Invalid playbook",
    }

    file_path = write_yaml(tmp_path, "invalid_playbook.yaml", invalid_playbook)

    with pytest.raises(PolicyLoaderError):
        load_playbook(file_path)


def test_load_jurisdictions_fails_when_default_missing(tmp_path):
    invalid_jurisdictions = {
        "version": "1.0",
        "owner": "ICRAS Team",
        "description": "Invalid jurisdictions",
        "jurisdictions": {},
    }

    file_path = write_yaml(tmp_path, "invalid_jurisdictions.yaml", invalid_jurisdictions)

    with pytest.raises(PolicyLoaderError):
        load_jurisdictions(file_path)


def test_load_approval_policy_fails_when_priority_missing(tmp_path):
    invalid_approval_policy = {
        "version": "1.0",
        "owner": "ICRAS Team",
        "description": "Invalid approval policy",
        "approval_rules": [],
        "department_routing": {},
    }

    file_path = write_yaml(tmp_path, "invalid_approval_policy.yaml", invalid_approval_policy)

    with pytest.raises(PolicyLoaderError):
        load_approval_policy(file_path)