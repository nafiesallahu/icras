"""Tests for Agent A — the Intake & Context Agent / Gatekeeper.

Every test builds a self-contained contract bundle inside pytest's ``tmp_path``
and points the agent at a temporary runs root, so the suite is hermetic: it
never touches ``data/bundles/`` or the repository ``runs/`` directory.
"""

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from app.agents.intake_agent import IntakeAgent, IntakeError
from app.schemas.context_packet import ContextPacket, DocumentType
from app.utils.hashing import sha256_file

# Minimal valid single-page PDF used as a deterministic contract.pdf stand-in.
MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n"
    b"2 0 obj<< /Type /Page >>endobj\n"
    b"trailer<< /Root 1 0 R >>\n%%EOF\n"
)

PLAYBOOK_YAML = """version: "1.0"
required_clauses:
  - "liability_cap"
  - "termination"
"""

APPROVAL_POLICY_YAML = """version: "1.0"
approval_rules:
  - risk_level: "low"
    action: "auto_approve"
"""

VENDOR_MASTER_CSV = (
    "vendor_id,vendor_name,country,status,risk_level\n"
    "V001,Globex Industries GmbH,Germany,approved,low\n"
    "V003,Acme Services GmbH,Germany,approved,low\n"
    "V004,Umbrella SaaS Inc,United States,approved,medium\n"
)

JURISDICTION_RULES_YAML = """version: "1.0"
jurisdictions:
  Germany:
    risk_level: "low"
    risk_score: 10
  Russia:
    risk_level: "high"
    risk_score: 75
  North Korea:
    risk_level: "critical"
    risk_score: 95
default_jurisdiction:
  risk_level: "medium"
"""

DEFAULT_MANIFEST = {
    "bundle_id": "scenario_03_net_90_payment_terms",
    "contract_id": "contract_003",
    "contract_type": "services_agreement",
    "input_file": "contract.pdf",
    "expected_behavior": "payment_terms_exception",
    "jurisdiction": "Germany",
    "counterparty_name": "Acme Services GmbH",
    "created_by": "test_team",
    "contains_personal_data": False,
}


def make_bundle(
    root: Path,
    *,
    manifest_overrides: dict | None = None,
    omit: set[str] | None = None,
    extra_files: dict[str, str] | None = None,
) -> Path:
    """Create a bundle directory under ``root`` and return its path.

    Args:
        root: Parent directory (typically ``tmp_path``).
        manifest_overrides: Values merged onto :data:`DEFAULT_MANIFEST`. A value
            of ``None`` removes that manifest key entirely.
        omit: Required filenames to leave out of the bundle.
        extra_files: Mapping of extra filename -> text content to add.
    """
    omit = omit or set()
    bundle_dir = root / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    if "contract.pdf" not in omit:
        (bundle_dir / "contract.pdf").write_bytes(MINIMAL_PDF)
    if "playbook.yaml" not in omit:
        (bundle_dir / "playbook.yaml").write_text(PLAYBOOK_YAML, encoding="utf-8")
    if "approval_policy.yaml" not in omit:
        (bundle_dir / "approval_policy.yaml").write_text(
            APPROVAL_POLICY_YAML, encoding="utf-8"
        )
    if "vendor_master.csv" not in omit:
        (bundle_dir / "vendor_master.csv").write_text(
            VENDOR_MASTER_CSV, encoding="utf-8"
        )
    if "jurisdiction_rules.yaml" not in omit:
        (bundle_dir / "jurisdiction_rules.yaml").write_text(
            JURISDICTION_RULES_YAML, encoding="utf-8"
        )

    if "manifest.yaml" not in omit:
        manifest = dict(DEFAULT_MANIFEST)
        for key, value in (manifest_overrides or {}).items():
            if value is None:
                manifest.pop(key, None)
            else:
                manifest[key] = value
        (bundle_dir / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
        )

    for filename, content in (extra_files or {}).items():
        (bundle_dir / filename).write_text(content, encoding="utf-8")

    return bundle_dir


@pytest.fixture
def agent(tmp_path: Path) -> IntakeAgent:
    """An IntakeAgent that writes runs into an isolated temporary directory."""
    return IntakeAgent(runs_root=tmp_path / "runs")


# ---------------------------------------------------------------------------
# Successful intake produces all required artifacts
# ---------------------------------------------------------------------------
def test_intake_creates_run_directory(agent: IntakeAgent, tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    packet = agent.run(bundle)
    assert Path(packet.run_directory).is_dir()


def test_intake_creates_input_snapshot(agent: IntakeAgent, tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    packet = agent.run(bundle)
    snapshot = Path(packet.input_snapshot_directory)
    assert snapshot.is_dir()
    for filename in (
        "contract.pdf",
        "manifest.yaml",
        "playbook.yaml",
        "approval_policy.yaml",
        "vendor_master.csv",
        "jurisdiction_rules.yaml",
    ):
        assert (snapshot / filename).is_file()


def test_intake_writes_context_packet(agent: IntakeAgent, tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    packet = agent.run(bundle)
    packet_path = Path(packet.run_directory) / "context_packet.json"
    assert packet_path.is_file()
    # Reloads and revalidates cleanly.
    reloaded = ContextPacket.from_json_file(packet_path)
    assert reloaded.contract_id == "contract_003"


def test_intake_writes_evidence_index(agent: IntakeAgent, tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    packet = agent.run(bundle)
    index_path = Path(packet.run_directory) / "evidence_index.json"
    assert index_path.is_file()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["evidence_items"] == []
    assert len(index["documents"]) == 6
    primary = index["documents"][0]
    assert primary["document_type"] == "primary_contract"
    assert primary["filename"] == "contract.pdf"
    assert set(primary) == {
        "document_id",
        "filename",
        "document_type",
        "sha256",
        "included",
    }


def test_intake_writes_audit_log(agent: IntakeAgent, tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    packet = agent.run(bundle)
    audit_path = Path(packet.run_directory) / "audit_log.md"
    assert audit_path.is_file()
    text = audit_path.read_text(encoding="utf-8")
    assert f"# Audit Log — {packet.run_id}" in text
    assert "## Step 1: Intake" in text
    assert "Status: intake_completed" in text


def test_intake_writes_metrics(agent: IntakeAgent, tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    packet = agent.run(bundle)
    metrics_path = Path(packet.run_directory) / "metrics.json"
    assert metrics_path.is_file()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["run_id"] == packet.run_id
    assert metrics["contract_id"] == "contract_003"
    assert metrics["agents_completed"] == ["intake"]
    assert metrics["documents_found"] == 6
    assert metrics["documents_included"] == 6
    assert metrics["documents_ignored"] == 0
    assert metrics["status"] == "intake_completed"


def test_intake_produces_all_required_output_files(
    agent: IntakeAgent, tmp_path: Path
) -> None:
    bundle = make_bundle(tmp_path)
    packet = agent.run(bundle)
    run_dir = Path(packet.run_directory)
    for artifact in (
        "context_packet.json",
        "evidence_index.json",
        "audit_log.md",
        "metrics.json",
    ):
        assert (run_dir / artifact).is_file()
    assert (run_dir / "input_snapshot").is_dir()


# ---------------------------------------------------------------------------
# Required-file validation
# ---------------------------------------------------------------------------
def test_required_files_are_validated(agent: IntakeAgent, tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    # A complete bundle does not raise.
    packet = agent.run(bundle)
    assert packet.status == "intake_completed"


def test_missing_contract_pdf_fails(agent: IntakeAgent, tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path, omit={"contract.pdf"})
    with pytest.raises(IntakeError) as exc:
        agent.run(bundle)
    assert "contract.pdf" in str(exc.value)


def test_missing_manifest_fails(agent: IntakeAgent, tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path, omit={"manifest.yaml"})
    with pytest.raises(IntakeError) as exc:
        agent.run(bundle)
    assert "manifest.yaml" in str(exc.value)


def test_missing_playbook_fails(agent: IntakeAgent, tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path, omit={"playbook.yaml"})
    with pytest.raises(IntakeError) as exc:
        agent.run(bundle)
    assert "playbook.yaml" in str(exc.value)


def test_missing_file_writes_failure_metrics(
    agent: IntakeAgent, tmp_path: Path
) -> None:
    bundle = make_bundle(tmp_path, omit={"playbook.yaml"})
    with pytest.raises(IntakeError):
        agent.run(bundle)
    # Exactly one failed run directory was created with failure artifacts.
    run_dirs = list((tmp_path / "runs").iterdir())
    assert len(run_dirs) == 1
    metrics = json.loads((run_dirs[0] / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "intake_failed"
    audit = (run_dirs[0] / "audit_log.md").read_text(encoding="utf-8")
    assert "intake_failed" in audit


def test_missing_bundle_directory_fails(agent: IntakeAgent, tmp_path: Path) -> None:
    with pytest.raises(IntakeError):
        agent.run(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# Classification & filtering
# ---------------------------------------------------------------------------
def test_document_classification_works(agent: IntakeAgent, tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    packet = agent.run(bundle)
    classification = {
        document["filename"]: document["document_type"]
        for document in packet.documents
    }
    assert classification == {
        "contract.pdf": "primary_contract",
        "manifest.yaml": "manifest",
        "playbook.yaml": "playbook",
        "approval_policy.yaml": "approval_policy",
        "vendor_master.csv": "vendor_master",
        "jurisdiction_rules.yaml": "jurisdiction_rules",
    }
    # Document ids are deterministic and contiguous.
    assert [d["document_id"] for d in packet.documents] == [
        "doc_001",
        "doc_002",
        "doc_003",
        "doc_004",
        "doc_005",
        "doc_006",
    ]


def test_ignored_files_are_recorded(agent: IntakeAgent, tmp_path: Path) -> None:
    bundle = make_bundle(
        tmp_path,
        extra_files={"notes.txt": "scratch", "screenshot.png": "x"},
    )
    packet = agent.run(bundle)
    ignored_names = {entry["filename"] for entry in packet.ignored_files}
    assert ignored_names == {"notes.txt", "screenshot.png"}
    for entry in packet.ignored_files:
        assert entry["reason"] == "unsupported_or_irrelevant_file_type"
    # Ignored files are not snapshotted.
    snapshot = Path(packet.input_snapshot_directory)
    assert not (snapshot / "notes.txt").exists()


def test_ignored_files_create_risk_indicators(
    agent: IntakeAgent, tmp_path: Path
) -> None:
    bundle = make_bundle(tmp_path, extra_files={"notes.txt": "scratch"})
    packet = agent.run(bundle)
    assert "ignored_files_present" in packet.initial_risk_indicators
    assert "unsupported_file_type:notes.txt" in packet.initial_risk_indicators


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def test_sha256_hash_is_generated(agent: IntakeAgent, tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    packet = agent.run(bundle)
    expected = sha256_file(bundle / "contract.pdf")
    assert packet.contract_sha256 == expected
    assert len(packet.contract_sha256) == 64
    # Every document carries its own digest.
    for document in packet.documents:
        assert len(document["sha256"]) == 64


# ---------------------------------------------------------------------------
# Manifest metadata flows into the context packet
# ---------------------------------------------------------------------------
def test_manifest_metadata_appears_in_packet(
    agent: IntakeAgent, tmp_path: Path
) -> None:
    bundle = make_bundle(tmp_path)
    packet = agent.run(bundle)
    assert packet.bundle_id == "scenario_03_net_90_payment_terms"
    assert packet.contract_id == "contract_003"
    assert packet.contract_type == "services_agreement"
    assert packet.input_file == "contract.pdf"
    assert packet.document_type == DocumentType.SERVICES_AGREEMENT


def test_jurisdiction_appears_in_packet(agent: IntakeAgent, tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    packet = agent.run(bundle)
    assert packet.jurisdiction == "Germany"


def test_counterparty_appears_in_packet(agent: IntakeAgent, tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    packet = agent.run(bundle)
    assert packet.counterparty_name_from_manifest == "Acme Services GmbH"


def test_missing_critical_manifest_field_fails(
    agent: IntakeAgent, tmp_path: Path
) -> None:
    bundle = make_bundle(tmp_path, manifest_overrides={"contract_id": None})
    with pytest.raises(IntakeError) as exc:
        agent.run(bundle)
    assert "contract_id" in str(exc.value)


# ---------------------------------------------------------------------------
# Intake-level risk indicators
# ---------------------------------------------------------------------------
def test_high_risk_jurisdiction_creates_indicator(
    agent: IntakeAgent, tmp_path: Path
) -> None:
    bundle = make_bundle(tmp_path, manifest_overrides={"jurisdiction": "Russia"})
    packet = agent.run(bundle)
    assert "high_risk_jurisdiction_candidate" in packet.initial_risk_indicators


def test_low_risk_jurisdiction_creates_no_indicator(
    agent: IntakeAgent, tmp_path: Path
) -> None:
    bundle = make_bundle(tmp_path, manifest_overrides={"jurisdiction": "Germany"})
    packet = agent.run(bundle)
    assert "high_risk_jurisdiction_candidate" not in packet.initial_risk_indicators


def test_unknown_counterparty_creates_indicator(
    agent: IntakeAgent, tmp_path: Path
) -> None:
    bundle = make_bundle(
        tmp_path, manifest_overrides={"counterparty_name": "Rogue Systems LLC"}
    )
    packet = agent.run(bundle)
    assert "unknown_counterparty_candidate" in packet.initial_risk_indicators
    assert "new_counterparty_candidate" in packet.initial_risk_indicators


def test_known_counterparty_creates_no_indicator(
    agent: IntakeAgent, tmp_path: Path
) -> None:
    bundle = make_bundle(
        tmp_path, manifest_overrides={"counterparty_name": "Acme Services GmbH"}
    )
    packet = agent.run(bundle)
    assert "unknown_counterparty_candidate" not in packet.initial_risk_indicators
    assert "new_counterparty_candidate" not in packet.initial_risk_indicators


def test_contains_personal_data_creates_indicator(
    agent: IntakeAgent, tmp_path: Path
) -> None:
    bundle = make_bundle(
        tmp_path, manifest_overrides={"contains_personal_data": True}
    )
    packet = agent.run(bundle)
    assert packet.contains_personal_data is True
    assert "contains_personal_data" in packet.initial_risk_indicators


# ---------------------------------------------------------------------------
# Schema strictness & determinism
# ---------------------------------------------------------------------------
def test_context_packet_rejects_extra_fields() -> None:
    bundle_payload = {
        "run_id": "run_x",
        "bundle_id": "b",
        "contract_id": "c",
        "contract_type": "services_agreement",
        "input_file": "contract.pdf",
        "bundle_path": "/tmp/bundle",
        "run_directory": "/tmp/runs/run_x",
        "input_snapshot_directory": "/tmp/runs/run_x/input_snapshot",
        "contract_sha256": "a" * 64,
        "received_timestamp": "2026-06-14T16:00:00+02:00",
        "document_type": "SERVICES_AGREEMENT",
        "status": "intake_completed",
        "unexpected_field": "not allowed",
    }
    with pytest.raises(ValidationError):
        ContextPacket.model_validate(bundle_payload)


def test_context_packet_rejects_bad_sha256() -> None:
    with pytest.raises(ValidationError):
        ContextPacket.model_validate(
            {
                "run_id": "run_x",
                "bundle_id": "b",
                "contract_id": "c",
                "contract_type": "services_agreement",
                "input_file": "contract.pdf",
                "bundle_path": "/tmp/bundle",
                "run_directory": "/tmp/runs/run_x",
                "input_snapshot_directory": "/tmp/runs/run_x/input_snapshot",
                "contract_sha256": "tooshort",
                "received_timestamp": "2026-06-14T16:00:00+02:00",
                "document_type": "SERVICES_AGREEMENT",
                "status": "intake_completed",
            }
        )


def test_deterministic_fields_are_stable(agent: IntakeAgent, tmp_path: Path) -> None:
    bundle = make_bundle(tmp_path)
    first = agent.run(bundle)
    second = agent.run(bundle)

    volatile = {
        "run_id",
        "run_directory",
        "input_snapshot_directory",
        "received_timestamp",
    }
    first_dump = first.model_dump()
    second_dump = second.model_dump()
    for field, value in first_dump.items():
        if field in volatile:
            continue
        assert second_dump[field] == value, f"field {field} should be deterministic"

    # The volatile run_id genuinely differs between runs.
    assert first.run_id != second.run_id
