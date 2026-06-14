"""Manual smoke test for Agent A (Intake & Context Agent / Gatekeeper).

This is NOT a pytest module (it is intentionally named ``smoke_test_*`` so the
collector ignores it). Run it directly to exercise the full intake flow against
a real on-disk bundle::

    python tests/smoke_test_agent_a.py

It builds a temporary, fully-formed contract bundle, runs the agent, and asserts
that every required artifact was produced before cleaning up after itself.
"""

import shutil
import sys
import tempfile
from pathlib import Path

# Allow running this script directly (``python tests/smoke_test_agent_a.py``)
# by ensuring the repository root is importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from app.agents.intake_agent import IntakeAgent  # noqa: E402
from app.schemas.context_packet import ContextPacket  # noqa: E402

MINIMAL_PDF = b"%PDF-1.4\n1 0 obj<< /Type /Catalog >>endobj\ntrailer<< /Root 1 0 R >>\n%%EOF\n"

MANIFEST = {
    "bundle_id": "scenario_smoke_test",
    "contract_id": "contract_smoke",
    "contract_type": "services_agreement",
    "input_file": "contract.pdf",
    "expected_behavior": "smoke_test",
    "jurisdiction": "Germany",
    "counterparty_name": "Acme Services GmbH",
    "created_by": "test_team",
    "contains_personal_data": False,
}


def setup_mock_bundle(root: Path) -> Path:
    """Create a complete mock contract bundle under ``root``."""
    bundle_dir = root / "mock_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    (bundle_dir / "contract.pdf").write_bytes(MINIMAL_PDF)
    (bundle_dir / "manifest.yaml").write_text(
        yaml.safe_dump(MANIFEST, sort_keys=False), encoding="utf-8"
    )
    (bundle_dir / "playbook.yaml").write_text(
        'version: "1.0"\nrequired_clauses:\n  - liability_cap\n', encoding="utf-8"
    )
    (bundle_dir / "approval_policy.yaml").write_text(
        'version: "1.0"\napproval_rules: []\n', encoding="utf-8"
    )
    (bundle_dir / "vendor_master.csv").write_text(
        "vendor_id,vendor_name,country,status,risk_level\n"
        "V003,Acme Services GmbH,Germany,approved,low\n",
        encoding="utf-8",
    )
    (bundle_dir / "jurisdiction_rules.yaml").write_text(
        'version: "1.0"\njurisdictions:\n  Germany:\n    risk_level: low\n',
        encoding="utf-8",
    )
    return bundle_dir


def run_smoke_test() -> None:
    print("====== AGENT A (INTAKE) SMOKE TEST ======")
    work_root = Path(tempfile.mkdtemp(prefix="icras_smoke_"))
    runs_root = work_root / "runs"

    try:
        bundle_path = setup_mock_bundle(work_root)
        agent = IntakeAgent(runs_root=runs_root)

        print(f"-> Scanning bundle: {bundle_path}")
        packet: ContextPacket = agent.run(bundle_path)

        print("\n--- VERIFICATION RESULTS ---")
        assert packet is not None, "Agent returned no ContextPacket!"
        print("PASSED: ContextPacket constructed.")
        print(f"   - Run ID: {packet.run_id}")
        print(f"   - Contract ID: {packet.contract_id}")
        print(f"   - Document Type: {packet.document_type}")
        print(f"   - Contract SHA256: {packet.contract_sha256}")
        print(f"   - Risk indicators: {packet.initial_risk_indicators}")

        run_dir = Path(packet.run_directory)
        assert run_dir.is_dir(), "Run directory was not created on disk!"
        print(f"PASSED: Run directory created at {run_dir}")

        for artifact in (
            "context_packet.json",
            "evidence_index.json",
            "audit_log.md",
            "metrics.json",
        ):
            assert (run_dir / artifact).is_file(), f"Missing artifact: {artifact}"
        print("PASSED: context_packet.json, evidence_index.json, audit_log.md, "
              "metrics.json all written.")

        snapshot = run_dir / "input_snapshot"
        assert snapshot.is_dir(), "input_snapshot/ was not created!"
        assert (snapshot / "contract.pdf").is_file()
        print("PASSED: input_snapshot/ populated with bundle files.")

        try:
            packet.contract_id = "illegal_mutation"
            raise RuntimeError("Model is not frozen!")
        except (ValidationError, TypeError):
            print("PASSED: ContextPacket is frozen=True (immutable).")

        print("\nFINAL RESULT: Agent A works as expected.")

    except Exception as exc:
        print(f"\nSMOKE TEST FAILED: {exc}")
        raise

    finally:
        print("\n-> Cleaning up test environment...")
        shutil.rmtree(work_root, ignore_errors=True)
        print("Environment cleaned.")


if __name__ == "__main__":
    run_smoke_test()
