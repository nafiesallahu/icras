"""Deterministic builder for ICRAS contract bundles (Agent A inputs).

Each scenario bundle under ``data/bundles/<scenario>/`` is assembled to match the
required Agent A bundle format::

    data/bundles/scenario_x/
    |-- contract.pdf
    |-- manifest.yaml
    |-- playbook.yaml
    |-- approval_policy.yaml
    |-- vendor_master.csv
    |-- jurisdiction_rules.yaml

The four policy files (playbook, approval_policy, vendor_master,
jurisdiction_rules) are copied verbatim from the canonical ``policies/``
directory so every bundle shares the same corporate rules. ``manifest.yaml`` is
generated per scenario.

``contract.pdf`` is a REAL born-digital, text-based PDF (not an empty
placeholder). Each scenario PDF contains readable, scenario-specific contract
clauses aligned with that scenario's expected behavior, so Agent B can extract
clauses directly from the PDF with pdfplumber / PyMuPDF. The PDF is produced by a
small dependency-free writer that embeds a real text content stream using the
standard Helvetica font, which keeps generation deterministic.

Run with::

    python scripts/build_bundles.py
"""

from __future__ import annotations

import shutil
import sys
import textwrap
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

POLICIES_DIR = REPO_ROOT / "policies"
BUNDLES_DIR = REPO_ROOT / "data" / "bundles"

# Policy files copied identically into every bundle.
SHARED_POLICY_FILES = (
    "playbook.yaml",
    "approval_policy.yaml",
    "vendor_master.csv",
    "jurisdiction_rules.yaml",
)

# The contracting buyer is constant across scenarios; only the counterparty,
# jurisdiction and risk-relevant clauses change.
CUSTOMER_NAME = "Genpact Procurement GmbH"


# ---------------------------------------------------------------------------
# Born-digital PDF writer (dependency-free, deterministic)
# ---------------------------------------------------------------------------
# A page is roughly Letter size (612 x 792 pt). Text starts near the top and
# steps down by the leading for every line, wrapping at a fixed column so the
# output stays within the page width with a Helvetica 11pt font.
_PAGE_WIDTH = 612
_PAGE_HEIGHT = 792
_LEFT_MARGIN = 54
_TOP_Y = 750
_FONT_SIZE = 11
_LEADING = 15
_WRAP_COLUMNS = 92
_MAX_LINES_PER_PAGE = 46


def _escape_pdf_text(text: str) -> str:
    """Escape characters that are special inside a PDF literal string."""
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_paragraphs(paragraphs: list[str]) -> list[str]:
    """Wrap paragraphs into display lines, preserving blank-line spacing."""
    lines: list[str] = []
    for paragraph in paragraphs:
        if paragraph == "":
            lines.append("")
            continue
        wrapped = textwrap.wrap(paragraph, width=_WRAP_COLUMNS)
        lines.extend(wrapped or [""])
    return lines or [""]


def _paginate(lines: list[str]) -> list[list[str]]:
    """Split display lines into per-page chunks."""
    return [
        lines[start : start + _MAX_LINES_PER_PAGE]
        for start in range(0, len(lines), _MAX_LINES_PER_PAGE)
    ] or [[""]]


def _content_stream(lines: list[str]) -> bytes:
    """Build a text content stream that renders ``lines`` top-to-bottom."""
    parts = [
        "BT",
        f"/F1 {_FONT_SIZE} Tf",
        f"{_LEFT_MARGIN} {_TOP_Y} Td",
        f"{_LEADING} TL",
    ]
    for index, line in enumerate(lines):
        escaped = _escape_pdf_text(line)
        if index == 0:
            parts.append(f"({escaped}) Tj")
        else:
            parts.append("T*")
            if escaped:
                parts.append(f"({escaped}) Tj")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1")


def build_text_pdf(paragraphs: list[str]) -> bytes:
    """Render contract paragraphs into a real, parseable born-digital PDF.

    Args:
        paragraphs: Logical paragraphs (use ``""`` for a blank spacer line).

    Returns:
        The full PDF file as bytes, with selectable text in a Helvetica font.
    """
    pages_lines = _paginate(_wrap_paragraphs(paragraphs))
    num_pages = len(pages_lines)

    font_obj = 3
    # Object numbering: 1 catalog, 2 pages, 3 font, then (content, page) pairs.
    content_objs: list[int] = []
    page_objs: list[int] = []
    next_obj = 4
    for _ in range(num_pages):
        content_objs.append(next_obj)
        page_objs.append(next_obj + 1)
        next_obj += 2

    objects: dict[int, bytes] = {}
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    kids = " ".join(f"{num} 0 R" for num in page_objs)
    objects[2] = (
        f"<< /Type /Pages /Kids [{kids}] /Count {num_pages} >>".encode("latin-1")
    )
    objects[font_obj] = (
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
        b"/Encoding /WinAnsiEncoding >>"
    )
    for page_index in range(num_pages):
        stream = _content_stream(pages_lines[page_index])
        content_num = content_objs[page_index]
        objects[content_num] = (
            b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)
        )
        page_num = page_objs[page_index]
        objects[page_num] = (
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 {_PAGE_WIDTH} {_PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 {font_obj} 0 R >> >> "
            f"/Contents {content_num} 0 R >>"
        ).encode("latin-1")

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode("latin-1")
        out += objects[num]
        out += b"\nendobj\n"

    xref_position = len(out)
    size = max(objects) + 1
    out += f"xref\n0 {size}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for num in range(1, size):
        out += f"{offsets[num]:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {size} /Root 1 0 R >>\n"
        f"startxref\n{xref_position}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(out)


# ---------------------------------------------------------------------------
# Contract document assembly (shared scaffolding)
# ---------------------------------------------------------------------------
def _assemble_contract(
    *,
    title: str,
    counterparty: str,
    effective_date: str,
    clauses: list[tuple[str, str]],
    signature_line: str,
) -> list[str]:
    """Assemble a full contract as an ordered list of paragraphs."""
    paragraphs: list[str] = [
        title,
        "",
        f"This Agreement is made and entered into as of {effective_date} "
        f'by and between {CUSTOMER_NAME} ("Customer") and {counterparty} '
        '("Counterparty"). The Customer and the Counterparty are each a '
        '"Party" and together the "Parties".',
        "",
    ]
    for index, (heading, body) in enumerate(clauses, start=1):
        paragraphs.append(f"{index}. {heading}")
        paragraphs.append(body)
        paragraphs.append("")

    paragraphs.append(
        "IN WITNESS WHEREOF, the Parties have executed this Agreement as of the "
        "Effective Date first written above."
    )
    paragraphs.append("")
    paragraphs.append(f"Customer: {CUSTOMER_NAME}")
    paragraphs.append(
        "Signature: ______________________   Name: Alex Buyer   "
        "Title: Procurement Director"
    )
    paragraphs.append("")
    paragraphs.append(f"Counterparty: {counterparty}")
    paragraphs.append(signature_line)
    return paragraphs


# Reusable clause bodies (kept ASCII-only for robust Helvetica encoding).
_CONFIDENTIALITY = (
    "Confidentiality. Each Party shall keep the other Party's Confidential "
    "Information in strict confidence, use it solely to perform this Agreement, "
    "and disclose it only to representatives with a need to know who are bound "
    "by equivalent obligations of confidentiality."
)
_TERMINATION = (
    "Termination. Either Party may terminate this Agreement for material breach "
    "upon thirty (30) days written notice if such breach remains uncured."
)
_GDPR = (
    "Data Protection. To the extent any personal data is processed under this "
    "Agreement, the Parties shall comply with Regulation (EU) 2016/679 (GDPR) "
    "and applicable data protection law, applying appropriate technical and "
    "organisational measures."
)
_LIABILITY_CAP = (
    "Limitation of Liability. The aggregate liability of either Party under "
    "this Agreement shall not exceed the total fees paid or payable under this "
    "Agreement in the twelve (12) months preceding the claim."
)
_PAYMENT_NET_30 = (
    "Payment Terms. The Customer shall pay all undisputed invoices within "
    "thirty (30) days of the date of invoice (Net 30)."
)
_GOVERNING_LAW_GERMANY = (
    "Governing Law. This Agreement shall be governed by and construed in "
    "accordance with the laws of Germany, and the courts of Frankfurt shall "
    "have exclusive jurisdiction."
)
_SIGNATURE_STANDARD = (
    "Signature: ______________________   Name: Jordan Vendor   "
    "Title: Authorized Signatory"
)


# ---------------------------------------------------------------------------
# Per-scenario contract builders
# ---------------------------------------------------------------------------
def _contract_01(counterparty: str) -> list[str]:
    """SC01 Clean NDA: confidentiality, governing law, parties, signature."""
    return _assemble_contract(
        title="MUTUAL NON-DISCLOSURE AGREEMENT",
        counterparty=counterparty,
        effective_date="01 January 2025",
        clauses=[
            ("Purpose", "The Parties wish to explore a potential business "
             "relationship and may disclose Confidential Information for that "
             "purpose."),
            ("Confidentiality", _CONFIDENTIALITY),
            ("Term", "This Agreement remains in effect for three (3) years from "
             "the Effective Date; confidentiality obligations survive "
             "termination."),
            ("Termination", _TERMINATION),
            ("Limitation of Liability", _LIABILITY_CAP),
            ("Governing Law", _GOVERNING_LAW_GERMANY),
        ],
        signature_line=_SIGNATURE_STANDARD,
    )


def _contract_02(counterparty: str) -> list[str]:
    """SC02 Missing Liability Cap: normal terms but NO liability cap clause."""
    return _assemble_contract(
        title="SUPPLY AGREEMENT",
        counterparty=counterparty,
        effective_date="15 March 2025",
        clauses=[
            ("Scope of Supply", "The Counterparty shall supply the goods and "
             "components described in the applicable purchase orders."),
            ("Payment Terms", _PAYMENT_NET_30),
            ("Confidentiality", _CONFIDENTIALITY),
            ("Data Protection", _GDPR),
            ("Termination", _TERMINATION),
            ("Governing Law", _GOVERNING_LAW_GERMANY),
        ],
        signature_line=_SIGNATURE_STANDARD,
    )


def _contract_03(counterparty: str) -> list[str]:
    """SC03 Net 90 Payment Terms: payment clause with ninety (90) days / Net 90."""
    return _assemble_contract(
        title="SERVICES AGREEMENT",
        counterparty=counterparty,
        effective_date="01 February 2025",
        clauses=[
            ("Services", "The Counterparty shall provide the professional "
             "services described in the applicable statement of work."),
            ("Payment Terms", "Payment Terms. The Customer shall pay all "
             "undisputed invoices within ninety (90) days of the date of "
             "invoice receipt (Net 90)."),
            ("Limitation of Liability", _LIABILITY_CAP),
            ("Confidentiality", _CONFIDENTIALITY),
            ("Data Protection", _GDPR),
            ("Termination", _TERMINATION),
            ("Governing Law", _GOVERNING_LAW_GERMANY),
        ],
        signature_line=_SIGNATURE_STANDARD,
    )


def _contract_04(counterparty: str) -> list[str]:
    """SC04 High-Risk Jurisdiction: governing law in a high-risk jurisdiction."""
    return _assemble_contract(
        title="SERVICES AGREEMENT",
        counterparty=counterparty,
        effective_date="01 April 2025",
        clauses=[
            ("Services", "The Counterparty shall provide operational support "
             "services for the Customer's eastern region operations."),
            ("Payment Terms", _PAYMENT_NET_30),
            ("Limitation of Liability", _LIABILITY_CAP),
            ("Confidentiality", _CONFIDENTIALITY),
            ("Data Protection", _GDPR),
            ("Termination", _TERMINATION),
            ("Governing Law", "Governing Law. This Agreement shall be governed "
             "by and construed in accordance with the laws of the Russian "
             "Federation (Russia), and the courts seated in Russia shall have "
             "exclusive jurisdiction."),
        ],
        signature_line=_SIGNATURE_STANDARD,
    )


def _contract_05(counterparty: str) -> list[str]:
    """SC05 Auto-Renewal Without Opt-Out: auto-renewal but no opt-out window."""
    return _assemble_contract(
        title="SUBSCRIPTION AGREEMENT",
        counterparty=counterparty,
        effective_date="01 February 2025",
        clauses=[
            ("Subscription", "The Counterparty grants the Customer a "
             "subscription to its software-as-a-service platform."),
            ("Payment Terms", _PAYMENT_NET_30),
            ("Auto-Renewal", "Auto-Renewal. This Agreement shall automatically "
             "renew for successive twelve (12) month terms at the end of each "
             "then-current term unless otherwise agreed in writing."),
            ("Limitation of Liability", _LIABILITY_CAP),
            ("Confidentiality", _CONFIDENTIALITY),
            ("Governing Law", _GOVERNING_LAW_GERMANY),
        ],
        signature_line=_SIGNATURE_STANDARD,
    )


def _contract_06(counterparty: str) -> list[str]:
    """SC06 Conflicting Governing Law: two conflicting governing law clauses."""
    return _assemble_contract(
        title="SERVICES AGREEMENT",
        counterparty=counterparty,
        effective_date="01 May 2025",
        clauses=[
            ("Services", "The Counterparty shall provide the services described "
             "in the applicable statement of work."),
            ("Payment Terms", _PAYMENT_NET_30),
            ("Limitation of Liability", _LIABILITY_CAP),
            ("Governing Law", "Governing Law. This Agreement shall be governed "
             "by and construed in accordance with the laws of Germany, and the "
             "courts of Frankfurt, Germany shall have exclusive jurisdiction "
             "over any dispute arising under this Agreement."),
            ("Governing Law (Alternative)", "Notwithstanding the foregoing, "
             "this Agreement shall be governed by and construed in accordance "
             "with the laws of the State of New York, United States, and the "
             "courts of New York shall have exclusive jurisdiction."),
            ("Termination", _TERMINATION),
        ],
        signature_line=_SIGNATURE_STANDARD,
    )


def _contract_07(counterparty: str) -> list[str]:
    """SC07 Low Confidence Signature: ambiguous/illegible signature block."""
    return _assemble_contract(
        title="MUTUAL NON-DISCLOSURE AGREEMENT",
        counterparty=counterparty,
        effective_date="10 March 2025",
        clauses=[
            ("Confidentiality", _CONFIDENTIALITY),
            ("Term", "This Agreement remains in effect for five (5) years from "
             "the Effective Date."),
            ("Termination", _TERMINATION),
            ("Governing Law", _GOVERNING_LAW_GERMANY),
        ],
        # Deliberately degraded signature block to exercise low-confidence
        # signature handling downstream.
        signature_line=(
            "Signature: ~~illegible handwritten mark~~   Name: [unclear]   "
            "Title: [unclear]   (scanned signature, low legibility)"
        ),
    )


def _contract_08(counterparty: str) -> list[str]:
    """SC08 Missing GDPR Clause: personal data processed but NO GDPR clause."""
    return _assemble_contract(
        title="SERVICES AGREEMENT",
        counterparty=counterparty,
        effective_date="01 June 2025",
        clauses=[
            ("Services", "The Counterparty shall provide managed services to the "
             "Customer."),
            ("Personal Data Scope", "In performing the services, the "
             "Counterparty processes personal data of the Customer's employees "
             "and end users, including names, contact details and usage "
             "records, in order to deliver the services."),
            ("Payment Terms", _PAYMENT_NET_30),
            ("Limitation of Liability", _LIABILITY_CAP),
            ("Confidentiality", _CONFIDENTIALITY),
            ("Termination", _TERMINATION),
            ("Governing Law", _GOVERNING_LAW_GERMANY),
        ],
        # No Data Protection / GDPR clause is included on purpose.
        signature_line=_SIGNATURE_STANDARD,
    )


def _contract_09(counterparty: str) -> list[str]:
    """SC09 Clean Services Agreement: standard clean terms under threshold."""
    return _assemble_contract(
        title="STANDARD SERVICES AGREEMENT",
        counterparty=counterparty,
        effective_date="01 July 2025",
        clauses=[
            ("Services", "The Counterparty shall provide the standard "
             "professional services described in the applicable statement of "
             "work."),
            ("Payment Terms", _PAYMENT_NET_30),
            ("Limitation of Liability", _LIABILITY_CAP),
            ("Confidentiality", _CONFIDENTIALITY),
            ("Data Protection", _GDPR),
            ("Termination", _TERMINATION),
            ("Governing Law", _GOVERNING_LAW_GERMANY),
        ],
        signature_line=_SIGNATURE_STANDARD,
    )


def _contract_10(counterparty: str) -> list[str]:
    """SC10 Multiple High Risks: Net 120, Iran, no cap, personal data, no GDPR."""
    return _assemble_contract(
        title="MASTER AGREEMENT",
        counterparty=counterparty,
        effective_date="01 August 2025",
        clauses=[
            ("Services", "The Counterparty shall provide offshore master "
             "services to the Customer."),
            ("Personal Data Scope", "Under this Agreement the Counterparty "
             "processes personal data of data subjects located in the European "
             "Union, including identifiers and financial records."),
            ("Payment Terms", "Payment Terms. The Customer shall pay all "
             "undisputed invoices within one hundred twenty (120) days of the "
             "date of invoice (Net 120)."),
            ("Auto-Renewal", "Auto-Renewal. This Agreement shall automatically "
             "renew for successive annual terms."),
            ("Governing Law", "Governing Law. This Agreement shall be governed "
             "by and construed in accordance with the laws of the Islamic "
             "Republic of Iran (Iran), and the courts seated in Iran shall have "
             "exclusive jurisdiction."),
        ],
        # No Limitation of Liability clause and no Data Protection / GDPR clause.
        signature_line=_SIGNATURE_STANDARD,
    )


# Maps bundle id -> contract builder. Each builder receives the counterparty.
CONTRACT_BUILDERS = {
    "scenario_01_clean_nda": _contract_01,
    "scenario_02_missing_liability_cap": _contract_02,
    "scenario_03_net_90_payment_terms": _contract_03,
    "scenario_04_high_risk_jurisdiction": _contract_04,
    "scenario_05_auto_renewal_no_opt_out": _contract_05,
    "scenario_06_conflicting_governing_law": _contract_06,
    "scenario_07_low_confidence_signature": _contract_07,
    "scenario_08_missing_gdpr_clause": _contract_08,
    "scenario_09_clean_services_agreement": _contract_09,
    "scenario_10_multiple_high_risks": _contract_10,
}


# Per-scenario manifest metadata. Bundle id equals the directory name.
SCENARIO_MANIFESTS: dict[str, dict] = {
    "scenario_01_clean_nda": {
        "contract_id": "contract_001",
        "contract_type": "nda",
        "expected_behavior": "clean_auto_approve",
        "jurisdiction": "Germany",
        "counterparty_name": "Globex Industries GmbH",
        "contains_personal_data": False,
    },
    "scenario_02_missing_liability_cap": {
        "contract_id": "contract_002",
        "contract_type": "services_agreement",
        "expected_behavior": "missing_liability_cap",
        "jurisdiction": "Germany",
        "counterparty_name": "Initech Components Ltd",
        "contains_personal_data": False,
    },
    "scenario_03_net_90_payment_terms": {
        "contract_id": "contract_003",
        "contract_type": "services_agreement",
        "expected_behavior": "payment_terms_exception",
        "jurisdiction": "Germany",
        "counterparty_name": "Acme Services GmbH",
        "contains_personal_data": False,
    },
    "scenario_04_high_risk_jurisdiction": {
        "contract_id": "contract_004",
        "contract_type": "services_agreement",
        "expected_behavior": "high_risk_jurisdiction",
        "jurisdiction": "Russia",
        "counterparty_name": "Globex Industries GmbH",
        "contains_personal_data": False,
    },
    "scenario_05_auto_renewal_no_opt_out": {
        "contract_id": "contract_005",
        "contract_type": "subscription_agreement",
        "expected_behavior": "auto_renewal_without_opt_out",
        "jurisdiction": "Germany",
        "counterparty_name": "Umbrella SaaS Inc",
        "contains_personal_data": False,
    },
    "scenario_06_conflicting_governing_law": {
        "contract_id": "contract_006",
        "contract_type": "services_agreement",
        "expected_behavior": "jurisdiction_conflict",
        "jurisdiction": "Germany",
        "counterparty_name": "Wayne Enterprises GmbH",
        "contains_personal_data": False,
    },
    "scenario_07_low_confidence_signature": {
        "contract_id": "contract_007",
        "contract_type": "nda",
        "expected_behavior": "low_confidence_manual_review",
        "jurisdiction": "Germany",
        "counterparty_name": "Stark Innovations GmbH",
        "contains_personal_data": False,
    },
    "scenario_08_missing_gdpr_clause": {
        "contract_id": "contract_008",
        "contract_type": "services_agreement",
        "expected_behavior": "missing_gdpr_clause",
        "jurisdiction": "Germany",
        "counterparty_name": "Hooli Services GmbH",
        "contains_personal_data": True,
    },
    "scenario_09_clean_services_agreement": {
        "contract_id": "contract_009",
        "contract_type": "services_agreement",
        "expected_behavior": "clean_auto_approve",
        "jurisdiction": "Germany",
        "counterparty_name": "Hooli Services GmbH",
        "contains_personal_data": False,
    },
    "scenario_10_multiple_high_risks": {
        "contract_id": "contract_010",
        "contract_type": "services_agreement",
        "expected_behavior": "multiple_high_risks",
        "jurisdiction": "Iran",
        "counterparty_name": "Oscorp Offshore LLC",
        "contains_personal_data": True,
    },
}


def build_manifest(bundle_id: str, meta: dict) -> dict:
    """Assemble an ordered manifest mapping for a scenario."""
    return {
        "bundle_id": bundle_id,
        "contract_id": meta["contract_id"],
        "contract_type": meta["contract_type"],
        "input_file": "contract.pdf",
        "expected_behavior": meta["expected_behavior"],
        "jurisdiction": meta["jurisdiction"],
        "counterparty_name": meta["counterparty_name"],
        "created_by": "test_team",
        "contains_personal_data": meta["contains_personal_data"],
    }


def build_contract_pdf(bundle_id: str, meta: dict) -> bytes:
    """Build the born-digital contract PDF bytes for a scenario."""
    builder = CONTRACT_BUILDERS[bundle_id]
    paragraphs = builder(meta["counterparty_name"])
    return build_text_pdf(paragraphs)


def write_bundle(bundle_id: str, meta: dict) -> Path:
    """Materialize a single bundle directory and return its path."""
    bundle_dir = BUNDLES_DIR / bundle_id
    bundle_dir.mkdir(parents=True, exist_ok=True)

    (bundle_dir / "contract.pdf").write_bytes(build_contract_pdf(bundle_id, meta))

    manifest = build_manifest(bundle_id, meta)
    (bundle_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    for policy_file in SHARED_POLICY_FILES:
        shutil.copy2(POLICIES_DIR / policy_file, bundle_dir / policy_file)

    return bundle_dir


def main() -> None:
    """Build every scenario bundle and report what was written."""
    for policy_file in SHARED_POLICY_FILES:
        if not (POLICIES_DIR / policy_file).is_file():
            raise FileNotFoundError(
                f"Missing canonical policy file: policies/{policy_file}"
            )

    written: list[str] = []
    for bundle_id, meta in SCENARIO_MANIFESTS.items():
        bundle_dir = write_bundle(bundle_id, meta)
        written.append(str(bundle_dir.relative_to(REPO_ROOT)))
        print(f"[OK] built bundle {bundle_dir.relative_to(REPO_ROOT)}")

    print("\n" + "=" * 70)
    print(f"ICRAS bundle build complete. {len(written)} bundles ready.")
    print("=" * 70)


if __name__ == "__main__":
    main()
