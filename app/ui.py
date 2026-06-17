"""ICRAS observability-style Streamlit dashboard (demo layer).

Dark, data-dense UI driven by a mock-data dictionary per scenario. Swap
``MOCK_SCENARIOS`` (or map real pipeline artifacts into the same shape) when
wiring production output.

Run::

    streamlit run app/ui.py
"""

from __future__ import annotations

import html
import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Theme tokens
# ---------------------------------------------------------------------------
BG = "#1a1a1f"
CARD = "#26262c"
BORDER = "#3a3a42"
TEXT = "#ffffff"
MUTED = "#9a9aa3"
STREAMLIT_DEFAULT_TEXT = "#31333f"  # Streamlit baseline — always override to white
PRIMARY = "#3b6fe0"
MONO = "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace"
SANS = "'Inter', 'Segoe UI', system-ui, sans-serif"

RISK_COLORS = {
    "low": "#22c55e",
    "medium": "#3b6fe0",
    "info": "#3b6fe0",
    "high": "#f59e0b",
    "critical": "#ef4444",
}

SEVERITY_ORDER = ["critical", "high", "medium", "low"]

AGENT_ROSTER = {
    "A": "Intake",
    "B": "Clause Extraction",
    "C": "Risk Scoring",
    "D": "Evidence Retrieval",
    "E": "Synthesis",
    "F": "Validation",
    "G": "Posting Prep",
    "H": "Approval Packet",
}

STATUS_STYLE = {
    "completed": ("Completed", RISK_COLORS["low"], "✓"),
    "running": ("Running", PRIMARY, "◉"),
    "skipped": ("Skipped", MUTED, "—"),
    "not_implemented": ("Not implemented", MUTED, "—"),
    "failed": ("Failed", RISK_COLORS["critical"], "✕"),
}


# ---------------------------------------------------------------------------
# Mock data — one entry per scenario selectbox option
# ---------------------------------------------------------------------------
MOCK_SCENARIOS: dict[str, dict] = {
    "Acme Corp — Master Services Agreement": {
        "last_run": "2026-06-17 09:14:22",
        "run_summary": {
            "decision": "Approve with Conditions",
            "decision_dot": RISK_COLORS["high"],
            "overall_risk": "high",
            "total_findings": 12,
            "manual_review_count": 4,
            "confidence_pct": 87,
            "evidence_coverage_pct": 92,
        },
        "agents": [
            {"id": "A", "status": "completed", "artifact": "intake.json", "duration_s": 1.2},
            {"id": "B", "status": "completed", "artifact": "clauses.json", "duration_s": 3.8},
            {"id": "C", "status": "completed", "artifact": "risk_scores.json", "duration_s": 0.9},
            {"id": "D", "status": "completed", "artifact": "evidence.json", "duration_s": 2.1},
            {"id": "E", "status": "completed", "artifact": "synthesis.md", "duration_s": 1.5},
            {"id": "F", "status": "completed", "artifact": "validation_result.json", "duration_s": 0.7},
            {"id": "G", "status": "skipped", "artifact": "posting_payload.json", "duration_s": None},
            {"id": "H", "status": "completed", "artifact": "approval_packet.json", "duration_s": 0.4},
        ],
        "findings": [
            {"id": "FND-001", "clause": "Payment Terms", "category": "Commercial", "severity": "high", "risk_level": "high", "confidence": 0.91, "manual_review": True, "evidence_refs": "EV-003, EV-004"},
            {"id": "FND-002", "clause": "Liability Cap", "category": "Liability", "severity": "high", "risk_level": "high", "confidence": 0.88, "manual_review": True, "evidence_refs": "EV-007"},
            {"id": "FND-003", "clause": "Termination for Convenience", "category": "Termination", "severity": "medium", "risk_level": "medium", "confidence": 0.85, "manual_review": False, "evidence_refs": "EV-009"},
            {"id": "FND-004", "clause": "Governing Law", "category": "Jurisdiction", "severity": "low", "risk_level": "low", "confidence": 0.94, "manual_review": False, "evidence_refs": "EV-011"},
            {"id": "FND-005", "clause": "Auto-Renewal", "category": "Renewal", "severity": "medium", "risk_level": "medium", "confidence": 0.82, "manual_review": False, "evidence_refs": "EV-012"},
            {"id": "FND-006", "clause": "Indemnification", "category": "Liability", "severity": "high", "risk_level": "high", "confidence": 0.79, "manual_review": True, "evidence_refs": "EV-014, EV-015"},
            {"id": "FND-007", "clause": "Data Processing", "category": "Privacy", "severity": "medium", "risk_level": "medium", "confidence": 0.86, "manual_review": False, "evidence_refs": "EV-016"},
            {"id": "FND-008", "clause": "SLA Credits", "category": "Commercial", "severity": "low", "risk_level": "low", "confidence": 0.90, "manual_review": False, "evidence_refs": "EV-018"},
            {"id": "FND-009", "clause": "Insurance Requirements", "category": "Compliance", "severity": "medium", "risk_level": "medium", "confidence": 0.84, "manual_review": False, "evidence_refs": "EV-019"},
            {"id": "FND-010", "clause": "Assignment Restrictions", "category": "General", "severity": "low", "risk_level": "low", "confidence": 0.92, "manual_review": False, "evidence_refs": "EV-020"},
            {"id": "FND-011", "clause": "Audit Rights", "category": "Compliance", "severity": "medium", "risk_level": "medium", "confidence": 0.81, "manual_review": True, "evidence_refs": "EV-021"},
            {"id": "FND-012", "clause": "Force Majeure", "category": "General", "severity": "low", "risk_level": "low", "confidence": 0.93, "manual_review": False, "evidence_refs": "EV-022"},
        ],
        "clause_confidence": [
            {"clause": "Payment Terms", "confidence": 0.91, "risk_level": "high"},
            {"clause": "Liability Cap", "confidence": 0.88, "risk_level": "high"},
            {"clause": "Termination", "confidence": 0.85, "risk_level": "medium"},
            {"clause": "Governing Law", "confidence": 0.94, "risk_level": "low"},
            {"clause": "Auto-Renewal", "confidence": 0.82, "risk_level": "medium"},
            {"clause": "Indemnification", "confidence": 0.79, "risk_level": "high"},
            {"clause": "Data Processing", "confidence": 0.86, "risk_level": "medium"},
            {"clause": "SLA Credits", "confidence": 0.90, "risk_level": "low"},
        ],
        "artifacts": {
            "intake.json": {"type": "json", "exists": True, "content": {"bundle_id": "acme_msa_2026", "contract_type": "msa", "status": "intake_completed"}},
            "clauses.json": {"type": "json", "exists": True, "content": {"contract_id": "ACME-MSA-001", "clauses_extracted": 24}},
            "risk_scores.json": {"type": "json", "exists": True, "content": {"overall_risk": "high", "score": 72}},
            "evidence.json": {"type": "json", "exists": True, "content": {"evidence_items": 22, "documents": 6}},
            "synthesis.md": {"type": "markdown", "exists": True, "content": "## Synthesis\n\nContract requires conditional approval due to liability and payment term exceptions."},
            "approval_packet.json": {"type": "json", "exists": True, "content": {"final_decision": "approve_with_conditions", "approvers": ["legal@acme.com"]}},
            "metrics.json": {"type": "json", "exists": True, "content": {"findings_count": 12, "exceptions_count": 2}},
            "findings_export.csv": {"type": "csv", "exists": True, "content": None},
            "validation_result.json": {"type": "json", "exists": True, "content": {"validation_status": "completed_with_findings"}},
            "posting_payload.json": {"type": "json", "exists": False, "content": None},
        },
    },
    "Globex — Mutual NDA": {
        "last_run": "2026-06-17 08:02:11",
        "run_summary": {
            "decision": "Approve",
            "decision_dot": RISK_COLORS["low"],
            "overall_risk": "low",
            "total_findings": 3,
            "manual_review_count": 0,
            "confidence_pct": 94,
            "evidence_coverage_pct": 98,
        },
        "agents": [
            {"id": "A", "status": "completed", "artifact": "intake.json", "duration_s": 0.9},
            {"id": "B", "status": "completed", "artifact": "clauses.json", "duration_s": 2.4},
            {"id": "C", "status": "completed", "artifact": "risk_scores.json", "duration_s": 0.6},
            {"id": "D", "status": "completed", "artifact": "evidence.json", "duration_s": 1.8},
            {"id": "E", "status": "completed", "artifact": "synthesis.md", "duration_s": 1.1},
            {"id": "F", "status": "completed", "artifact": "validation_result.json", "duration_s": 0.5},
            {"id": "G", "status": "not_implemented", "artifact": "posting_payload.json", "duration_s": None},
            {"id": "H", "status": "completed", "artifact": "approval_packet.json", "duration_s": 0.3},
        ],
        "findings": [
            {"id": "FND-101", "clause": "Confidentiality Term", "category": "Confidentiality", "severity": "low", "risk_level": "low", "confidence": 0.96, "manual_review": False, "evidence_refs": "EV-001"},
            {"id": "FND-102", "clause": "Governing Law", "category": "Jurisdiction", "severity": "low", "risk_level": "low", "confidence": 0.95, "manual_review": False, "evidence_refs": "EV-002"},
            {"id": "FND-103", "clause": "Return of Materials", "category": "General", "severity": "low", "risk_level": "low", "confidence": 0.92, "manual_review": False, "evidence_refs": "EV-003"},
        ],
        "clause_confidence": [
            {"clause": "Confidentiality", "confidence": 0.96, "risk_level": "low"},
            {"clause": "Governing Law", "confidence": 0.95, "risk_level": "low"},
            {"clause": "Return of Materials", "confidence": 0.92, "risk_level": "low"},
            {"clause": "Term", "confidence": 0.94, "risk_level": "low"},
            {"clause": "Exceptions", "confidence": 0.93, "risk_level": "low"},
        ],
        "artifacts": {
            "intake.json": {"type": "json", "exists": True, "content": {"bundle_id": "globex_nda", "contract_type": "nda", "status": "intake_completed"}},
            "clauses.json": {"type": "json", "exists": True, "content": {"contract_id": "GLBX-NDA-004", "clauses_extracted": 8}},
            "risk_scores.json": {"type": "json", "exists": True, "content": {"overall_risk": "low", "score": 18}},
            "evidence.json": {"type": "json", "exists": True, "content": {"evidence_items": 8, "documents": 6}},
            "synthesis.md": {"type": "markdown", "exists": True, "content": "## Synthesis\n\nClean NDA — auto-approve recommended. No material exceptions."},
            "approval_packet.json": {"type": "json", "exists": True, "content": {"final_decision": "approve", "approvers": []}},
            "metrics.json": {"type": "json", "exists": True, "content": {"findings_count": 3, "exceptions_count": 0}},
            "findings_export.csv": {"type": "csv", "exists": True, "content": None},
            "validation_result.json": {"type": "json", "exists": True, "content": {"validation_status": "completed"}},
            "posting_payload.json": {"type": "json", "exists": False, "content": None},
        },
    },
    "Initech — SaaS Subscription": {
        "last_run": "2026-06-16 16:47:55",
        "run_summary": {
            "decision": "Escalate to Legal",
            "decision_dot": RISK_COLORS["critical"],
            "overall_risk": "critical",
            "total_findings": 18,
            "manual_review_count": 7,
            "confidence_pct": 72,
            "evidence_coverage_pct": 78,
        },
        "agents": [
            {"id": "A", "status": "completed", "artifact": "intake.json", "duration_s": 1.0},
            {"id": "B", "status": "completed", "artifact": "clauses.json", "duration_s": 4.2},
            {"id": "C", "status": "completed", "artifact": "risk_scores.json", "duration_s": 1.1},
            {"id": "D", "status": "failed", "artifact": "evidence.json", "duration_s": 0.3},
            {"id": "E", "status": "skipped", "artifact": "synthesis.md", "duration_s": None},
            {"id": "F", "status": "completed", "artifact": "validation_result.json", "duration_s": 0.8},
            {"id": "G", "status": "not_implemented", "artifact": "posting_payload.json", "duration_s": None},
            {"id": "H", "status": "skipped", "artifact": "approval_packet.json", "duration_s": None},
        ],
        "findings": [
            {"id": "FND-201", "clause": "Net-90 Payment Terms", "category": "Commercial", "severity": "critical", "risk_level": "critical", "confidence": 0.68, "manual_review": True, "evidence_refs": "EV-002"},
            {"id": "FND-202", "clause": "Unlimited Liability", "category": "Liability", "severity": "critical", "risk_level": "critical", "confidence": 0.71, "manual_review": True, "evidence_refs": "EV-005"},
            {"id": "FND-203", "clause": "Missing GDPR Clause", "category": "Privacy", "severity": "high", "risk_level": "high", "confidence": 0.74, "manual_review": True, "evidence_refs": "missing:gdpr"},
            {"id": "FND-204", "clause": "Auto-Renewal (no opt-out)", "category": "Renewal", "severity": "high", "risk_level": "high", "confidence": 0.76, "manual_review": True, "evidence_refs": "EV-008"},
            {"id": "FND-205", "clause": "Conflicting Governing Law", "category": "Jurisdiction", "severity": "high", "risk_level": "high", "confidence": 0.70, "manual_review": True, "evidence_refs": "EV-010, EV-011"},
            {"id": "FND-206", "clause": "Low-Confidence Signature", "category": "Execution", "severity": "medium", "risk_level": "medium", "confidence": 0.55, "manual_review": True, "evidence_refs": "EV-012"},
            {"id": "FND-207", "clause": "Data Residency", "category": "Privacy", "severity": "high", "risk_level": "high", "confidence": 0.73, "manual_review": False, "evidence_refs": "EV-013"},
            {"id": "FND-208", "clause": "Service Level", "category": "Commercial", "severity": "medium", "risk_level": "medium", "confidence": 0.80, "manual_review": False, "evidence_refs": "EV-014"},
            {"id": "FND-209", "clause": "IP Ownership", "category": "IP", "severity": "high", "risk_level": "high", "confidence": 0.77, "manual_review": True, "evidence_refs": "EV-015"},
            {"id": "FND-210", "clause": "Subprocessor List", "category": "Privacy", "severity": "medium", "risk_level": "medium", "confidence": 0.82, "manual_review": False, "evidence_refs": "EV-016"},
            {"id": "FND-211", "clause": "Price Escalation", "category": "Commercial", "severity": "medium", "risk_level": "medium", "confidence": 0.79, "manual_review": False, "evidence_refs": "EV-017"},
            {"id": "FND-212", "clause": "Audit Rights", "category": "Compliance", "severity": "low", "risk_level": "low", "confidence": 0.88, "manual_review": False, "evidence_refs": "EV-018"},
            {"id": "FND-213", "clause": "Warranty Disclaimer", "category": "Liability", "severity": "medium", "risk_level": "medium", "confidence": 0.81, "manual_review": False, "evidence_refs": "EV-019"},
            {"id": "FND-214", "clause": "Export Controls", "category": "Compliance", "severity": "low", "risk_level": "low", "confidence": 0.86, "manual_review": False, "evidence_refs": "EV-020"},
            {"id": "FND-215", "clause": "Change of Control", "category": "General", "severity": "medium", "risk_level": "medium", "confidence": 0.75, "manual_review": False, "evidence_refs": "EV-021"},
            {"id": "FND-216", "clause": "Support Hours", "category": "Commercial", "severity": "low", "risk_level": "low", "confidence": 0.90, "manual_review": False, "evidence_refs": "EV-022"},
            {"id": "FND-217", "clause": "Beta Features", "category": "General", "severity": "low", "risk_level": "low", "confidence": 0.84, "manual_review": False, "evidence_refs": "EV-023"},
            {"id": "FND-218", "clause": "Termination Notice", "category": "Termination", "severity": "medium", "risk_level": "medium", "confidence": 0.78, "manual_review": False, "evidence_refs": "EV-024"},
        ],
        "clause_confidence": [
            {"clause": "Payment Terms", "confidence": 0.68, "risk_level": "critical"},
            {"clause": "Liability", "confidence": 0.71, "risk_level": "critical"},
            {"clause": "GDPR", "confidence": 0.74, "risk_level": "high"},
            {"clause": "Auto-Renewal", "confidence": 0.76, "risk_level": "high"},
            {"clause": "Governing Law", "confidence": 0.70, "risk_level": "high"},
            {"clause": "Signature", "confidence": 0.55, "risk_level": "medium"},
            {"clause": "SLA", "confidence": 0.80, "risk_level": "medium"},
            {"clause": "IP Ownership", "confidence": 0.77, "risk_level": "high"},
        ],
        "artifacts": {
            "intake.json": {"type": "json", "exists": True, "content": {"bundle_id": "initech_saas", "contract_type": "saas", "status": "intake_completed"}},
            "clauses.json": {"type": "json", "exists": True, "content": {"contract_id": "INIT-SaaS-009", "clauses_extracted": 31}},
            "risk_scores.json": {"type": "json", "exists": True, "content": {"overall_risk": "critical", "score": 91}},
            "evidence.json": {"type": "json", "exists": False, "content": None},
            "synthesis.md": {"type": "markdown", "exists": False, "content": None},
            "approval_packet.json": {"type": "json", "exists": False, "content": None},
            "metrics.json": {"type": "json", "exists": True, "content": {"findings_count": 18, "exceptions_count": 5}},
            "findings_export.csv": {"type": "csv", "exists": True, "content": None},
            "validation_result.json": {"type": "json", "exists": True, "content": {"validation_status": "completed_with_findings"}},
            "posting_payload.json": {"type": "json", "exists": False, "content": None},
        },
    },
}


# ---------------------------------------------------------------------------
# CSS & HTML helpers
# ---------------------------------------------------------------------------
def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* App shell — remove default white header / decoration bar */
        .stApp {{
            background-color: {BG};
            color: {TEXT};
            font-family: {SANS};
        }}
        header[data-testid="stHeader"] {{
            background: {BG} !important;
            border-bottom: none;
            visibility: hidden;
            height: 0 !important;
            min-height: 0 !important;
            overflow: hidden;
        }}
        header[data-testid="stHeader"] > div {{
            background: transparent !important;
        }}
        [data-testid="stToolbar"] {{
            background: transparent !important;
        }}
        [data-testid="stDecoration"] {{
            background: {BG} !important;
        }}
        .main .block-container {{
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 100%;
        }}
        [data-testid="stSidebar"] {{ background-color: {CARD}; }}
        hr {{ border-color: {BORDER}; opacity: 0.6; }}

        /* Buttons — ensure readable text on dark backgrounds */
        .stButton > button,
        button[data-testid="stBaseButton-secondary"],
        button[data-testid="stBaseButton-primary"] {{
            border-radius: 8px !important;
            font-weight: 600 !important;
        }}
        .stButton > button[kind="secondary"],
        button[data-testid="stBaseButton-secondary"] {{
            background-color: {CARD} !important;
            color: {TEXT} !important;
            border: 1px solid {BORDER} !important;
        }}
        .stButton > button[kind="secondary"]:hover,
        button[data-testid="stBaseButton-secondary"]:hover {{
            background-color: #32323a !important;
            color: {TEXT} !important;
            border-color: {PRIMARY} !important;
        }}
        .stButton > button[kind="primary"],
        button[data-testid="stBaseButton-primary"] {{
            background-color: {PRIMARY} !important;
            color: #ffffff !important;
            border: 1px solid {PRIMARY} !important;
        }}
        .stButton > button[kind="primary"]:hover,
        button[data-testid="stBaseButton-primary"]:hover {{
            background-color: #2f5fc4 !important;
            color: #ffffff !important;
        }}

        /* Selectbox / multiselect — dark surfaces (#26262c) */
        [data-testid="stSelectbox"] div[data-baseweb="select"],
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
        [data-testid="stSelectbox"] div[data-baseweb="select"] > div > div,
        [data-testid="stMultiSelect"] div[data-baseweb="select"],
        [data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
        [data-testid="stMultiSelect"] div[data-baseweb="select"] > div > div,
        [data-testid="stMultiSelect"] [data-baseweb="input"] {{
            background-color: {CARD} !important;
            color: {TEXT} !important;
            border-color: {BORDER} !important;
        }}
        [data-testid="stSelectbox"] div[data-baseweb="select"] span,
        [data-testid="stMultiSelect"] div[data-baseweb="select"] span,
        [data-testid="stMultiSelect"] input {{
            background-color: transparent !important;
            color: {TEXT} !important;
        }}
        [data-testid="stMultiSelect"] [data-baseweb="tag"],
        [data-testid="stMultiSelect"] span[data-baseweb="tag"] {{
            background-color: {BORDER} !important;
            color: {TEXT} !important;
            border-color: {BORDER} !important;
        }}
        [data-testid="stSelectbox"] label,
        [data-testid="stMultiSelect"] label {{
            color: {MUTED} !important;
        }}
        [data-testid="stSelectbox"] svg,
        [data-testid="stMultiSelect"] svg {{
            fill: {MUTED} !important;
        }}

        /* Dropdown popover / open menu (Scenario, Severity, Category, Artifacts) */
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div,
        div[data-baseweb="popover"] ul,
        div[data-baseweb="popover"] li,
        [data-baseweb="menu"],
        [data-baseweb="menu"] ul,
        [data-baseweb="menu"] li,
        ul[role="listbox"],
        li[role="option"] {{
            background-color: {CARD} !important;
            color: {TEXT} !important;
        }}
        li[role="option"]:hover,
        [data-baseweb="menu"] li:hover,
        div[data-baseweb="popover"] li:hover {{
            background-color: #32323a !important;
            color: {TEXT} !important;
        }}
        li[role="option"][aria-selected="true"],
        [data-baseweb="menu"] li[aria-selected="true"],
        div[data-baseweb="popover"] li[aria-selected="true"] {{
            background-color: {PRIMARY}33 !important;
            color: {TEXT} !important;
        }}

        /*
         * Streamlit uses #31333f as default body text — force white everywhere
         * that color appears so dark-theme labels stay readable.
         */
        .stApp,
        .stApp p,
        .stApp span,
        .stApp label,
        .stApp h1, .stApp h2, .stApp h3, .stApp h4,
        .stMarkdown,
        .stMarkdown p,
        .stMarkdown span,
        .stMarkdown li,
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] span,
        [data-testid="stCaption"],
        [data-testid="stMetricLabel"] label,
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricValue"],
        [data-testid="stWidgetLabel"],
        [data-testid="stCheckbox"] label,
        [data-testid="stTabs"] [data-baseweb="tab"],
        [data-testid="stTabs"] [data-baseweb="tab"] p,
        [data-baseweb="select"] span,
        [data-baseweb="popover"] li,
        [data-baseweb="menu"] li,
        .stJson,
        .stDataFrame,
        div[style*="{STREAMLIT_DEFAULT_TEXT}"] {{
            color: #ffffff !important;
        }}

        .icras-header {{
            display: flex; align-items: center; justify-content: space-between;
            padding: 0.75rem 0 1rem 0; gap: 1rem; flex-wrap: wrap;
        }}
        .icras-brand {{ display: flex; align-items: center; gap: 0.85rem; }}
        .icras-logo {{
            width: 42px; height: 42px; background: {PRIMARY}; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-family: {MONO}; font-weight: 700; font-size: 0.85rem; color: #fff;
        }}
        .icras-title {{ font-size: 1.35rem; font-weight: 700; line-height: 1.2; margin: 0; color: {TEXT}; }}
        .icras-subtitle {{ font-size: 0.78rem; color: {MUTED}; margin: 0; }}
        .icras-kpi-label {{
            font-size: 0.68rem; font-weight: 600; letter-spacing: 0.06em;
            text-transform: uppercase; color: {MUTED}; margin-bottom: 0.35rem;
        }}
        .icras-kpi-value {{ font-size: 1.45rem; font-weight: 700; color: {TEXT}; line-height: 1.2; }}
        .icras-kpi-sub {{ font-size: 0.78rem; color: {MUTED}; margin-top: 0.35rem; }}
        .icras-decision-row {{ display: flex; align-items: center; gap: 0.5rem; }}
        .icras-dot {{ width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }}
        .icras-progress {{
            height: 4px; background: {BORDER}; border-radius: 2px; margin-top: 0.5rem; overflow: hidden;
        }}
        .icras-progress-fill {{ height: 100%; background: {PRIMARY}; border-radius: 2px; }}
        .icras-badge {{
            display: inline-block; padding: 0.2rem 0.65rem; border-radius: 999px;
            font-size: 0.75rem; font-weight: 600; text-transform: capitalize;
            width: fit-content; max-width: max-content;
        }}
        .icras-agent-card {{
            background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px;
            padding: 0.75rem 0.85rem; min-height: 118px;
        }}
        .icras-agent-letter {{
            display: inline-block; background: {BORDER}; color: {TEXT};
            font-family: {MONO}; font-size: 0.7rem; font-weight: 700;
            padding: 0.15rem 0.4rem; border-radius: 4px; margin-right: 0.35rem;
        }}
        .icras-agent-role {{ font-size: 0.78rem; font-weight: 600; color: {TEXT}; }}
        .icras-agent-status {{ font-size: 0.75rem; margin-top: 0.45rem; display: flex; align-items: center; gap: 0.35rem; }}
        .icras-agent-meta {{ font-size: 0.68rem; color: {MUTED}; margin-top: 0.35rem; font-family: {MONO}; }}
        .icras-status-pill {{
            display: inline-flex; align-items: center; gap: 0.35rem;
            background: {CARD}; border: 1px solid {BORDER}; border-radius: 999px;
            padding: 0.35rem 0.75rem; font-size: 0.78rem; color: {MUTED};
        }}
        .icras-section-title {{
            font-size: 0.95rem; font-weight: 600; color: {TEXT};
            margin: 1.25rem 0 0.75rem 0;
        }}
        div[data-testid="stMetric"] {{
            background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; padding: 0.75rem;
        }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 0.5rem; background: transparent; }}
        .stTabs [data-baseweb="tab"] {{
            background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px 8px 0 0;
            color: {MUTED}; padding: 0.5rem 1.25rem;
        }}
        .stTabs [aria-selected="true"] {{ background: {BG}; color: {PRIMARY}; border-bottom-color: {PRIMARY}; }}
        .artifact-missing {{ color: {MUTED}; font-style: italic; font-size: 0.85rem; }}
        .manual-flag {{ color: {RISK_COLORS['high']}; font-weight: 600; }}

        /* Code / JSON preview — dark panel instead of default white block */
        [data-testid="stJson"],
        [data-testid="stJson"] > div,
        [data-testid="stCodeBlock"],
        [data-testid="stCodeBlock"] > div,
        [data-testid="stCodeBlock"] pre,
        [data-testid="stCodeBlock"] code,
        .stCodeBlock,
        .stCodeBlock pre,
        .stCodeBlock code {{
            background-color: {CARD} !important;
            color: #ffffff !important;
            border: 1px solid {BORDER} !important;
            border-radius: 10px !important;
        }}
        [data-testid="stCodeBlock"] pre {{
            padding: 1rem !important;
            font-family: {MONO} !important;
            font-size: 0.82rem !important;
            line-height: 1.5 !important;
            white-space: pre-wrap !important;
            word-break: break-word !important;
        }}
        .icras-code-preview {{
            background-color: {CARD};
            color: #ffffff;
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 1rem 1.1rem;
            margin: 0;
            font-family: {MONO};
            font-size: 0.82rem;
            line-height: 1.5;
            white-space: pre-wrap;
            word-break: break-word;
            overflow-x: auto;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_color(level: str) -> str:
    return RISK_COLORS.get(level.lower(), MUTED)


def risk_badge_html(level: str) -> str:
    color = risk_color(level)
    return (
        f'<span class="icras-badge" style="background:{color}22;color:{color};'
        f'border:1px solid {color}55;">{level}</span>'
    )


def progress_bar_html(pct: int, color: str = PRIMARY) -> str:
    pct = max(0, min(100, pct))
    return (
        f'<div class="icras-progress"><div class="icras-progress-fill" '
        f'style="width:{pct}%;background:{color};"></div></div>'
    )


def kpi_card(label: str, value_html: str, sub_html: str = "") -> None:
    st.markdown(
        f'<div class="icras-kpi-label">{label}</div>'
        f'<div class="icras-kpi-value">{value_html}</div>'
        f'<div class="icras-kpi-sub">{sub_html}</div>',
        unsafe_allow_html=True,
    )


def agent_card_html(agent: dict) -> str:
    letter = agent["id"]
    role = AGENT_ROSTER.get(letter, "Unknown")
    status_key = agent.get("status", "skipped")
    label, color, icon = STATUS_STYLE.get(status_key, STATUS_STYLE["skipped"])
    artifact = agent.get("artifact", "—")
    duration = agent.get("duration_s")
    dur_text = f"{duration:.1f}s" if isinstance(duration, (int, float)) else "—"
    return f"""
    <div class="icras-agent-card">
        <div><span class="icras-agent-letter">{letter}</span>
        <span class="icras-agent-role">{role}</span></div>
        <div class="icras-agent-status">
            <span style="color:{color};">{icon}</span>
            <span style="color:{color};">{label}</span>
        </div>
        <div class="icras-agent-meta">{artifact}</div>
        <div class="icras-agent-meta">{dur_text}</div>
    </div>
    """


def plotly_dark_layout() -> dict:
    return dict(
        paper_bgcolor=CARD,
        plot_bgcolor=CARD,
        font=dict(color=TEXT, family=SANS),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(bgcolor=CARD, bordercolor=BORDER),
    )


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------
def render_header(scenario: str, last_run: str) -> bool:
    """Top bar. Returns True if Run Pipeline was clicked."""
    brand_col, controls_col = st.columns([2.5, 3.5])

    with brand_col:
        st.markdown(
            f"""
            <div class="icras-brand">
                <div class="icras-logo">IC</div>
                <div>
                    <p class="icras-title">ICRAS</p>
                    <p class="icras-subtitle">Intelligent Contract Risk Analysis System</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with controls_col:
        c_scenario, c_status, c_run = st.columns([2.2, 1.6, 1.2])
        with c_scenario:
            st.markdown(
                '<div style="font-size:0.75rem;color:#9a9aa3;margin-bottom:0.25rem;">📋 Scenario</div>',
                unsafe_allow_html=True,
            )
            selected = st.selectbox(
                "Scenario",
                options=list(MOCK_SCENARIOS.keys()),
                index=list(MOCK_SCENARIOS.keys()).index(scenario),
                label_visibility="collapsed",
            )
            if selected != st.session_state.get("scenario"):
                st.session_state["scenario"] = selected
                st.session_state.pop("last_run_override", None)
            else:
                st.session_state["scenario"] = selected

        with c_status:
            st.markdown(
                f'<div class="icras-status-pill" style="margin-top:1.6rem;">'
                f'Last run · {last_run}</div>',
                unsafe_allow_html=True,
            )

        with c_run:
            run_clicked = st.button(
                "▶ Run Pipeline",
                type="primary",
                width="stretch",
            )

    st.divider()
    return run_clicked


def render_kpi_grid(summary: dict) -> None:
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        with st.container(border=True):
            dot = summary.get("decision_dot", MUTED)
            decision = summary.get("decision", "—")
            kpi_card(
                "Decision",
                f'<div class="icras-decision-row"><span class="icras-dot" '
                f'style="background:{dot};"></span>{decision}</div>',
            )

    with c2:
        with st.container(border=True):
            risk = summary.get("overall_risk", "medium")
            kpi_card("Overall Risk", risk_badge_html(risk))

    with c3:
        with st.container(border=True):
            total = summary.get("total_findings", 0)
            manual = summary.get("manual_review_count", 0)
            kpi_card(
                "Total Findings",
                str(total),
                f"{total} findings · {manual} need manual review",
            )

    with c4:
        with st.container(border=True):
            conf = summary.get("confidence_pct", 0)
            kpi_card(
                "Confidence",
                f"{conf}%",
                progress_bar_html(conf),
            )

    with c5:
        with st.container(border=True):
            cov = summary.get("evidence_coverage_pct", 0)
            kpi_card(
                "Evidence Coverage",
                f"{cov}%",
                progress_bar_html(cov, RISK_COLORS["low"]),
            )


def render_agent_pipeline(agents: list[dict]) -> None:
    st.markdown('<p class="icras-section-title">Agent Pipeline</p>', unsafe_allow_html=True)
    cols = st.columns(8)
    for col, agent in zip(cols, agents):
        with col:
            st.markdown(agent_card_html(agent), unsafe_allow_html=True)


def findings_table_html(df: pd.DataFrame) -> str:
    """Render findings as an HTML table with severity badges and manual-review highlight."""
    headers = list(df.columns)
    rows_html = []
    for _, row in df.iterrows():
        manual = row.get("Manual Review") == "⚑ Yes"
        row_style = "background:#f59e0b12;" if manual else ""
        cells = []
        for col in headers:
            val = row[col]
            if col == "Severity":
                color = risk_color(str(val).lower())
                cell = (
                    f'<span class="icras-badge" style="background:{color}22;color:{color};'
                    f'border:1px solid {color}55;">{val}</span>'
                )
            elif col == "Manual Review" and val == "⚑ Yes":
                cell = f'<span class="manual-flag">{val}</span>'
            else:
                cell = str(val)
            cells.append(f"<td style='padding:0.45rem 0.65rem;border-bottom:1px solid {BORDER};{row_style}'>{cell}</td>")
        rows_html.append(f"<tr style='{row_style}'>{''.join(cells)}</tr>")

    thead = "".join(
        f"<th style='text-align:left;padding:0.5rem 0.65rem;color:{MUTED};"
        f"font-size:0.72rem;text-transform:uppercase;letter-spacing:0.04em;"
        f"border-bottom:1px solid {BORDER};'>{h}</th>"
        for h in headers
    )
    return (
        f"<div style='overflow-x:auto;border:1px solid {BORDER};border-radius:10px;'>"
        f"<table style='width:100%;border-collapse:collapse;font-size:0.85rem;color:{TEXT};'>"
        f"<thead><tr>{thead}</tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>"
    )


def render_findings_tab(findings: list[dict]) -> None:
    df = pd.DataFrame(findings)
    if df.empty:
        st.info("No findings for this scenario.")
        return

    df = df.rename(
        columns={
            "id": "ID",
            "clause": "Clause/Title",
            "category": "Category",
            "severity": "Severity",
            "risk_level": "Risk Level",
            "confidence": "Confidence",
            "manual_review": "Manual Review",
            "evidence_refs": "Evidence refs",
        }
    )
    df["Confidence"] = (df["Confidence"] * 100).round(0).astype(int).astype(str) + "%"
    df["Manual Review"] = df["Manual Review"].map({True: "⚑ Yes", False: "—"})

    fc1, fc2, fc3 = st.columns([2, 2, 1.5])
    severities = sorted(df["Severity"].unique(), key=lambda s: SEVERITY_ORDER.index(s) if s in SEVERITY_ORDER else 99)
    with fc1:
        selected_sev = st.multiselect("Severity", options=severities, default=severities)
    with fc2:
        categories = sorted(df["Category"].unique())
        selected_cat = st.multiselect("Category", options=categories, default=categories)
    with fc3:
        manual_only = st.checkbox("Manual review only")

    filtered = df[
        df["Severity"].isin(selected_sev) & df["Category"].isin(selected_cat)
    ]
    if manual_only:
        filtered = filtered[filtered["Manual Review"] == "⚑ Yes"]

    st.caption(f"Showing {len(filtered)} of {len(df)} findings")
    st.markdown(findings_table_html(filtered), unsafe_allow_html=True)


def render_analytics_tab(findings: list[dict], clause_confidence: list[dict], summary: dict) -> None:
    df = pd.DataFrame(findings)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Findings", summary.get("total_findings", 0))
    k2.metric("Manual Review", summary.get("manual_review_count", 0))
    k3.metric("Confidence", f"{summary.get('confidence_pct', 0)}%")
    k4.metric("Evidence Coverage", f"{summary.get('evidence_coverage_pct', 0)}%")

    st.divider()
    col_pie, col_bar = st.columns(2)

    with col_pie:
        st.markdown("**Findings by Severity**")
        if not df.empty:
            sev_counts = df["severity"].value_counts().reset_index()
            sev_counts.columns = ["severity", "count"]
            colors = [risk_color(s) for s in sev_counts["severity"]]
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=sev_counts["severity"],
                        values=sev_counts["count"],
                        hole=0.55,
                        marker=dict(colors=colors),
                        textinfo="label+value",
                        textfont=dict(color=TEXT),
                    )
                ]
            )
            fig.update_layout(**plotly_dark_layout(), showlegend=True, height=340)
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("No findings data.")

    with col_bar:
        st.markdown("**Findings by Category**")
        if not df.empty:
            cat_counts = (
                df.groupby("category")
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=True)
            )
            fig = px.bar(
                cat_counts,
                x="count",
                y="category",
                orientation="h",
                color_discrete_sequence=[PRIMARY],
            )
            fig.update_layout(**plotly_dark_layout(), height=340, xaxis_title="Count", yaxis_title="")
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("No findings data.")

    st.markdown("**Clause Confidence Overview**")
    cc_df = pd.DataFrame(clause_confidence)
    if not cc_df.empty:
        cc_df["confidence_pct"] = (cc_df["confidence"] * 100).round(1)
        colors = [risk_color(r) for r in cc_df["risk_level"]]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=cc_df["clause"],
                    y=cc_df["confidence_pct"],
                    marker_color=colors,
                )
            ]
        )
        fig.update_layout(
            **plotly_dark_layout(),
            height=360,
            xaxis_title="",
            yaxis_title="Confidence %",
            yaxis=dict(range=[0, 100]),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("No clause confidence data.")

    manual = df[df["manual_review"] == True] if not df.empty else pd.DataFrame()  # noqa: E712
    if not manual.empty:
        st.markdown(
            f"**Manual review summary:** {len(manual)} finding(s) flagged "
            f"({round(100 * len(manual) / len(df), 1)}% of total)."
        )


def _artifact_download_bytes(name: str, meta: dict, findings: list[dict]) -> bytes:
    if name == "findings_export.csv" and meta.get("type") == "csv":
        return pd.DataFrame(findings).to_csv(index=False).encode("utf-8")
    content = meta.get("content")
    if meta.get("type") == "markdown" and isinstance(content, str):
        return content.encode("utf-8")
    return json.dumps(content, indent=2).encode("utf-8")


def render_json_preview(content) -> None:
    """Pretty-print JSON in a dark code block (readable on the dark theme)."""
    text = json.dumps(content, indent=2, ensure_ascii=False)
    st.markdown(
        f'<pre class="icras-code-preview">{html.escape(text)}</pre>',
        unsafe_allow_html=True,
    )


def render_artifacts_tab(artifacts: dict, findings: list[dict]) -> None:
    list_col, preview_col = st.columns([1, 2.2])

    artifact_names = list(artifacts.keys())
    if "selected_artifact" not in st.session_state:
        st.session_state["selected_artifact"] = artifact_names[0]

    with list_col:
        st.markdown("**Artifacts**")
        selected = st.selectbox(
            "Artifact file",
            options=artifact_names,
            index=artifact_names.index(
                st.session_state.get("selected_artifact", artifact_names[0])
            ),
            format_func=lambda n: (
                n if artifacts[n].get("exists") else f"{n}  ·  N/A"
            ),
            label_visibility="collapsed",
            key="artifact_select",
        )
        st.session_state["selected_artifact"] = selected

    with preview_col:
        selected = st.session_state.get("selected_artifact", artifact_names[0])
        meta = artifacts.get(selected, {})
        st.markdown(f"**Preview:** `{selected}`")

        if not meta.get("exists", False):
            st.markdown(
                '<p class="artifact-missing">Artifact not generated yet.</p>',
                unsafe_allow_html=True,
            )
            return

        atype = meta.get("type", "json")
        content = meta.get("content")

        if atype == "markdown" and isinstance(content, str):
            st.markdown(content)
        elif atype == "csv":
            st.dataframe(pd.DataFrame(findings), width="stretch", hide_index=True)
        else:
            render_json_preview(content)

        st.download_button(
            "Download",
            data=_artifact_download_bytes(selected, meta, findings),
            file_name=selected,
            mime="application/octet-stream",
            width="stretch",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        layout="wide",
        page_title="ICRAS — Contract Risk Analysis",
        initial_sidebar_state="collapsed",
    )
    inject_theme()

    if "scenario" not in st.session_state:
        st.session_state["scenario"] = list(MOCK_SCENARIOS.keys())[0]

    scenario_key = st.session_state["scenario"]
    data = MOCK_SCENARIOS[scenario_key]

    if "last_run_override" in st.session_state:
        last_run = st.session_state["last_run_override"]
    else:
        last_run = data["last_run"]

    run_clicked = render_header(scenario_key, last_run)

    if run_clicked:
        st.session_state["last_run_override"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with st.spinner("Running pipeline…"):
            pass
        st.rerun()

    # Refresh data if scenario changed via selectbox
    data = MOCK_SCENARIOS[st.session_state["scenario"]]
    if "last_run_override" in st.session_state:
        last_run = st.session_state["last_run_override"]
    else:
        last_run = data["last_run"]

    render_kpi_grid(data["run_summary"])
    render_agent_pipeline(data["agents"])

    tab_findings, tab_analytics, tab_artifacts = st.tabs(
        ["Findings", "Analytics", "Artifacts"]
    )

    with tab_findings:
        render_findings_tab(data["findings"])

    with tab_analytics:
        render_analytics_tab(
            data["findings"],
            data["clause_confidence"],
            data["run_summary"],
        )

    with tab_artifacts:
        render_artifacts_tab(data["artifacts"], data["findings"])


if __name__ == "__main__":
    main()
