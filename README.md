# ICRAS

Intelligent Contract Risk Analysis System.


## Description

This task initializes the repository structure, local project architecture, agents, services, schemas, policies, tests, documentation, and Git setup.

## Optional Streamlit Demo UI

An optional, demo-focused dashboard is available at `app/ui.py`. Run it with:

```bash
streamlit run app/ui.py
```

The dashboard is a dark, observability-style demo UI driven by mock scenario
data (Acme MSA, Globex NDA, Initech SaaS). It shows run KPIs, agent pipeline
status (A–H), a filterable findings table, Plotly analytics charts, and an
artifact preview/download pane. Swap the ``MOCK_SCENARIOS`` dictionary for real
pipeline output when ready.

This UI is a stretch/demo layer only. It does not implement or modify any
business logic and is entirely optional and safe to remove.

## One-command CLI demo

The canonical entry point runs the full pipeline (Agent A → B → C → D → E → H)
on a single bundle. Run it in deterministic synthetic mode with
`ENV_MODE=test`:

```bash
ENV_MODE=test python -m app.main --bundle data/bundles/scenario_03_net_90_payment_terms
```

`ENV_MODE=test` selects the curated synthetic extraction fixtures so the demo is
fully deterministic and reproduces the documented per-scenario decisions (see
the decision table below). This is the supported way to reproduce the expected
outcomes.

> **Note on live mode.** Without `ENV_MODE=test`, Agent B performs real PDF
> parsing. Live extraction is best-effort and may extract clauses with lower
> confidence or miss optional clauses, so a clean contract can be scored
> differently from synthetic mode. For reproducing the expected scenario
> decisions, always use `ENV_MODE=test`.

### Expected scenario decisions (synthetic mode)

| Scenario | Expected decision | Overall risk |
| --- | --- | --- |
| `scenario_01_clean_nda` | `auto_approve` | low |
| `scenario_02_missing_liability_cap` | `legal_review_required` | high/critical |
| `scenario_03_net_90_payment_terms` | `finance_review_required` | high/critical |
| `scenario_04_high_risk_jurisdiction` | `compliance_review_required` | high/critical |
| `scenario_05_auto_renewal_no_opt_out` | `legal_review_required` | high |
| `scenario_06_conflicting_governing_law` | `legal_review_required` | high/critical |
| `scenario_07_low_confidence_signature` | `manual_review_required` | medium/high |
| `scenario_08_missing_gdpr_clause` | `compliance_review_required` | high/critical |
| `scenario_09_clean_services_agreement` | `auto_approve` | low |
| `scenario_10_multiple_high_risks` | `reject_or_block` | critical |

These outcomes are enforced by `tests/test_scenario_expected_decisions.py`.
