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

The required one-command CLI demo remains the canonical entry point:

```bash
python -m app.main --bundle data/bundles/scenario_03_net_90_payment_terms
```
