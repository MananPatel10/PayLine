# Payline — System & Architecture Handoff

## Overview
Enterprise-grade financial intelligence engine and algorithmic micro-investment platform.
Ingests heterogeneous bank statement CSV exports -> Edge ML classification ->
2-sigma anomaly & overcharge detection -> 50/30/20 wealth audit ->
Automated spare-change micro-investing -> Google Gemini 3.8 Flash generative CFO advisory.

## Technical Architecture
- **De-Cluttered Institutional Design:** Minimalist layout inspired by Payline/Revolut, soft gray canvas (`#eef2f6`), pure white inner card (`#ffffff`), matte black pill navbar (`#090a0f`), electric lime (`#c2eb24`) and indigo accents. Zero emojis or symbols across all templates and code.
- **Strict Route & Feature Segregation:**
  - `GET /` — Explanatory landing page explaining system architecture, technical pillars, and data flows. No feature widgets or upload boxes on the home page.
  - `GET /upload` — Dedicated statement intake portal with drag-and-drop CSV upload and 1-click verified benchmark statement loader.
  - `GET /dashboard` — Financial health KPIs, 50/30/20 compliance audit, category breakdowns, 2-sigma anomaly alerts, and searchable transaction ledger.
  - `GET /invest` — Dedicated micro-investment engine (round-up rules, AI safe-to-invest liquidity buffer, tri-asset allocation pockets, and 10-year compound growth matrix).
  - `GET /advisor` — Executive Gemini AI Advisor powered by Google Gemini 3.8 Flash with automated CFO dossier and interactive natural-language query terminal.
  - `POST /api/chat` — Context-grounded Gemini query endpoint answering ad-hoc user financial inquiries.
  - `GET /api/analysis` — Machine-readable JSON telemetry.

## Project Structure
```
payline/
├── analyzer.py          - Core engine: CSV ingestion, ML categorizer, spending
│                          analytics, 2-sigma anomaly detection, 50/30/20 audit,
│                          and micro-investment compounding models.
├── gemini_advisor.py    - Institutional CFO advisor powered by Google Gemini API
│                          with multi-model fallback and resilient offline heuristic.
├── app.py               - Flask server wiring all 5 dedicated segments and APIs.
├── templates/           - Clean Jinja2 templates (zero emojis, modular layout):
│   ├── base.html        - Institutional master layout with top pill navbar & status.
│   ├── index.html       - Explanatory landing page & architecture overview.
│   ├── upload.html      - Statement intake & normalization portal.
│   ├── dashboard.html   - Financial health & 2-sigma anomaly dashboard.
│   ├── invest.html      - Algorithmic micro-investment & wealth compounding engine.
│   └── advisor.html     - Executive Gemini AI CFO dossier & query terminal.
├── sample_data.py       - Generates realistic 3-month Indian bank statements.
├── sample_statement.csv - Verified 3-month dataset (103 transactions, INR 239k credits).
├── requirements.txt     - Dependencies: flask, pandas, numpy, scikit-learn, werkzeug.
└── README.md            - System documentation.
```

## Running the Application
```bash
# Start Flask server
python app.py
```
Open browser at: `http://localhost:5000`
