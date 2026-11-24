# Employer Anti-Fraud OSINT Verification Engine (Stage 2 — Anthropic-Only Stack)

Production-ready, fault-tolerant FastAPI backend + built-in Web Dashboard for detecting fraudulent job postings and fake recruiting agencies targeting foreign workers in Europe (**Czech ARES + Poland MF White List + EU OSINT via Anthropic Web Search**).

Built on an **Anthropic-only stack** — zero Serper or Google Places API keys required.

## Stage 2 Capabilities

- **Built-in Web Dashboard (`GET /`):** Drag-and-drop screenshots (`.png`/`.jpg`), Word contracts (`.docx`), PDF presentations (`.pdf`), or paste a job URL (`layboard.com`, `eures`, public `instagram.com/p/...`), inspect full score breakdowns, browse verification history, and manage the blacklist directly in the browser.
- **Multimodal Intake Endpoint (`POST /api/v1/verify/intake`):** Extracts structured `VerificationRequest` fields from URLs, `.docx` contracts, `.pdf` documents (text + native visual PDF fallback), screenshots via Claude Vision, and raw text — with SSRF protection and offline regex fallback.
- **EU Government Registries (Module A, max 25 pts):**
  - **Czech Republic (`CZ`):** Live Czech ARES API + automatic IČO resolution by company name when `reg_id="UNKNOWN"`.
  - **Poland (`PL`):** Live official Ministry of Finance White List API (`wl-api.mf.gov.pl`) by 10-digit NIP.
- **Upgraded Domain, WHOIS/RDAP & Site Crawler (Module B, max 25 pts):**
  - Handles large Wix/Webflow pages (1.5 MB limit), extracts JSON-LD (`application/ld+json`) contacts and official social URLs, filters Sentry/Wixpress telemetry emails, and falls back to public RDAP when WHOIS redacts GDPR creation dates.
  - Passes verified `official_site_social_urls` to Claude so legitimate employer Instagram/Facebook accounts are not misflagged as impersonation.
- **Calibrated Unified Claude OSINT & Vacancy Auditor (Module C, max 50 pts):**
  - Official `anthropic` SDK (`AsyncAnthropic`) + server-side `web_search_20250305` (`max_uses=5`), compatible with the latest Anthropic API (never passes `temperature`, `top_p`, or `top_k`).
  - Credit-saving early abort skips Claude whenever Step 0 or Stage 1 triggers a hard kill-switch (`BLACKLISTED_ENTITY`, `UNREGISTERED_ENTITY`, `COMPANY_LIQUIDATED_OR_INSOLVENT`).
  - Calibrated `UPFRONT_FEE_SCAM` detection: distinguishes illegal agency/broker upfront payments from reimbursable official embassy consular fees and normal worker rent.
  - Filters generic encyclopedia/boilerplate URLs out of `evidence_links`.

---

## Project Structure

```text
.
├── app/
│   ├── main.py                        # FastAPI app, Stage 2 intake, history, blacklist & dashboard routes
│   ├── config.py                      # Pydantic BaseSettings (.env loading)
│   ├── database.py                    # Async SQLAlchemy 2.0 engine & sessions
│   ├── models.py                      # ORM tables: companies, verification_logs, blacklist
│   ├── schemas.py                     # Pydantic v2 contracts & deterministic normalizers
│   ├── orchestrator.py                # Two-stage credit-saving pipeline, scoring & verdict math
│   ├── static/
│   │   └── index.html                 # Built-in Stage 2 Web Dashboard UI
│   └── modules/
│       ├── ares.py                    # Module A: Czech ARES + Poland MF NIP registry client (25 pts)
│       ├── domain_crawler.py          # Module B: WHOIS/RDAP + JSON-LD + async EU crawler (25 pts)
│       ├── claude_osint_auditor.py    # Module C: Anthropic SDK + web_search_20250305 (50 pts)
│       └── intake.py                  # Multimodal URL / DOCX / PDF / Screenshot intake parser
├── tests/
│   └── test_engine.py                 # Full Stage 1 + Stage 2 pytest suite
├── .env.example
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Quick Start

```powershell
# 1. Activate virtualenv & install dependencies
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Run full pytest suite
pytest -v

# 3. Start local server (SQLite zero-config)
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Web Dashboard UI:** `http://127.0.0.1:8000/`
- **Swagger / OpenAPI Docs:** `http://127.0.0.1:8000/docs`

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Built-in Stage 2 Web Dashboard UI |
| `POST` | `/api/v1/verify` | Structured JSON verification |
| `POST` | `/api/v1/verify/intake` | Multimodal intake (URL, `.docx`, `.pdf`, screenshots, raw text) |
| `GET` | `/api/v1/verifications` | Paginated verification history (`?limit=25&offset=0&status=RED`) |
| `GET` | `/api/v1/verify/{verification_id}` | Fetch stored verification report by UUID |
| `GET` | `/api/v1/blacklist` | List blacklisted entities |
| `POST` | `/api/v1/blacklist` | Add normalized `reg_id`, `domain`, `phone`, or `email` to blacklist |
| `DELETE` | `/api/v1/blacklist/{entry_id}` | Remove entry from blacklist |
| `GET` | `/health` | Service liveness, DB status & configured integrations |
