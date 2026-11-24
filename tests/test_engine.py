"""Comprehensive pytest suite for the Anthropic-only Employer Verification Engine (Stage 1).

Covers:
- Deterministic normalization, input hashing, and verdict/confidence scoring math
- Module A (Czech ARES) and Module B (Domain, WHOIS & async site crawler)
- Unified Module C (claude_osint_auditor.py) with official Anthropic SDK + web_search_20250305:
  - Asserts temperature/top_p/top_k are NEVER passed to messages.create()
  - Asserts cited & web_search_tool_result URLs are extracted into evidence_links
  - Asserts strict Pydantic validation, score clamping (20 + 15 + 15 = 50), and UPFRONT_FEE_SCAM
  - Asserts automatic fallback when Anthropic rejects an unsupported country code
- Credit-saving early abort: verifies Claude is NEVER called on BLACKLISTED_ENTITY,
  UNREGISTERED_ENTITY, or COMPANY_LIQUIDATED_OR_INSOLVENT kill-switches
- FastAPI routes, SQL caching (14-day TTL), force_refresh, blacklist & timeout fault tolerance
- Real live integration test against the public Czech ARES REST API
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from anthropic import BadRequestError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.database import Base, create_async_db_engine, get_db
from app.main import app
from app.modules.ares import (
    _evaluate_czech_ares_payload,
    _evaluate_poland_mf_payload,
    check_gov_registry,
    search_czech_ares_by_name,
)
from app.modules.claude_osint_auditor import ClaudeOSINTCheckOutput, audit_with_claude_osint
from app.modules.domain_crawler import DomainCheckOutput, check_domain_and_site
from app.modules.intake import (
    UploadedAttachment,
    extract_text_from_docx,
    extract_verification_request,
)
from app.orchestrator import compute_verdict, run_verification_pipeline
from app.schemas import (
    ApplicantType,
    ModuleResult,
    ModulesOutput,
    ModuleStatus,
    VerificationRequest,
    VerificationStatus,
    compute_input_hash,
    extract_root_domain,
    normalize_phone_e164,
    normalize_reg_id,
)

# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db_session() -> AsyncSession:
    """Provide an isolated in-memory async SQLite database session per test."""
    test_engine = create_async_db_engine("sqlite+aiosqlite:///:memory:")
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    async with session_factory() as session:
        yield session

    await test_engine.dispose()


def _make_settings(**overrides: Any) -> Settings:
    """Create isolated Settings without reading host .env keys."""
    defaults: dict[str, Any] = {
        "DATABASE_URL": "sqlite+aiosqlite:///:memory:",
        "ANTHROPIC_API_KEY": None,
        "ANTHROPIC_MODEL": "claude-haiku-4-5-20251001",
        "CLAUDE_MAX_SEARCH_USES": 5,
        "MODULE_TIMEOUT_SECONDS": 5.0,
        "CLAUDE_TIMEOUT_SECONDS": 10.0,
        "CACHE_TTL_DAYS": 14,
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def _sample_request(**overrides: Any) -> VerificationRequest:
    data: dict[str, Any] = {
        "company_name": "Acme Czech Logistics s.r.o.",
        "country_code": "CZ",
        "reg_id": "261 68 685",
        "website_url": "https://www.acmelogistics.cz",
        "contact_email": "hr@acmelogistics.cz",
        "contact_phone": "+420 777 123 456",
        "applicant_type": ApplicantType.direct_employer,
        "vacancy_text": (
            "Warehouse Coordinator in Prague. Gross salary 42,000 CZK/month, "
            "standard HPP employment contract, 25 days vacation, no upfront fees."
        ),
        "force_refresh": False,
    }
    data.update(overrides)
    return VerificationRequest(**data)


class _FakeAnthropicMessages:
    """Fake AsyncAnthropic.messages namespace capturing exact kwargs for assertions."""

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        next_item = self._responses.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item


class _FakeAnthropicClient:
    def __init__(self, responses: list[Any]) -> None:
        self.messages = _FakeAnthropicMessages(responses)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _build_fake_claude_response(
    audit_payload: dict[str, Any],
    search_urls: list[str] | None = None,
    citation_urls: list[str] | None = None,
) -> Any:
    """Build a realistic Anthropic SDK response containing web_search blocks and final JSON text."""
    blocks: list[Any] = [
        SimpleNamespace(type="server_tool_use", name="web_search", input={"query": "test"}),
        SimpleNamespace(
            type="web_search_tool_result",
            content=[
                SimpleNamespace(type="web_search_result", url=u, title="Result")
                for u in (search_urls or [])
            ],
        ),
        SimpleNamespace(
            type="text",
            text=f"```json\n{json.dumps(audit_payload)}\n```",
            citations=[
                SimpleNamespace(type="web_search_result_location", url=u, title="Cited")
                for u in (citation_urls or [])
            ],
        ),
    ]
    return SimpleNamespace(content=blocks, stop_reason="end_turn")


# ---------------------------------------------------------------------------
# 1. Normalization & Deterministic Verdict Scoring Tests
# ---------------------------------------------------------------------------


def test_normalization_helpers() -> None:
    assert normalize_reg_id(" 261 68 685 ", "CZ") == "26168685"
    assert normalize_reg_id("177041", "CZ") == "00177041"
    assert extract_root_domain("https://careers.sub.example.cz/jobs") == "example.cz"
    assert extract_root_domain("hr@mail.example.co.uk") == "example.co.uk"
    assert normalize_phone_e164("777 123 456", "CZ") == "+420777123456"
    assert normalize_phone_e164("00420 (234) 694-111", "CZ") == "+420234694111"

    req1 = _sample_request(company_name="  Acme Czech   Logistics s.r.o. ", reg_id="261 68 685")
    req2 = _sample_request(company_name="acme czech logistics s.r.o.", reg_id="26168685")
    assert compute_input_hash(req1) == compute_input_hash(req2)


def test_compute_verdict_decision_tree() -> None:
    def make_modules(
        a: tuple[ModuleStatus, int],
        b: tuple[ModuleStatus, int],
        c: tuple[ModuleStatus, int],
    ) -> ModulesOutput:
        return ModulesOutput(
            gov_registry=ModuleResult(status=a[0], score=a[1], max_score=25),
            domain_site=ModuleResult(status=b[0], score=b[1], max_score=25),
            claude_osint=ModuleResult(status=c[0], score=c[1], max_score=50),
        )

    # Rule 2: >=80 and confidence >=0.70 -> GREEN
    status, score, conf = compute_verdict(
        make_modules(
            (ModuleStatus.SUCCESS, 25),
            (ModuleStatus.SUCCESS, 20),
            (ModuleStatus.SUCCESS, 45),
        ),
        [],
    )
    assert status == VerificationStatus.GREEN
    assert score == 90
    assert conf == 1.0

    # Rule 3: Only free Modules A & B succeed (50/50, conf=0.50) -> YELLOW
    status, score, conf = compute_verdict(
        make_modules(
            (ModuleStatus.SUCCESS, 25),
            (ModuleStatus.SUCCESS, 25),
            (ModuleStatus.UNAVAILABLE, 0),
        ),
        [],
    )
    assert status == VerificationStatus.YELLOW
    assert score == 50
    assert conf == 0.5

    # Rule 3: Score >= 80 with confidence < 0.70 -> YELLOW
    low_conf_modules = ModulesOutput(
        gov_registry=ModuleResult(status=ModuleStatus.SUCCESS, score=40, max_score=30),
        domain_site=ModuleResult(status=ModuleStatus.SUCCESS, score=45, max_score=35),
        claude_osint=ModuleResult(status=ModuleStatus.UNAVAILABLE, score=0, max_score=35),
    )
    status, score, conf = compute_verdict(low_conf_modules, [])
    assert status == VerificationStatus.YELLOW
    assert score == 85
    assert conf == 0.65

    # Rule 4: score < 50 -> RED
    status, score, conf = compute_verdict(
        make_modules(
            (ModuleStatus.SUCCESS, 15),
            (ModuleStatus.SUCCESS, 10),
            (ModuleStatus.SUCCESS, 20),
        ),
        [],
    )
    assert status == VerificationStatus.RED
    assert score == 45
    assert conf == 1.0

    # Rule 1: Kill-switch overrides perfect 100/100 -> RED, trust_score = 0
    status, score, conf = compute_verdict(
        make_modules(
            (ModuleStatus.SUCCESS, 25),
            (ModuleStatus.SUCCESS, 25),
            (ModuleStatus.SUCCESS, 50),
        ),
        ["UPFRONT_FEE_SCAM"],
    )
    assert status == VerificationStatus.RED
    assert score == 0
    assert conf == 1.0


# ---------------------------------------------------------------------------
# 2. Module A (Czech ARES) Unit Tests
# ---------------------------------------------------------------------------


def test_ares_payload_scoring_variants() -> None:
    url = "https://ares.gov.cz/ekonomicke-subjekty-v-be/rest/ekonomicke-subjekty/26168685"

    mature = _evaluate_czech_ares_payload(
        "26168685",
        {
            "ico": "26168685",
            "obchodniJmeno": "Seznam.cz, a.s.",
            "sidlo": {"textovaAdresa": "Radlická 3294/10, Praha 5"},
            "datumVzniku": "2000-04-05",
            "czNace": ["63120"],
            "seznamRegistraci": {"stavZdrojeRos": "AKTIVNI", "stavZdrojeIr": "NEEXISTUJICI"},
        },
        url,
    )
    assert mature.result.score == 25
    assert mature.kill_switches == []
    assert mature.legal_name == "Seznam.cz, a.s."
    assert mature.official_address == "Radlická 3294/10, Praha 5"

    young_date = (datetime.now(timezone.utc) - timedelta(days=365)).date().isoformat()
    young = _evaluate_czech_ares_payload(
        "44444444",
        {
            "ico": "44444444",
            "obchodniJmeno": "YoungCo s.r.o.",
            "datumVzniku": young_date,
            "czNace": ["62010"],
            "seznamRegistraci": {"stavZdrojeRos": "AKTIVNI", "stavZdrojeIr": "NEEXISTUJICI"},
        },
        url,
    )
    assert young.result.score == 20
    assert "YOUNG_COMPANY" in young.red_flags

    recent_date = (datetime.now(timezone.utc) - timedelta(days=45)).date().isoformat()
    very_new = _evaluate_czech_ares_payload(
        "11111111",
        {
            "ico": "11111111",
            "obchodniJmeno": "FreshCo s.r.o.",
            "datumVzniku": recent_date,
            "czNace": ["78100"],
            "seznamRegistraci": {"stavZdrojeRos": "AKTIVNI", "stavZdrojeIr": "NEEXISTUJICI"},
        },
        url,
    )
    assert very_new.result.score == 15
    assert "VERY_NEW_COMPANY" in very_new.red_flags

    insolvent = _evaluate_czech_ares_payload(
        "22222222",
        {
            "ico": "22222222",
            "obchodniJmeno": "BankruptCo s.r.o.",
            "datumVzniku": "2015-01-01",
            "seznamRegistraci": {"stavZdrojeRos": "AKTIVNI", "stavZdrojeIr": "AKTIVNI"},
        },
        url,
    )
    assert insolvent.result.score == 0
    assert "COMPANY_LIQUIDATED_OR_INSOLVENT" in insolvent.kill_switches

    liquidated = _evaluate_czech_ares_payload(
        "33333333",
        {
            "ico": "33333333",
            "obchodniJmeno": "OldCo s.r.o. v likvidaci",
            "datumVzniku": "2010-01-01",
            "likvidace": True,
            "seznamRegistraci": {"stavZdrojeRos": "AKTIVNI", "stavZdrojeIr": "NEEXISTUJICI"},
        },
        url,
    )
    assert liquidated.result.score == 0
    assert "COMPANY_LIQUIDATED_OR_INSOLVENT" in liquidated.kill_switches


@pytest.mark.asyncio
async def test_ares_404_and_graceful_503() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/99999999"):
            return httpx.Response(404, json={"kod": "NENALEZENO"})
        return httpx.Response(503, text="Service Unavailable")

    cfg = _make_settings()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        out_404 = await check_gov_registry(_sample_request(reg_id="99999999"), client, cfg)
        assert out_404.result.status == ModuleStatus.SUCCESS
        assert out_404.result.score == 0
        assert "UNREGISTERED_ENTITY" in out_404.kill_switches

        out_503 = await check_gov_registry(_sample_request(reg_id="26168685"), client, cfg)
        assert out_503.result.status == ModuleStatus.UNAVAILABLE
        assert out_503.result.score == 0
        assert out_503.kill_switches == []


# ---------------------------------------------------------------------------
# 3. Module B (Domain, WHOIS & Site Crawler) Unit Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_domain_crawler_full_match_and_red_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    def html_handler(req: httpx.Request) -> httpx.Response:
        if req.url.path in ("", "/"):
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="""
                <html><body>
                  <a href="/kontakt">Kontakt</a>
                  <a href="https://www.linkedin.com/company/acme-czech-logistics">LinkedIn</a>
                </body></html>
                """,
            )
        if req.url.path == "/kontakt":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<p>IČO: 26168685 | hr@acmelogistics.cz | +420 777 123 456</p>",
            )
        return httpx.Response(404)

    monkeypatch.setattr(
        "app.modules.domain_crawler._whois_lookup_sync",
        lambda _domain: (datetime.now(timezone.utc) - timedelta(days=1825), None),
    )

    cfg = _make_settings()
    async with httpx.AsyncClient(transport=httpx.MockTransport(html_handler)) as client:
        out_clean = await check_domain_and_site(_sample_request(), client, cfg)
        assert out_clean.result.status == ModuleStatus.SUCCESS
        assert out_clean.result.score == 25
        assert out_clean.red_flags == []

        # Fresh domain + free email + unlisted contact
        monkeypatch.setattr(
            "app.modules.domain_crawler._whois_lookup_sync",
            lambda _domain: (datetime.now(timezone.utc) - timedelta(days=15), None),
        )
        out_shady = await check_domain_and_site(
            _sample_request(contact_email="jobs@gmail.com", contact_phone="+420 600 999 888"),
            client,
            cfg,
        )
        assert out_shady.result.score == 5
        assert "FREE_EMAIL_PROVIDER" in out_shady.red_flags
        assert "FRESH_DOMAIN_UNDER_60_DAYS" in out_shady.red_flags
        assert "CONTACT_NOT_LISTED_ON_OFFICIAL_SITE" in out_shady.red_flags

        # Different corporate domain -> DOMAIN_MISMATCH
        out_mismatch = await check_domain_and_site(
            _sample_request(contact_email="hr@lookalike-spoof.eu"),
            client,
            cfg,
        )
        assert "DOMAIN_MISMATCH" in out_mismatch.red_flags


# ---------------------------------------------------------------------------
# 4. Unified Module C (Claude OSINT Auditor) Unit Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_claude_osint_tool_config_no_sampling_params_and_url_extraction() -> None:
    """Verify web_search_20250305 config, NO temperature/top_p/top_k, score clamping & URL extraction."""
    payload = {
        "maps_reviews_score": 25,  # Clamped to 20
        "web_osint_score": 15,
        "vacancy_audit_score": 15,
        "maps_presence_found": True,
        "maps_address_or_website_match": True,
        "maps_rating": 4.5,
        "maps_reviews_count": 80,
        "maps_reviews_scam_keywords": False,
        "google_maps_url": "https://maps.google.com/?cid=123",
        "linkedin_company_url": "https://www.linkedin.com/company/acme-czech-logistics/",
        "negative_osint_reports_found": False,
        "virtual_office_detected": False,
        "requires_upfront_payment": False,
        "unrealistic_salary_or_promises": False,
        "artificial_urgency_detected": False,
        "shady_messenger_only_funnel": False,
        "reviews_indicate_fraud": False,
        "cited_urls": ["https://www.firmy.cz/detail/acme-czech"],
        "red_flags": [],
        "reasoning": "Verified physical headquarters, strong reviews, active LinkedIn, clean vacancy.",
    }
    fake_resp = _build_fake_claude_response(
        audit_payload=payload,
        search_urls=[
            "https://www.linkedin.com/company/acme-czech-logistics/",
            "https://rejstrik-firem.kurzy.cz/26168685/",
        ],
        citation_urls=["https://www.google.com/maps/place/Acme+Czech"],
    )
    fake_client = _FakeAnthropicClient([fake_resp])
    cfg = _make_settings(ANTHROPIC_API_KEY="test-key")

    out = await audit_with_claude_osint(
        request=_sample_request(),
        settings=cfg,
        legal_name="Acme Czech Logistics s.r.o.",
        official_address="Radlická 3294/10, 15000 Praha 5",
        anthropic_client=fake_client,  # type: ignore[arg-type]
    )

    assert len(fake_client.messages.calls) == 1
    call_kwargs = fake_client.messages.calls[0]

    # CRITICAL: temperature, top_p, and top_k must NEVER be passed
    assert "temperature" not in call_kwargs
    assert "top_p" not in call_kwargs
    assert "top_k" not in call_kwargs

    # Verify exact built-in server-side web_search_20250305 tool schema
    assert call_kwargs["tools"] == [
        {
            "type": "web_search_20250305",
            "name": "web_search",
            "max_uses": 5,
            "user_location": {"type": "approximate", "country": "CZ"},
        }
    ]

    assert out.result.status == ModuleStatus.SUCCESS
    assert out.result.score == 50  # Clamped 20 + 15 + 15
    assert out.result.max_score == 50
    assert out.kill_switches == []
    assert out.result.details["score_breakdown"] == {
        "maps_reviews": 20,
        "web_osint": 15,
        "vacancy_audit": 15,
    }
    # All cited & search URLs extracted into evidence_links
    assert "https://www.google.com/maps/place/Acme+Czech" in out.evidence_links
    assert "https://www.linkedin.com/company/acme-czech-logistics/" in out.evidence_links
    assert "https://maps.google.com/?cid=123" in out.evidence_links
    assert "https://www.firmy.cz/detail/acme-czech" in out.evidence_links
    assert "https://rejstrik-firem.kurzy.cz/26168685/" in out.evidence_links


@pytest.mark.asyncio
async def test_claude_osint_country_code_fallback_retry() -> None:
    """If Anthropic rejects user_location.country='CZ', module retries cleanly with Prague fallback."""
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp_400 = httpx.Response(
        400,
        request=req,
        json={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "tools.0.web_search_20250305: Country code CZ is not supported.",
            },
        },
    )
    bad_req_err = BadRequestError(
        message="Error code: 400 - Country code CZ is not supported.",
        response=resp_400,
        body=resp_400.json(),
    )
    ok_resp = _build_fake_claude_response(
        audit_payload={
            "maps_reviews_score": 15,
            "web_osint_score": 15,
            "vacancy_audit_score": 14,
            "maps_presence_found": True,
            "reasoning": "Clean after fallback retry.",
        },
        search_urls=["https://www.linkedin.com/company/acme/"],
    )
    fake_client = _FakeAnthropicClient([bad_req_err, ok_resp])
    cfg = _make_settings(ANTHROPIC_API_KEY="test-key")

    out = await audit_with_claude_osint(
        request=_sample_request(),
        settings=cfg,
        anthropic_client=fake_client,  # type: ignore[arg-type]
    )
    assert out.result.status == ModuleStatus.SUCCESS
    assert out.result.score == 44
    assert len(fake_client.messages.calls) == 2
    assert fake_client.messages.calls[0]["tools"][0]["user_location"] == {
        "type": "approximate",
        "country": "CZ",
    }
    assert fake_client.messages.calls[1]["tools"][0]["user_location"] == {
        "type": "approximate",
        "city": "Prague",
        "timezone": "Europe/Prague",
    }


@pytest.mark.asyncio
async def test_claude_osint_missing_api_key_degrades_gracefully() -> None:
    cfg = _make_settings(ANTHROPIC_API_KEY=None)
    out = await audit_with_claude_osint(_sample_request(), settings=cfg)
    assert out.result.status == ModuleStatus.UNAVAILABLE
    assert out.result.score == 0
    assert out.result.max_score == 50


# ---------------------------------------------------------------------------
# 5. Credit-Saving Early-Abort & Upfront Fee Scam Kill-Switch Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_early_abort_skips_claude_on_unregistered_and_liquidated(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ARES triggers UNREGISTERED_ENTITY or COMPANY_LIQUIDATED_OR_INSOLVENT, Claude MUST NOT be called."""
    claude_calls = 0

    async def spy_claude(*_args: Any, **_kwargs: Any) -> ClaudeOSINTCheckOutput:
        nonlocal claude_calls
        claude_calls += 1
        raise AssertionError("Claude OSINT must be skipped after Stage 1 kill-switch!")

    async def fake_domain(*_args: Any, **_kwargs: Any) -> DomainCheckOutput:
        return DomainCheckOutput(
            result=ModuleResult(status=ModuleStatus.SUCCESS, score=25, max_score=25, details={})
        )

    def ares_handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/99999999"):
            return httpx.Response(404, json={"kod": "NENALEZENO"})
        return httpx.Response(
            200,
            json={
                "ico": "33333333",
                "obchodniJmeno": "Dead Agency s.r.o. v likvidaci",
                "datumVzniku": "2018-01-01",
                "likvidace": True,
                "seznamRegistraci": {"stavZdrojeRos": "AKTIVNI", "stavZdrojeIr": "AKTIVNI"},
            },
        )

    monkeypatch.setattr("app.orchestrator.audit_with_claude_osint", spy_claude)
    monkeypatch.setattr("app.orchestrator.check_domain_and_site", fake_domain)

    cfg = _make_settings(ANTHROPIC_API_KEY="test-key")
    async with httpx.AsyncClient(transport=httpx.MockTransport(ares_handler)) as client:
        # Case 1: Unregistered entity (ARES 404)
        r_unreg = await run_verification_pipeline(
            _sample_request(reg_id="99999999"), db=db_session, settings=cfg, http_client=client
        )
        assert r_unreg.status == VerificationStatus.RED
        assert r_unreg.trust_score == 0
        assert "UNREGISTERED_ENTITY" in r_unreg.kill_switches_triggered
        assert r_unreg.modules.claude_osint.status == ModuleStatus.UNAVAILABLE
        assert "save api credits" in str(r_unreg.modules.claude_osint.details.get("skipped", "")).lower()
        assert claude_calls == 0

        # Case 2: Liquidated / insolvent company
        r_liq = await run_verification_pipeline(
            _sample_request(reg_id="33333333"), db=db_session, settings=cfg, http_client=client
        )
        assert r_liq.status == VerificationStatus.RED
        assert r_liq.trust_score == 0
        assert "COMPANY_LIQUIDATED_OR_INSOLVENT" in r_liq.kill_switches_triggered
        assert r_liq.modules.claude_osint.status == ModuleStatus.UNAVAILABLE
        assert claude_calls == 0


@pytest.mark.asyncio
async def test_upfront_fee_scam_kill_switch_overrides_perfect_ares_and_domain(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Even when ARES (25/25) and Domain (25/25) pass, UPFRONT_FEE_SCAM forces RED & score=0."""

    def mock_http(req: httpx.Request) -> httpx.Response:
        if "ares.gov.cz" in str(req.url):
            return httpx.Response(
                200,
                json={
                    "ico": "26168685",
                    "obchodniJmeno": "Impersonated Firm s.r.o.",
                    "sidlo": {"textovaAdresa": "Radlická 3294/10, 15000 Praha 5"},
                    "datumVzniku": "2014-02-01",
                    "czNace": ["78100"],
                    "seznamRegistraci": {"stavZdrojeRos": "AKTIVNI", "stavZdrojeIr": "NEEXISTUJICI"},
                },
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><body>IČO 26168685 hr@acmelogistics.cz +420 777 123 456</body></html>",
        )

    monkeypatch.setattr(
        "app.modules.domain_crawler._whois_lookup_sync",
        lambda _domain: (datetime.now(timezone.utc) - timedelta(days=1500), None),
    )

    scam_claude_resp = _build_fake_claude_response(
        audit_payload={
            "maps_reviews_score": 15,
            "web_osint_score": 10,
            "vacancy_audit_score": 0,
            "maps_presence_found": True,
            "maps_address_or_website_match": True,
            "requires_upfront_payment": True,
            "unrealistic_salary_or_promises": True,
            "artificial_urgency_detected": True,
            "shady_messenger_only_funnel": True,
            "reviews_indicate_fraud": False,
            "red_flags": ["UPFRONT_VISA_DEPOSIT"],
            "reasoning": "Demands 350 EUR prepayment via Telegram for work invitation.",
        },
        search_urls=["https://www.linkedin.com/company/impersonated-firm/"],
    )
    fake_anthropic = _FakeAnthropicClient([scam_claude_resp])
    cfg = _make_settings(ANTHROPIC_API_KEY="test-key")

    scam_req = _sample_request(
        applicant_type=ApplicantType.agency,
        vacancy_text="URGENT! 4500 EUR/month no Czech! Message Telegram @fastvisa and pay 350 EUR deposit today!",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(mock_http)) as http_client:
        resp = await run_verification_pipeline(
            scam_req,
            db=db_session,
            settings=cfg,
            http_client=http_client,
            anthropic_client=fake_anthropic,  # type: ignore[arg-type]
        )

    assert resp.status == VerificationStatus.RED
    assert resp.trust_score == 0
    assert resp.confidence_score == 1.0
    assert "UPFRONT_FEE_SCAM" in resp.kill_switches_triggered
    assert resp.modules.gov_registry.score == 25
    assert resp.modules.domain_site.score == 25
    assert resp.modules.claude_osint.score == 0


# ---------------------------------------------------------------------------
# 6. End-to-End FastAPI Routes, Caching, Blacklist & Timeout Fault Tolerance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_endpoints_green_verdict_cache_and_blacklist(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db

    ares_calls = 0
    claude_calls = 0

    async def fake_ares(request: VerificationRequest, client: httpx.AsyncClient, settings: Settings):
        nonlocal ares_calls
        ares_calls += 1
        return _evaluate_czech_ares_payload(
            request.reg_id,
            {
                "ico": request.reg_id,
                "obchodniJmeno": "Acme Czech Logistics s.r.o.",
                "sidlo": {"textovaAdresa": "Radlická 3294/10, 15000 Praha 5"},
                "datumVzniku": "2015-01-01",
                "czNace": ["52290"],
                "seznamRegistraci": {"stavZdrojeRos": "AKTIVNI", "stavZdrojeIr": "NEEXISTUJICI"},
            },
            f"https://ares.gov.cz/ekonomicke-subjekty-v-be/rest/ekonomicke-subjekty/{request.reg_id}",
        )

    async def fake_domain(request: VerificationRequest, client: httpx.AsyncClient, settings: Settings):
        return DomainCheckOutput(
            result=ModuleResult(status=ModuleStatus.SUCCESS, score=25, max_score=25, details={}),
            evidence_links=["https://www.acmelogistics.cz"],
        )

    async def fake_claude(*_args: Any, **_kwargs: Any) -> ClaudeOSINTCheckOutput:
        nonlocal claude_calls
        claude_calls += 1
        return ClaudeOSINTCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.SUCCESS,
                score=45,
                max_score=50,
                details={"score_breakdown": {"maps_reviews": 18, "web_osint": 13, "vacancy_audit": 14}},
            ),
            evidence_links=["https://www.linkedin.com/company/acme-czech-logistics/"],
        )

    monkeypatch.setattr("app.orchestrator.check_gov_registry", fake_ares)
    monkeypatch.setattr("app.orchestrator.check_domain_and_site", fake_domain)
    monkeypatch.setattr("app.orchestrator.audit_with_claude_osint", fake_claude)
    monkeypatch.setattr("app.main.get_settings", lambda: _make_settings(ANTHROPIC_API_KEY="test-key"))

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
            # 1. GET /health
            health = await ac.get("/health")
            assert health.status_code == 200
            assert health.json()["status"] == "ok"
            assert health.json()["database_backend"] == "sqlite"
            assert health.json()["configured_integrations"]["claude_osint"] is True

            payload = {
                "company_name": "Acme Czech Logistics s.r.o.",
                "country_code": "CZ",
                "reg_id": "26168685",
                "website_url": "https://www.acmelogistics.cz",
                "contact_email": "hr@acmelogistics.cz",
                "contact_phone": "+420777123456",
                "applicant_type": "direct_employer",
                "vacancy_text": "Logistics dispatcher in Prague, 45000 CZK/month.",
                "force_refresh": False,
            }

            # 2. First POST /api/v1/verify -> GREEN (25 + 25 + 45 = 95, conf=1.0)
            r1 = await ac.post("/api/v1/verify", json=payload)
            assert r1.status_code == 200
            body1 = r1.json()
            assert body1["cached"] is False
            assert body1["status"] == "GREEN"
            assert body1["trust_score"] == 95
            assert body1["confidence_score"] == 1.0
            assert set(body1["modules"].keys()) == {"gov_registry", "domain_site", "claude_osint"}
            assert ares_calls == 1 and claude_calls == 1

            # 3. Second POST /api/v1/verify -> cache hit (0 extra calls)
            r2 = await ac.post("/api/v1/verify", json=payload)
            assert r2.status_code == 200
            assert r2.json()["cached"] is True
            assert r2.json()["verification_id"] == body1["verification_id"]
            assert ares_calls == 1 and claude_calls == 1

            # 4. GET /api/v1/verify/{verification_id} + 404 for unknown ID
            fetched = await ac.get(f"/api/v1/verify/{body1['verification_id']}")
            assert fetched.status_code == 200
            assert fetched.json()["verification_id"] == body1["verification_id"]
            assert (await ac.get(f"/api/v1/verify/{uuid4()}")).status_code == 404

            # 5. force_refresh=True bypasses cache
            r3 = await ac.post("/api/v1/verify", json={**payload, "force_refresh": True})
            assert r3.status_code == 200
            assert r3.json()["cached"] is False
            assert ares_calls == 2 and claude_calls == 2

            # 6. POST /api/v1/blacklist -> immediate RED kill-switch, skips ARES & Claude!
            bl = await ac.post(
                "/api/v1/blacklist",
                json={
                    "entity_type": "phone",
                    "value": "777 123 456",
                    "reason": "Confirmed advance-fee scam phone",
                },
            )
            assert bl.status_code == 201
            assert bl.json()["value"] == "+420777123456"

            # Duplicate blacklist returns 409
            assert (
                await ac.post(
                    "/api/v1/blacklist",
                    json={"entity_type": "phone", "value": "+420777123456", "reason": "Dup"},
                )
            ).status_code == 409

            r4 = await ac.post("/api/v1/verify", json=payload)
            assert r4.status_code == 200
            body4 = r4.json()
            assert body4["status"] == "RED"
            assert body4["trust_score"] == 0
            assert "BLACKLISTED_ENTITY" in body4["kill_switches_triggered"]
            assert ares_calls == 2 and claude_calls == 2  # Zero external/LLM calls
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_module_timeout_graceful_degradation(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hanging module must degrade cleanly to UNAVAILABLE without crashing the pipeline."""

    async def slow_domain(*_args: Any, **_kwargs: Any):
        await asyncio.sleep(5.0)

    async def fast_ares(request: VerificationRequest, client: httpx.AsyncClient, settings: Settings):
        return _evaluate_czech_ares_payload(
            request.reg_id,
            {
                "ico": request.reg_id,
                "obchodniJmeno": "Timeout Test s.r.o.",
                "datumVzniku": "2018-01-01",
                "czNace": ["62010"],
                "seznamRegistraci": {"stavZdrojeRos": "AKTIVNI", "stavZdrojeIr": "NEEXISTUJICI"},
            },
            "https://ares.gov.cz/ekonomicke-subjekty-v-be/rest/ekonomicke-subjekty/26168685",
        )

    monkeypatch.setattr("app.orchestrator.check_gov_registry", fast_ares)
    monkeypatch.setattr("app.orchestrator.check_domain_and_site", slow_domain)

    cfg = _make_settings(MODULE_TIMEOUT_SECONDS=0.15, ANTHROPIC_API_KEY=None)
    resp = await run_verification_pipeline(_sample_request(), db=db_session, settings=cfg)
    assert resp.modules.gov_registry.status == ModuleStatus.SUCCESS
    assert resp.modules.gov_registry.score == 25
    assert resp.modules.domain_site.status == ModuleStatus.UNAVAILABLE
    assert resp.modules.domain_site.score == 0
    assert resp.modules.claude_osint.status == ModuleStatus.UNAVAILABLE
    assert resp.confidence_score == 0.25


# ---------------------------------------------------------------------------
# 7. Real Live Czech ARES Integration Test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_czech_ares_integration() -> None:
    """Live integration test against public Czech ARES REST API (Seznam.cz ICO 26168685)."""
    cfg = _make_settings(MODULE_TIMEOUT_SECONDS=10.0)
    async with httpx.AsyncClient() as client:
        seznam_req = _sample_request(
            company_name="Seznam.cz, a.s.",
            reg_id="261 68 685",
            website_url="https://o.seznam.cz",
            contact_email="info@firma.seznam.cz",
        )
        live_out = await check_gov_registry(seznam_req, client=client, settings=cfg)
        assert live_out.result.status == ModuleStatus.SUCCESS
        assert live_out.result.score == 25
        assert live_out.kill_switches == []
        assert live_out.registered is True
        assert live_out.legal_name is not None and "seznam" in live_out.legal_name.lower()
        assert live_out.official_address is not None and "praha" in live_out.official_address.lower()

        ghost_req = _sample_request(company_name="Ghost Scam s.r.o.", reg_id="99999999")
        ghost_out = await check_gov_registry(ghost_req, client=client, settings=cfg)
        assert ghost_out.result.status == ModuleStatus.SUCCESS
        assert ghost_out.result.score == 0
        assert "UNREGISTERED_ENTITY" in ghost_out.kill_switches
        assert ghost_out.registered is False


# ---------------------------------------------------------------------------
# 8. Stage 2 Tests: Poland MF NIP, ARES Auto-Resolve, Noise Filtering, Intake & New Routes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poland_mf_registry_and_cz_name_autoresolve() -> None:
    # Poland MF evaluation unit checks
    url = "https://wl-api.mf.gov.pl/api/search/nip/5252674798?date=2026-11-21"
    active_pl = _evaluate_poland_mf_payload(
        "5252674798",
        {
            "name": "ALLEGRO SP. Z O.O.",
            "nip": "5252674798",
            "statusVat": "Czynny",
            "krs": "0000635012",
            "regon": "365331553",
            "residenceAddress": "WIERZBIĘCICE 1B, 61-569 POZNAŃ",
            "registrationLegalDate": "2016-09-12",
            "removalDate": None,
        },
        url,
    )
    assert active_pl.result.score == 25
    assert active_pl.kill_switches == []
    assert active_pl.legal_name == "ALLEGRO SP. Z O.O."

    removed_pl = _evaluate_poland_mf_payload(
        "1234567890",
        {
            "name": "DELETED SP. Z O.O.",
            "nip": "1234567890",
            "statusVat": "Niezarejestrowany",
            "registrationLegalDate": "2019-01-01",
            "removalDate": "2024-05-01",
            "restorationDate": None,
        },
        url,
    )
    assert removed_pl.result.score == 0
    assert "COMPANY_LIQUIDATED_OR_INSOLVENT" in removed_pl.kill_switches

    # Mocked HTTP for PL UNKNOWN, PL 404, and CZ ARES name auto-resolve
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/vyhledat"):
            body = json.loads(req.content.decode("utf-8"))
            if body.get("obchodniJmeno") == "Manpower":
                return httpx.Response(
                    200,
                    json={
                        "ekonomickeSubjekty": [
                            {
                                "ico": "05946344",
                                "obchodniJmeno": "NARA manpower s.r.o.",
                                "seznamRegistraci": {"stavZdrojeRos": "AKTIVNI"},
                            }
                        ]
                    },
                )
            return httpx.Response(
                200,
                json={
                    "ekonomickeSubjekty": [
                        {
                            "ico": "41194659",
                            "obchodniJmeno": "ManpowerGroup s.r.o.",
                            "seznamRegistraci": {"stavZdrojeRos": "AKTIVNI"},
                        }
                    ]
                },
            )
        if "wl-api.mf.gov.pl" in str(req.url):
            return httpx.Response(200, json={"result": {"subject": None}})
        return httpx.Response(404)

    cfg = _make_settings()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        # Must NOT falsely resolve placeholder names ("UNKNOWN", "Anonymous Employer") or middle-word matches
        assert await search_czech_ares_by_name("UNKNOWN", client, cfg) is None
        assert await search_czech_ares_by_name("Anonymous Employer", client, cfg) is None
        assert await search_czech_ares_by_name("Manpower", client, cfg) is None
        resolved = await search_czech_ares_by_name("ManpowerGroup Česká republika", client, cfg)
        assert resolved == "41194659"

        pl_unknown = await check_gov_registry(
            _sample_request(country_code="PL", reg_id="UNKNOWN"), client, cfg
        )
        assert pl_unknown.result.status == ModuleStatus.UNAVAILABLE
        assert pl_unknown.kill_switches == []

        pl_missing = await check_gov_registry(
            _sample_request(country_code="PL", reg_id="PL 525-267-47-98"), client, cfg
        )
        assert pl_missing.result.status == ModuleStatus.SUCCESS
        assert "UNREGISTERED_ENTITY" in pl_missing.kill_switches


@pytest.mark.asyncio
async def test_domain_crawler_jsonld_sentry_filter_and_rdap_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if "rdap.org" in str(req.url):
            return httpx.Response(
                200,
                json={"events": [{"eventAction": "registration", "eventDate": "2018-04-10T00:00:00Z"}]},
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="""
            <html><head>
            <script type="application/ld+json">
            {"@type": "Organization", "email": "hr@acmelogistics.cz", "telephone": "+420 777 123 456",
             "sameAs": ["https://www.instagram.com/acme_cz/"]}
            </script></head>
            <body>abc123@sentry.wixpress.com | IČO 26168685</body></html>
            """,
        )

    # Simulate GDPR-redacted WHOIS -> falls back to RDAP
    monkeypatch.setattr(
        "app.modules.domain_crawler._whois_lookup_sync",
        lambda _domain: (None, "WHOIS creation_date unavailable or redacted"),
    )

    cfg = _make_settings()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        out = await check_domain_and_site(_sample_request(), client, cfg)
        assert out.result.score == 25
        assert out.result.details["whois"]["source"] == "rdap"
        assert "https://www.instagram.com/acme_cz/" in out.social_urls
        assert all("wixpress" not in e for e in out.extracted_emails)


@pytest.mark.asyncio
async def test_claude_evidence_noise_url_filtering() -> None:
    fake_resp = _build_fake_claude_response(
        audit_payload={
            "maps_reviews_score": 20,
            "web_osint_score": 15,
            "vacancy_audit_score": 15,
            "maps_presence_found": True,
            "linkedin_company_url": "https://www.linkedin.com/company/acme/",
            "reasoning": "Verified.",
        },
        search_urls=[
            "https://en.wikipedia.org/wiki/Telemarketing_fraud",
            "https://www.attorneygeneral.gov/submit-a-complaint/scams/",
            "https://www.bbb.org/scamtracker/lookupscam",
            "https://www.firmy.cz/detail/acme",
        ],
    )
    fake_client = _FakeAnthropicClient([fake_resp])
    out = await audit_with_claude_osint(
        _sample_request(),
        settings=_make_settings(ANTHROPIC_API_KEY="k"),
        anthropic_client=fake_client,  # type: ignore[arg-type]
    )
    assert "https://www.firmy.cz/detail/acme" in out.evidence_links
    assert "https://www.linkedin.com/company/acme/" in out.evidence_links
    assert not any("wikipedia.org" in u for u in out.evidence_links)
    assert not any("attorneygeneral.gov" in u for u in out.evidence_links)
    assert not any("bbb.org/scamtracker/lookupscam" in u for u in out.evidence_links)


@pytest.mark.asyncio
async def test_stage2_multimodal_intake_docx_image_ssrf_and_api_routes(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    import io
    import zipfile

    # Build a minimal valid .docx in memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "word/document.xml",
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            "<w:body><w:p><w:r><w:t>ТОО Test Agency БИН 220940008449 Предоплата 500 000 тенге "
            "scam@gmail.com +77089046070</w:t></w:r></w:p></w:body></w:document>",
        )
    docx_bytes = buf.getvalue()
    assert "500 000 тенге" in extract_text_from_docx(docx_bytes)

    # 1. Offline regex fallback (no Anthropic key)
    res_offline = await extract_verification_request(
        settings=_make_settings(ANTHROPIC_API_KEY=None),
        attachments=[UploadedAttachment("contract.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", docx_bytes)],
    )
    assert res_offline.extraction_method == "regex_fallback"
    assert res_offline.request.country_code == "KZ"
    assert res_offline.request.reg_id == "220940008449"
    assert res_offline.request.contact_email == "scam@gmail.com"

    # 2. SSRF protection blocks localhost
    with pytest.raises(ValueError, match="public http"):
        await extract_verification_request(
            settings=_make_settings(),
            url="http://127.0.0.1:8000/health",
        )

    # 3. Claude multimodal extraction with image + docx (and NO temperature/top_p/top_k!)
    fake_intake_msg = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="text",
                text=json.dumps(
                    {
                        "company_name": "ТОО Test Agency",
                        "country_code": "KZ",
                        "reg_id": "220940008449",
                        "website_url": None,
                        "contact_email": "scam@gmail.com",
                        "contact_phone": "+77089046070",
                        "applicant_type": "agency",
                        "vacancy_text": "Contract demands 500,000 KZT upfront fee upon signing.",
                    }
                ),
            )
        ]
    )
    fake_ai = _FakeAnthropicClient([fake_intake_msg])
    # 1x1 transparent PNG bytes
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00"
        b"\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    res_claude = await extract_verification_request(
        settings=_make_settings(ANTHROPIC_API_KEY="k"),
        attachments=[
            UploadedAttachment("contract.docx", "application/octet-stream", docx_bytes),
            UploadedAttachment("ad.png", "image/png", png_bytes),
        ],
        anthropic_client=fake_ai,  # type: ignore[arg-type]
    )
    assert res_claude.extraction_method == "claude_multimodal"
    assert "docx:contract.docx" in res_claude.sources
    assert "image:ad.png" in res_claude.sources
    assert "temperature" not in fake_ai.messages.calls[0]
    assert "top_p" not in fake_ai.messages.calls[0]
    assert "top_k" not in fake_ai.messages.calls[0]
    assert any(b.get("type") == "image" for b in fake_ai.messages.calls[0]["messages"][0]["content"])

    # 4. Stage 2 FastAPI endpoints: GET /, POST /api/v1/verify/intake, GET /api/v1/verifications, GET & DELETE /api/v1/blacklist
    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    monkeypatch.setattr("app.main.get_settings", lambda: _make_settings(ANTHROPIC_API_KEY=None))
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
            # Built-in Dashboard HTML
            ui = await ac.get("/")
            assert ui.status_code == 200
            assert "Employer Anti-Fraud OSINT Engine" in ui.text

            # Intake endpoint with uploaded .docx
            intake_resp = await ac.post(
                "/api/v1/verify/intake",
                files={"files": ("contract.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
                data={"force_refresh": "true"},
            )
            assert intake_resp.status_code == 200
            intake_body = intake_resp.json()
            assert intake_body["extracted_input"]["reg_id"] == "220940008449"
            assert intake_body["verification"]["status"] == "RED"

            # History endpoint
            hist = await ac.get("/api/v1/verifications?status=RED")
            assert hist.status_code == 200
            assert len(hist.json()) == 1
            assert hist.json()[0]["verification_id"] == intake_body["verification"]["verification_id"]

            # Blacklist GET + DELETE
            created_bl = await ac.post(
                "/api/v1/blacklist",
                json={"entity_type": "email", "value": "scam@gmail.com", "reason": "Test ban"},
            )
            assert created_bl.status_code == 201
            bl_id = created_bl.json()["id"]

            bl_list = await ac.get("/api/v1/blacklist")
            assert bl_list.status_code == 200
            assert len(bl_list.json()) == 1

            del_res = await ac.delete(f"/api/v1/blacklist/{bl_id}")
            assert del_res.status_code == 204
            assert (await ac.delete(f"/api/v1/blacklist/{bl_id}")).status_code == 404
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_conditional_contact_deep_check_trigger_logic(
    db_session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deep recruiter/phone check activates ONLY when phone/person wasn't already verified on site."""
    captured_contexts: list[dict[str, Any]] = []

    async def fake_ares(*_args: Any, **_kwargs: Any):
        from app.modules.ares import RegistryCheckOutput

        return RegistryCheckOutput(
            result=ModuleResult(status=ModuleStatus.UNAVAILABLE, score=0, max_score=25, details={})
        )

    async def spy_claude(
        request: VerificationRequest,
        settings: Settings,
        legal_name: Any = None,
        official_address: Any = None,
        deterministic_context: dict[str, Any] | None = None,
        anthropic_client: Any = None,
    ) -> ClaudeOSINTCheckOutput:
        captured_contexts.append(dict(deterministic_context or {}))
        return ClaudeOSINTCheckOutput(
            result=ModuleResult(status=ModuleStatus.SUCCESS, score=40, max_score=50, details={})
        )

    monkeypatch.setattr("app.orchestrator.check_gov_registry", fake_ares)
    monkeypatch.setattr("app.orchestrator.audit_with_claude_osint", spy_claude)

    # Case 1: Phone ALREADY verified on official site -> needs_contact_deep_check MUST be False (0 extra tokens)
    async def domain_verified(*_args: Any, **_kwargs: Any) -> DomainCheckOutput:
        return DomainCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.SUCCESS,
                score=25,
                max_score=25,
                details={"site_crawl": {"phone_found_on_site": True, "email_found_on_site": False}},
            )
        )

    monkeypatch.setattr("app.orchestrator.check_domain_and_site", domain_verified)
    await run_verification_pipeline(
        _sample_request(contact_person="Lav", contact_phone="+420703865303", force_refresh=True),
        db=db_session,
        settings=_make_settings(ANTHROPIC_API_KEY="k"),
    )
    assert captured_contexts[-1]["needs_contact_deep_check"] is False
    assert captured_contexts[-1]["phone_corporate_prefix_hint"] is None

    # Case 2: Phone NOT verified on official site -> needs_contact_deep_check MUST be True + 6-digit prefix
    async def domain_unverified(*_args: Any, **_kwargs: Any) -> DomainCheckOutput:
        return DomainCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.SUCCESS,
                score=0,
                max_score=25,
                details={"site_crawl": {"phone_found_on_site": False, "email_found_on_site": False}},
            )
        )

    monkeypatch.setattr("app.orchestrator.check_domain_and_site", domain_unverified)
    await run_verification_pipeline(
        _sample_request(contact_person="Lav", contact_phone="+420703865303", force_refresh=True),
        db=db_session,
        settings=_make_settings(ANTHROPIC_API_KEY="k"),
    )
    assert captured_contexts[-1]["needs_contact_deep_check"] is True
    assert captured_contexts[-1]["phone_corporate_prefix_hint"] == "703 865"


@pytest.mark.asyncio
async def test_eures_native_api_intake_and_empty_spa_guard() -> None:
    """Verify official EURES SPA URLs resolve via public JSON API and blank SPA shells raise 400."""

    def handler(req: httpx.Request) -> httpx.Response:
        if "europa.eu/eures/api/jv-searchengine/public/jv/id/TESTID123" in str(req.url):
            return httpx.Response(
                200,
                json={
                    "id": "TESTID123",
                    "reference": "10000-1207750458-S",
                    "source": "DE001",
                    "preferredLanguage": "de",
                    "jvProfiles": {
                        "de": {
                            "title": "Koch / Jungkoch (m/w/d)",
                            "euresFlag": True,
                            "description": "<p>Familiär geführtes Hotel in Bad Liebenstein.</p>",
                            "locations": [{"countryCode": "de", "cityName": "Bad Liebenstein", "postalCode": "36448"}],
                            "employer": {"name": "LANO Management GmbH", "website": None, "sectorCodes": ["i55.1.0"]},
                            "personContacts": [
                                {
                                    "givenName": "Heike",
                                    "familyName": "Tempel",
                                    "communications": {
                                        "emails": [{"uri": "buchhaltung@kulturhotel-kaiserhof.de"}],
                                        "telephoneNumbers": [{"countryDialing": "+49", "areaDialing": "36961", "dialNumber": "73370"}],
                                        "addresses": [{"addressLines": ["Esplanade 9, 36448 Bad Liebenstein"]}],
                                    },
                                }
                            ],
                        }
                    },
                },
            )
        if "spa-only.example.org" in str(req.url):
            return httpx.Response(200, text="<html><body><div id='root'>Loading...</div></body></html>")
        return httpx.Response(404)

    cfg = _make_settings(ANTHROPIC_API_KEY=None)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        # 1. EURES native JSON API extraction (0 Claude tokens!)
        res = await extract_verification_request(
            settings=cfg,
            url="https://europa.eu/eures/portal/jv-se/jv-details/TESTID123?jvDisplayLanguage=de&lang=en",
            http_client=client,
        )
        assert res.extraction_method == "eures_official_api"
        assert res.request.company_name == "LANO Management GmbH"
        assert res.request.country_code == "DE"
        assert res.request.website_url == "https://kulturhotel-kaiserhof.de"
        assert res.request.contact_person == "Heike Tempel"
        assert res.request.contact_email == "buchhaltung@kulturhotel-kaiserhof.de"
        assert res.request.contact_phone == "+493696173370"
        assert res.request.applicant_type == ApplicantType.direct_employer

        # Verify EU_EURES_PES attestation in Module A
        reg_out = await check_gov_registry(res.request, client=client, settings=cfg)
        assert reg_out.result.status == ModuleStatus.SUCCESS
        assert reg_out.result.score == 20
        assert reg_out.result.details["registry"] == "EU_EURES_PES"
        assert reg_out.result.details["pes_source"] == "DE001"

        # 2. Blank SPA shell raises clear ValueError instead of verifying "=== WEB PAGE ==="
        with pytest.raises(ValueError, match="JavaScript"):
            await extract_verification_request(
                settings=cfg,
                url="https://spa-only.example.org/jobs/123",
                http_client=client,
            )
