"""Verification pipeline orchestrator (Anthropic-only two-stage credit-saving pipeline).

Stage 0 (Pre-flight):
  1. SQL Blacklist check -> immediate RED kill-switch (skips all modules & LLM).
  2. SQL Cache check (14-day TTL, unless force_refresh=True).

Stage 1 (Parallel Free Deterministic Modules — Modules A & B):
  - Module A: Czech ARES registry (max 25 pts)
  - Module B: Domain, WHOIS & lightweight async site crawler (max 25 pts)
  - EARLY-ABORT GATE: If ANY Hard Kill-Switch is triggered in Stage 1 (e.g., UNREGISTERED_ENTITY
    or COMPANY_LIQUIDATED_OR_INSOLVENT), immediately abort with RED (trust_score = 0) and SKIP
    the Anthropic Claude API call to save API credits!

Stage 2 (Unified Claude OSINT & Vacancy Auditor — max 50 pts):
  - Uses official Anthropic SDK + server-side `web_search_20250305` for Google Maps/reviews (20),
    LinkedIn & scam OSINT (15), and vacancy audit (15, with UPFRONT_FEE_SCAM kill-switch).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, TypeVar
from uuid import UUID

import httpx
from anthropic import AsyncAnthropic
from sqlalchemy import and_, desc, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.models import Blacklist, Company, VerificationLog
from app.modules.ares import MAX_SCORE as ARES_MAX_SCORE
from app.modules.ares import RegistryCheckOutput, check_gov_registry
from app.modules.claude_osint_auditor import MAX_SCORE as CLAUDE_MAX_SCORE
from app.modules.claude_osint_auditor import ClaudeOSINTCheckOutput, audit_with_claude_osint
from app.modules.domain_crawler import MAX_SCORE as DOMAIN_MAX_SCORE
from app.modules.domain_crawler import DomainCheckOutput, check_domain_and_site
from app.schemas import (
    BlacklistEntityType,
    ModuleResult,
    ModulesOutput,
    ModuleStatus,
    VerificationRequest,
    VerificationResponse,
    VerificationStatus,
    compute_input_hash,
    extract_root_domain,
    normalize_email,
    normalize_phone_e164,
    normalize_reg_id,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _ensure_utc(dt: datetime) -> datetime:
    """Normalize naive datetimes (e.g., returned by SQLite) to UTC-aware datetimes."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def compute_verdict(
    modules: ModulesOutput,
    kill_switches: list[str],
) -> tuple[VerificationStatus, int, float]:
    """Compute final (status, trust_score, confidence_score) per Section 6 strict rules."""
    module_list = (
        modules.gov_registry,
        modules.domain_site,
        modules.claude_osint,
    )
    available_max = sum(
        m.max_score for m in module_list if m.status == ModuleStatus.SUCCESS
    )
    confidence_score = round(available_max / 100.0, 2)
    raw_trust_score = sum(
        m.score for m in module_list if m.status == ModuleStatus.SUCCESS
    )
    trust_score = max(0, min(100, raw_trust_score))

    # Strict decision tree order:
    # Rule 1: Any kill-switch -> RED, trust_score = 0
    if len(kill_switches) > 0:
        return VerificationStatus.RED, 0, confidence_score
    # Rule 2: trust_score >= 80 AND confidence_score >= 0.70 -> GREEN
    if trust_score >= 80 and confidence_score >= 0.70:
        return VerificationStatus.GREEN, trust_score, confidence_score
    # Rule 3: trust_score >= 50 (including >= 80 with confidence < 0.70) -> YELLOW
    if trust_score >= 50:
        return VerificationStatus.YELLOW, trust_score, confidence_score
    # Rule 4: trust_score < 50 -> RED
    return VerificationStatus.RED, trust_score, confidence_score


def build_summary(
    company_name: str,
    status: VerificationStatus,
    trust_score: int,
    confidence_score: float,
    kill_switches: list[str],
    red_flags: list[str],
    modules: ModulesOutput,
) -> str:
    """Generate a concise, deterministic, human-readable verification verdict summary."""
    if kill_switches:
        ks_text = ", ".join(kill_switches)
        skipped_note = (
            " Claude OSINT audit skipped to save API credits."
            if modules.claude_osint.details.get("skipped")
            else ""
        )
        return (
            f"REJECTED ({status.value}): Hard kill-switch triggered for '{company_name}' "
            f"[{ks_text}]. Trust score forced to 0 (confidence {confidence_score:.2f})."
            f"{skipped_note}"
        )

    unavailable = [
        name
        for name, mod in (
            ("gov_registry", modules.gov_registry),
            ("domain_site", modules.domain_site),
            ("claude_osint", modules.claude_osint),
        )
        if mod.status != ModuleStatus.SUCCESS
    ]

    if status == VerificationStatus.GREEN:
        verdict_phrase = f"VERIFIED ({status.value}): '{company_name}' passed verification"
    elif status == VerificationStatus.YELLOW:
        verdict_phrase = (
            f"MANUAL REVIEW REQUIRED ({status.value}): '{company_name}' needs review"
        )
    else:
        verdict_phrase = f"HIGH RISK ({status.value}): '{company_name}' failed verification"

    parts = [
        f"{verdict_phrase} with trust score {trust_score}/100 (confidence {confidence_score:.2f})."
    ]
    if red_flags:
        parts.append(f"Flags: {', '.join(red_flags[:6])}.")
    if unavailable:
        parts.append(f"Unavailable/skipped modules: {', '.join(unavailable)}.")
    return " ".join(parts)


def log_to_response(log: VerificationLog, cached: bool) -> VerificationResponse:
    """Hydrate a strict VerificationResponse from a stored VerificationLog ORM row."""
    modules_data = log.module_results or {}
    modules = ModulesOutput.model_validate(modules_data)
    return VerificationResponse(
        verification_id=log.id,
        cached=cached,
        status=log.status,
        trust_score=log.trust_score,
        confidence_score=log.confidence_score,
        kill_switches_triggered=list(log.kill_switches or []),
        red_flags=list(log.red_flags or []),
        evidence_links=list(log.evidence_links or []),
        summary=log.summary or "",
        modules=modules,
    )


async def _run_module_with_timeout(
    module_name: str,
    coro: Awaitable[T],
    timeout_seconds: float,
    fallback_factory: Callable[[str], T],
) -> T:
    """Execute a module coroutine under a strict timeout, degrading gracefully on any error."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        msg = f"Module '{module_name}' timed out after {timeout_seconds:.1f}s"
        logger.warning(msg)
        return fallback_factory(msg)
    except Exception as exc:
        msg = f"Module '{module_name}' raised unexpected error: {type(exc).__name__}: {exc}"
        logger.exception(msg)
        return fallback_factory(msg)


async def _check_blacklist(
    db: AsyncSession, request: VerificationRequest
) -> list[Blacklist]:
    """Step 0.1: Fast SQL lookup across normalized reg_id, website domain, email, and E.164 phone."""
    candidates: list[tuple[BlacklistEntityType, str]] = []

    reg_id = normalize_reg_id(request.reg_id, request.country_code)
    if reg_id:
        candidates.append((BlacklistEntityType.reg_id, reg_id))

    site_domain = extract_root_domain(request.website_url)
    if site_domain:
        candidates.append((BlacklistEntityType.domain, site_domain))

    email = normalize_email(request.contact_email)
    if email:
        candidates.append((BlacklistEntityType.email, email))

    phone = normalize_phone_e164(request.contact_phone, request.country_code)
    if phone:
        candidates.append((BlacklistEntityType.phone, phone))

    if not candidates:
        return []

    conditions = [
        and_(Blacklist.entity_type == entity_type, Blacklist.value == value)
        for entity_type, value in candidates
    ]
    stmt = select(Blacklist).where(or_(*conditions))
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def _find_valid_cache(
    db: AsyncSession, input_hash: str
) -> Optional[VerificationLog]:
    """Step 0.2: Return the latest non-expired VerificationLog matching input_hash."""
    stmt = (
        select(VerificationLog)
        .where(VerificationLog.input_hash == input_hash)
        .order_by(desc(VerificationLog.created_at))
        .limit(1)
    )
    result = await db.execute(stmt)
    log = result.scalar_one_or_none()
    if log is None:
        return None
    if _ensure_utc(log.expires_at) > datetime.now(timezone.utc):
        return log
    return None


async def _upsert_company(
    db: AsyncSession,
    request: VerificationRequest,
    ares_out: RegistryCheckOutput,
) -> Optional[Company]:
    """Persist or update company record when confirmed registered in government registry."""
    if not ares_out.registered:
        return None

    effective_reg_id = ares_out.resolved_reg_id or request.reg_id
    stmt = select(Company).where(
        and_(
            Company.country_code == request.country_code,
            Company.reg_id == effective_reg_id,
        )
    )
    result = await db.execute(stmt)
    company = result.scalar_one_or_none()
    website_domain = extract_root_domain(request.website_url) or None
    now = datetime.now(timezone.utc)

    if company is None:
        company = Company(
            country_code=request.country_code,
            reg_id=effective_reg_id,
            legal_name=ares_out.legal_name or request.company_name,
            website_domain=website_domain,
            ares_data=ares_out.raw_data,
            created_at=now,
            updated_at=now,
        )
        try:
            async with db.begin_nested():
                db.add(company)
                await db.flush()
            return company
        except IntegrityError:
            result = await db.execute(stmt)
            company = result.scalar_one()

    company.legal_name = ares_out.legal_name or company.legal_name or request.company_name
    if website_domain:
        company.website_domain = website_domain
    if ares_out.raw_data is not None:
        company.ares_data = ares_out.raw_data
    company.updated_at = now
    await db.flush()
    return company


async def _persist_verification_log(
    db: AsyncSession,
    company_id: Optional[UUID],
    input_hash: str,
    status: VerificationStatus,
    trust_score: int,
    confidence_score: float,
    kill_switches: list[str],
    red_flags: list[str],
    evidence_links: list[str],
    summary: str,
    modules: ModulesOutput,
    cache_ttl_days: int,
) -> VerificationLog:
    """Persist a VerificationLog row and commit the transaction."""
    now = datetime.now(timezone.utc)
    log = VerificationLog(
        company_id=company_id,
        input_hash=input_hash,
        status=status,
        trust_score=trust_score,
        confidence_score=confidence_score,
        kill_switches=kill_switches,
        red_flags=red_flags,
        evidence_links=evidence_links,
        summary=summary,
        module_results=modules.model_dump(mode="json"),
        created_at=now,
        expires_at=now + timedelta(days=cache_ttl_days),
    )
    db.add(log)
    await db.commit()
    await db.refresh(log)
    return log


async def run_verification_pipeline(
    request: VerificationRequest,
    db: AsyncSession,
    settings: Optional[Settings] = None,
    http_client: Optional[httpx.AsyncClient] = None,
    anthropic_client: Optional[AsyncAnthropic] = None,
) -> VerificationResponse:
    """Execute the full Anthropic-only two-stage verification pipeline."""
    cfg = settings or get_settings()
    input_hash = compute_input_hash(request)

    # ------------------------------------------------------------------
    # Step 0.1: Pre-flight Blacklist Check (aborts immediately; 0 LLM cost)
    # ------------------------------------------------------------------
    blacklist_hits = await _check_blacklist(db, request)
    if blacklist_hits:
        matched_info = [
            {"entity_type": hit.entity_type.value, "value": hit.value, "reason": hit.reason}
            for hit in blacklist_hits
        ]
        skip_detail: dict[str, Any] = {
            "skipped": "Terminated at pre-flight blacklist check to save API credits",
            "blacklist_matches": matched_info,
        }
        modules = ModulesOutput(
            gov_registry=ModuleResult(
                status=ModuleStatus.UNAVAILABLE,
                score=0,
                max_score=ARES_MAX_SCORE,
                details=skip_detail,
            ),
            domain_site=ModuleResult(
                status=ModuleStatus.UNAVAILABLE,
                score=0,
                max_score=DOMAIN_MAX_SCORE,
                details=skip_detail,
            ),
            claude_osint=ModuleResult(
                status=ModuleStatus.UNAVAILABLE,
                score=0,
                max_score=CLAUDE_MAX_SCORE,
                details=skip_detail,
            ),
        )
        kill_switches = ["BLACKLISTED_ENTITY"]
        red_flags = [f"BLACKLIST_MATCH:{hit.entity_type.value}" for hit in blacklist_hits]
        reasons_joined = "; ".join(
            f"{hit.entity_type.value}={hit.value} ({hit.reason})" for hit in blacklist_hits
        )
        summary = (
            f"REJECTED (RED): Pre-flight blacklist kill-switch triggered for "
            f"'{request.company_name}': {reasons_joined}. Claude OSINT skipped to save API credits."
        )
        log = await _persist_verification_log(
            db=db,
            company_id=None,
            input_hash=input_hash,
            status=VerificationStatus.RED,
            trust_score=0,
            confidence_score=1.0,
            kill_switches=kill_switches,
            red_flags=red_flags,
            evidence_links=[],
            summary=summary,
            modules=modules,
            cache_ttl_days=cfg.CACHE_TTL_DAYS,
        )
        logger.warning(
            "Verification blocked by blacklist verification_id=%s reg_id=%s matches=%d",
            log.id,
            request.reg_id,
            len(blacklist_hits),
        )
        return log_to_response(log, cached=False)

    # ------------------------------------------------------------------
    # Step 0.2: Pre-flight Cache Check (14-day TTL)
    # ------------------------------------------------------------------
    if not request.force_refresh:
        cached_log = await _find_valid_cache(db, input_hash)
        if cached_log is not None:
            logger.info(
                "Cache hit verification_id=%s input_hash=%s status=%s",
                cached_log.id,
                input_hash[:12],
                cached_log.status.value,
            )
            return log_to_response(cached_log, cached=True)

    # ------------------------------------------------------------------
    # Stage 1: Free Deterministic Modules A (ARES) & B (Domain/WHOIS/Site) FIRST
    # ------------------------------------------------------------------
    det_timeout = cfg.MODULE_TIMEOUT_SECONDS
    owns_client = http_client is None
    client = http_client or httpx.AsyncClient(
        timeout=httpx.Timeout(det_timeout),
        limits=httpx.Limits(max_connections=30, max_keepalive_connections=10),
    )

    try:
        ares_out, domain_out = await asyncio.gather(
            _run_module_with_timeout(
                "gov_registry",
                check_gov_registry(request, client=client, settings=cfg),
                timeout_seconds=det_timeout,
                fallback_factory=lambda err: RegistryCheckOutput(
                    result=ModuleResult(
                        status=ModuleStatus.UNAVAILABLE,
                        score=0,
                        max_score=ARES_MAX_SCORE,
                        details={"error": err},
                    )
                ),
            ),
            _run_module_with_timeout(
                "domain_site",
                check_domain_and_site(request, client=client, settings=cfg),
                timeout_seconds=det_timeout,
                fallback_factory=lambda err: DomainCheckOutput(
                    result=ModuleResult(
                        status=ModuleStatus.UNAVAILABLE,
                        score=0,
                        max_score=DOMAIN_MAX_SCORE,
                        details={"error": err},
                    )
                ),
            ),
        )
    finally:
        if owns_client:
            await client.aclose()

    # ------------------------------------------------------------------
    # Early-Abort Credit-Saving Kill-Switch Gate
    # ------------------------------------------------------------------
    stage1_kill_switches = list(
        dict.fromkeys(ares_out.kill_switches + domain_out.kill_switches)
    )
    if stage1_kill_switches:
        logger.warning(
            "Stage 1 kill-switch triggered reg_id=%s kill_switches=%s -> aborting RED & skipping Claude",
            request.reg_id,
            stage1_kill_switches,
        )
        claude_out = ClaudeOSINTCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.UNAVAILABLE,
                score=0,
                max_score=CLAUDE_MAX_SCORE,
                details={
                    "skipped": "Aborted after deterministic kill-switch to save API credits",
                    "triggered_kill_switches": stage1_kill_switches,
                },
            )
        )
    else:
        # --------------------------------------------------------------
        # Stage 2: Unified Claude OSINT + Maps/Reviews + Vacancy Audit (max 50 pts)
        # --------------------------------------------------------------
        site_crawl_details = domain_out.result.details.get("site_crawl") or {}
        phone_verified_on_site = bool(site_crawl_details.get("phone_found_on_site"))
        email_verified_on_site = bool(site_crawl_details.get("email_found_on_site"))
        needs_contact_deep_check = bool(
            (request.contact_phone or request.contact_person)
            and not phone_verified_on_site
            and not email_verified_on_site
        )
        phone_prefix_hint: Optional[str] = None
        if needs_contact_deep_check and request.contact_phone:
            digits_only = "".join(ch for ch in request.contact_phone if ch.isdigit())
            # Extract local 6-digit corporate block prefix (e.g. +420703865303 -> "703 865")
            local_9 = digits_only[-9:] if len(digits_only) >= 9 else digits_only
            if len(local_9) >= 6:
                phone_prefix_hint = f"{local_9[:3]} {local_9[3:6]}"

        deterministic_context = {
            "ares_score": ares_out.result.score,
            "ares_status": ares_out.result.status.value,
            "ares_flags": ares_out.red_flags,
            "resolved_reg_id": ares_out.resolved_reg_id,
            "domain_score": domain_out.result.score,
            "domain_status": domain_out.result.status.value,
            "domain_flags": domain_out.red_flags,
            "official_site_social_urls": domain_out.social_urls,
            "official_site_emails": domain_out.extracted_emails[:5],
            "official_site_phones": domain_out.extracted_phones[:5],
            "needs_contact_deep_check": needs_contact_deep_check,
            "phone_corporate_prefix_hint": phone_prefix_hint,
        }
        claude_out = await _run_module_with_timeout(
            "claude_osint",
            audit_with_claude_osint(
                request=request,
                settings=cfg,
                legal_name=ares_out.legal_name,
                official_address=ares_out.official_address,
                deterministic_context=deterministic_context,
                anthropic_client=anthropic_client,
            ),
            timeout_seconds=cfg.CLAUDE_TIMEOUT_SECONDS,
            fallback_factory=lambda err: ClaudeOSINTCheckOutput(
                result=ModuleResult(
                    status=ModuleStatus.UNAVAILABLE,
                    score=0,
                    max_score=CLAUDE_MAX_SCORE,
                    details={"error": err},
                )
            ),
        )

    # ------------------------------------------------------------------
    # Aggregate Signals, Deterministic Scoring & Verdict Calculation
    # ------------------------------------------------------------------
    modules = ModulesOutput(
        gov_registry=ares_out.result,
        domain_site=domain_out.result,
        claude_osint=claude_out.result,
    )

    kill_switches = list(
        dict.fromkeys(
            ares_out.kill_switches + domain_out.kill_switches + claude_out.kill_switches
        )
    )
    red_flags = list(
        dict.fromkeys(
            ares_out.red_flags + domain_out.red_flags + claude_out.red_flags
        )
    )
    evidence_links = list(
        dict.fromkeys(
            ares_out.evidence_links + domain_out.evidence_links + claude_out.evidence_links
        )
    )

    status, trust_score, confidence_score = compute_verdict(modules, kill_switches)
    summary = build_summary(
        company_name=ares_out.legal_name or request.company_name,
        status=status,
        trust_score=trust_score,
        confidence_score=confidence_score,
        kill_switches=kill_switches,
        red_flags=red_flags,
        modules=modules,
    )

    company = await _upsert_company(db, request, ares_out)
    log = await _persist_verification_log(
        db=db,
        company_id=company.id if company else None,
        input_hash=input_hash,
        status=status,
        trust_score=trust_score,
        confidence_score=confidence_score,
        kill_switches=kill_switches,
        red_flags=red_flags,
        evidence_links=evidence_links,
        summary=summary,
        modules=modules,
        cache_ttl_days=cfg.CACHE_TTL_DAYS,
    )

    logger.info(
        "Verification complete verification_id=%s reg_id=%s status=%s score=%d confidence=%.2f kill_switches=%s",
        log.id,
        request.reg_id,
        status.value,
        trust_score,
        confidence_score,
        kill_switches,
    )
    return log_to_response(log, cached=False)
