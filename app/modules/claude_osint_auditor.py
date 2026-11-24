"""Unified Claude OSINT & Vacancy Fraud Auditor (max 50 pts) using Anthropic SDK + web_search_20250305.

Replaces separate Google Places, Serper, and LLM modules with one server-side web-search-powered audit:
- 20 pts: Physical presence / Google Maps / employee & public reviews
- 15 pts: LinkedIn company footprint, phone/company scam OSINT, virtual office detection
- 15 pts: Vacancy text & reviews anti-fraud audit (+ calibrated UPFRONT_FEE_SCAM kill-switch)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlparse

from anthropic import APIError, APITimeoutError, AsyncAnthropic, BadRequestError
from pydantic import ValidationError

from app.config import Settings
from app.schemas import (
    ClaudeOSINTAuditOutput,
    ModuleResult,
    ModuleStatus,
    VerificationRequest,
)

logger = logging.getLogger(__name__)

MAX_SCORE = 50
MAPS_REVIEWS_MAX = 20
WEB_OSINT_MAX = 15
VACANCY_AUDIT_MAX = 15
MAX_EVIDENCE_LINKS = 25

# Fallback approximate city/timezone when Anthropic web_search rejects an unsupported ISO country code
COUNTRY_CITY_FALLBACK: dict[str, dict[str, str]] = {
    "CZ": {"type": "approximate", "city": "Prague", "timezone": "Europe/Prague"},
    "SK": {"type": "approximate", "city": "Bratislava", "timezone": "Europe/Bratislava"},
    "PL": {"type": "approximate", "city": "Warsaw", "timezone": "Europe/Warsaw"},
    "DE": {"type": "approximate", "city": "Berlin", "timezone": "Europe/Berlin"},
    "AT": {"type": "approximate", "city": "Vienna", "timezone": "Europe/Vienna"},
}

# Generic non-company noise domains that occasionally surface in scam-keyword searches
NOISE_EVIDENCE_DOMAINS: frozenset[str] = frozenset(
    {
        "wikipedia.org",
        "wikidata.org",
        "attorneygeneral.gov",
        "consumerfinance.gov",
        "reportfraud.microsoft.com",
        "house.gov",
        "cia.gov",
        "scannero.com",
        "call-abroad.com",
    }
)
NOISE_URL_SUBSTRINGS: tuple[str, ...] = (
    "bbb.org/scamtracker/lookupscam",
    "/wiki/",
)

SYSTEM_PROMPT = """You are a Senior Anti-Fraud OSINT & Employment Compliance Investigator specializing in Czech Republic (CZ) and EU job postings targeting foreign workers.

You have access to the `web_search` tool (up to 5 searches). Use targeted web searches to investigate:
1. Physical presence, Google Maps / Firmy.cz / Herold / public reviews for the company name + official address (check whether physical address/website matches, rating/review volume, and whether reviews mention scam keywords: scam, podvod, мошенники, кинули, не платят, fake, prepayment).
2. LinkedIn company page (`site:linkedin.com/company "<company_name>"`).
3. Scam / blacklist reports on the company name and contact phone (`scam OR podvod OR мошенники OR кидалово OR "не платят" OR отзывы`).
4. Virtual office / mass registration at the official address (`"<official_address>"` or `virtuální sídlo`).
5. Audit the vacancy text and applicant_type (`direct_employer` vs `agency`) alongside your web search findings AND the deterministic pre-check context.

CRITICAL CALIBRATION RULES:
A) UPFRONT FEE KILL-SWITCH (`requires_upfront_payment`):
   - Set `requires_upfront_payment: true` and `vacancy_audit_score: 0` ONLY when the candidate is asked to pay money to the agency, recruiter, broker, or employer upfront (for invitation letters, visa packages, recruitment/placement fees, "deposits", contract fees, paperwork, or agency registration — e.g. paying 500,000 KZT or 300 EUR to an agency).
   - Do NOT set `requires_upfront_payment: true` for:
     * Official government/embassy statutory consular fees paid directly to a consulate/embassy — especially when the employer states they REIMBURSE the consular fee upon presenting the receipt!
     * Standard ongoing monthly worker rent/accommodation paid while employed (e.g. 4000-5000 CZK/month).
B) OFFICIAL SOCIAL CHANNEL VERIFICATION:
   - Check `official_site_social_urls` in the Deterministic Pre-Check Summary. If the vacancy comes from an Instagram, Facebook, or LinkedIn handle that appears in `official_site_social_urls` (crawled directly from the employer's official corporate website), treat that social account as VERIFIED OFFICIAL — do NOT flag it as brand impersonation!
   - Conversely, if a posting claims a famous company name but uses an unverified personal WhatsApp/Telegram number or unlinked social account, DO flag impersonation risk.

C) CONDITIONAL RECRUITER / PHONE VERIFICATION (ONLY when `needs_contact_deep_check` is true):
   - When `needs_contact_deep_check` is false: leave `contact_verification_status: "NOT_NEEDED"` and the other contact verification fields null (do NOT waste searches on it).
   - When `needs_contact_deep_check` is true (a personal phone/recruiter name was given and NOT found on the company site by our free crawler):
     1. PRIORITIZE searching for the exact phone number (both full and local spaced format) and recruiter name alongside the company/brand (and on the company's real official domain, e.g., `site:manpower.cz`).
     2. ALSO search the 6-digit corporate mobile prefix (`phone_corporate_prefix_hint`, e.g., `"703 865"` on the company's official domain!) to check whether neighboring numbers belong to real corporate recruiters/staff at that company!
     3. Set `contact_verification_status`:
        - `"VERIFIED_EXACT"`: exact phone or recruiter+company found directly on the official company site, official LinkedIn, or official job posting.
        - `"VERIFIED_CORPORATE_BLOCK"`: exact number not listed publicly, BUT the 6-digit corporate phone prefix (`phone_corporate_prefix_hint`) IS confirmed on the official company website for other staff/recruiters (cite the exact neighboring staff numbers & URLs in `contact_verification_evidence`!). Do NOT mark as a scammer just for using WhatsApp when the corporate SIM block matches and there are no scam reports on that phone (give balanced cautious `vacancy_audit_score` 8-12).
        - `"UNVERIFIED"`: neither the exact phone/recruiter nor its corporate phone prefix appears on official company channels.
     4. Whenever status is `"VERIFIED_CORPORATE_BLOCK"` or `"UNVERIFIED"`:
        - Put the official company HR email, official phone, and official contacts page URL in `official_verification_contact`.
        - Write `verification_message_local`: a short, formal, grammatically accurate, ready-to-copy email/message in the **company's local language** ONLY (start with formal 'Dobrý den,' for Czech, 'Guten Tag,' for German, 'Dzień dobry,' for Polish — no informal 'Ahoj' and no English translation appended) asking HR to confirm whether the recruiter and phone number work for them — so a worker who doesn't speak the local language can copy-paste and send it directly!
        - Write `user_action_advice`: 2-3 clear sentences in **Russian** explaining what we found (did we find the phone or its corporate number block on the official site?) and what the worker should do next without needing to speak Czech/German/Polish.

SCORING RUBRIC (TOTAL MAX = 50 POINTS):
- `maps_reviews_score` (0 to 20):
  - Physical listing/Maps presence matching official address or website: +10 (found with mismatch: +5; not found: 0).
  - Public reviews >= 5 and rating >= 3.5: +5.
  - No fraud/scam keywords in reviews: +5 (0 if scam keywords found, or if no presence found).
- `web_osint_score` (0 to 15):
  - Valid LinkedIn company page found: +10.
  - No credible negative scam/fraud complaints found specifically targeting this company or phone: +5 (0 if credible fraud complaints found; ignore generic unrelated scam articles).
  - If official address is a virtual office (`virtuální sídlo`), set `virtual_office_detected: true` and include `VIRTUAL_OFFICE_ADDRESS` in `red_flags`.
- `vacancy_audit_score` (0 to 15):
  - 15 = realistic salary, transparent hiring, clean reviews/OSINT.
  - 10-14 = minor ambiguity.
  - 5-9 = notable risk indicators (unrealistic pay, urgency, Telegram/WhatsApp-only funnel, illegal discrimination).
  - 0-4 = severe fraud indicators.
  - 0 = `requires_upfront_payment: true` (upfront fee scam).

OUTPUT REQUIREMENT:
Do NOT narrate or write commentary between tool searches. After completing your searches, output ONLY a single valid JSON object (no prose outside JSON) matching this EXACT schema:
{
  "maps_reviews_score": 0,
  "web_osint_score": 0,
  "vacancy_audit_score": 0,
  "maps_presence_found": false,
  "maps_address_or_website_match": false,
  "maps_rating": null,
  "maps_reviews_count": null,
  "maps_reviews_scam_keywords": false,
  "google_maps_url": null,
  "linkedin_company_url": null,
  "negative_osint_reports_found": false,
  "virtual_office_detected": false,
  "requires_upfront_payment": false,
  "unrealistic_salary_or_promises": false,
  "artificial_urgency_detected": false,
  "shady_messenger_only_funnel": false,
  "reviews_indicate_fraud": false,
  "contact_verification_status": "NOT_NEEDED",
  "contact_verification_evidence": null,
  "official_verification_contact": null,
  "verification_message_local": null,
  "user_action_advice": null,
  "cited_urls": [],
  "red_flags": [],
  "reasoning": "Concise 1-3 sentence justification of findings."
}

Use short UPPER_SNAKE_CASE codes in `red_flags` and put only verified `https://` URLs directly relevant to this company in `cited_urls`.
"""


@dataclass
class ClaudeOSINTCheckOutput:
    """Internal result returned by Module C (Unified Claude OSINT & Vacancy Auditor)."""

    result: ModuleResult
    kill_switches: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    evidence_links: list[str] = field(default_factory=list)


def _build_web_search_tool(country_code: str, max_uses: int) -> dict[str, Any]:
    """Build Anthropic's server-side web_search_20250305 tool configuration."""
    return {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": max_uses,
        "user_location": {
            "type": "approximate",
            "country": country_code.upper(),
        },
    }


def _build_fallback_web_search_tool(country_code: str, max_uses: int) -> dict[str, Any]:
    """Fallback web_search_20250305 tool if Anthropic rejects an unsupported country code."""
    tool: dict[str, Any] = {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": max_uses,
    }
    city_loc = COUNTRY_CITY_FALLBACK.get(country_code.upper())
    if city_loc:
        tool["user_location"] = dict(city_loc)
    return tool


def _build_user_prompt(
    request: VerificationRequest,
    legal_name: Optional[str],
    official_address: Optional[str],
    deterministic_context: Optional[dict[str, Any]],
) -> str:
    """Assemble structured employer dossier from request + deterministic Modules A & B."""
    lines = [
        f"Company Name (submitted): {request.company_name}",
        f"Official Legal Name (Gov Registry): {legal_name or 'Unknown'}",
        f"Country Code: {request.country_code}",
        f"Registration ID: {request.reg_id}",
        f"Official Registered Address: {official_address or 'Unknown'}",
        f"Website URL: {request.website_url or 'Not provided'}",
        f"Contact Person / Recruiter Name: {request.contact_person or 'Not provided'}",
        f"Contact Email: {request.contact_email or 'Not provided'}",
        f"Contact Phone (E.164): {request.contact_phone or 'Not provided'}",
        f"Applicant Type: {request.applicant_type.value}",
    ]
    if deterministic_context:
        lines.append(
            f"Deterministic Pre-Check Summary: {json.dumps(deterministic_context, ensure_ascii=False)}"
        )
    vacancy_body = (request.vacancy_text or "").strip() or "(No vacancy text provided)"
    lines.append(f"\n--- VACANCY TEXT ---\n{vacancy_body[:8000]}")
    if deterministic_context and deterministic_context.get("needs_contact_deep_check"):
        prefix_hint = deterministic_context.get("phone_corporate_prefix_hint")
        lines.append(
            "\nPRIORITY TASK (`needs_contact_deep_check=true`): The submitted recruiter/phone was NOT verified "
            "on an official company website yet. Before generic checks, search for: "
            f"(1) exact phone {request.contact_phone or ''} and recruiter '{request.contact_person or ''}' "
            f"associated with '{request.company_name}', "
            f"(2) the 6-digit corporate phone prefix '{prefix_hint or ''}' on the company's official website "
            "(to detect if neighboring staff numbers belong to the same corporate SIM block), and "
            "(3) official HR email / contacts page so you can populate `contact_verification_status`, "
            "`contact_verification_evidence`, `official_verification_contact`, `verification_message_local`, "
            "and `user_action_advice` (in Russian). Output ONLY the strict JSON object."
        )
    else:
        lines.append(
            "\nPerform focused web searches (Maps/reviews, LinkedIn, scam reports on phone & company, "
            "virtual office check on official address) and respond ONLY with the strict JSON object."
        )
    return "\n".join(lines)


def _is_valid_http_url(value: Any) -> bool:
    return isinstance(value, str) and value.strip().startswith(("http://", "https://"))


def _is_relevant_evidence_url(url: str) -> bool:
    """Filter out generic encyclopedia / unrelated government boilerplate pages."""
    if not _is_valid_http_url(url):
        return False
    lower = url.lower()
    if any(sub in lower for sub in NOISE_URL_SUBSTRINGS):
        return False
    host = (urlparse(lower).hostname or "").lstrip("www.")
    for noisy in NOISE_EVIDENCE_DOMAINS:
        if host == noisy or host.endswith("." + noisy):
            return False
    return True


def _extract_web_search_artifacts(
    content_blocks: list[Any],
) -> tuple[list[str], list[str], list[str], int]:
    """Extract text chunks, direct citation URLs, tool result URLs, and web_search count."""
    text_chunks: list[str] = []
    direct_citation_urls: list[str] = []
    search_result_urls: list[str] = []
    search_uses = 0

    for block in content_blocks:
        block_type = getattr(block, "type", None) or (
            block.get("type") if isinstance(block, dict) else None
        )

        if block_type == "server_tool_use":
            tool_name = getattr(block, "name", None) or (
                block.get("name") if isinstance(block, dict) else None
            )
            if tool_name == "web_search":
                search_uses += 1

        elif block_type == "text":
            text_val = getattr(block, "text", None) if not isinstance(block, dict) else block.get("text")
            if isinstance(text_val, str) and text_val.strip():
                text_chunks.append(text_val)

            citations = (
                getattr(block, "citations", None)
                if not isinstance(block, dict)
                else block.get("citations")
            )
            if isinstance(citations, list):
                for cit in citations:
                    cit_url = (
                        getattr(cit, "url", None) if not isinstance(cit, dict) else cit.get("url")
                    )
                    if _is_valid_http_url(cit_url):
                        direct_citation_urls.append(str(cit_url).strip())

        elif block_type == "web_search_tool_result":
            items = (
                getattr(block, "content", None)
                if not isinstance(block, dict)
                else block.get("content")
            )
            if isinstance(items, list):
                for item in items:
                    item_url = (
                        getattr(item, "url", None) if not isinstance(item, dict) else item.get("url")
                    )
                    if _is_valid_http_url(item_url):
                        search_result_urls.append(str(item_url).strip())

    return text_chunks, direct_citation_urls, search_result_urls, search_uses


def _extract_json_object(raw_text: str) -> Optional[dict[str, Any]]:
    """Extract and parse a JSON object from raw text (handling optional markdown fences)."""
    cleaned = raw_text.strip()
    if not cleaned:
        return None

    fence_matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    candidates: list[str] = list(reversed(fence_matches))

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidates.append(cleaned[first_brace : last_brace + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            continue
    return None


def _parse_structured_audit(text_chunks: list[str]) -> ClaudeOSINTAuditOutput:
    """Parse and strictly validate ClaudeOSINTAuditOutput from response text blocks."""
    for chunk in reversed(text_chunks):
        parsed_dict = _extract_json_object(chunk)
        if parsed_dict is not None:
            return ClaudeOSINTAuditOutput.model_validate(parsed_dict)

    combined = "\n".join(text_chunks)
    parsed_dict = _extract_json_object(combined)
    if parsed_dict is not None:
        return ClaudeOSINTAuditOutput.model_validate(parsed_dict)

    raise ValueError("No valid JSON object found in Claude response")


def evaluate_claude_audit(
    audit: ClaudeOSINTAuditOutput,
    direct_citation_urls: list[str],
    search_result_urls: list[str],
    model_name: str,
    search_uses: int,
) -> ClaudeOSINTCheckOutput:
    """Deterministically score, clamp, apply kill-switches, and collect cited evidence URLs."""
    maps_score = max(0, min(MAPS_REVIEWS_MAX, audit.maps_reviews_score))
    osint_score = max(0, min(WEB_OSINT_MAX, audit.web_osint_score))
    vacancy_score = max(0, min(VACANCY_AUDIT_MAX, audit.vacancy_audit_score))

    kill_switches: list[str] = []
    red_flags: list[str] = [str(f).strip() for f in audit.red_flags if str(f).strip()]

    if audit.requires_upfront_payment:
        kill_switches.append("UPFRONT_FEE_SCAM")
        vacancy_score = 0
        if "UPFRONT_PAYMENT_REQUIRED" not in red_flags:
            red_flags.append("UPFRONT_PAYMENT_REQUIRED")

    canonical_flags = (
        (not audit.maps_presence_found, "NO_GOOGLE_MAPS_PRESENCE"),
        (
            audit.maps_presence_found and audit.maps_reviews_count == 0,
            "NO_GOOGLE_REVIEWS",
        ),
        (audit.maps_reviews_scam_keywords, "SCAM_KEYWORDS_IN_MAPS_REVIEWS"),
        (audit.negative_osint_reports_found, "NEGATIVE_OSINT_REPORTS"),
        (audit.virtual_office_detected, "VIRTUAL_OFFICE_ADDRESS"),
        (audit.unrealistic_salary_or_promises, "UNREALISTIC_SALARY_OR_PROMISES"),
        (audit.artificial_urgency_detected, "ARTIFICIAL_URGENCY_DETECTED"),
        (audit.shady_messenger_only_funnel, "SHADY_MESSENGER_ONLY_FUNNEL"),
        (audit.reviews_indicate_fraud, "REVIEWS_INDICATE_FRAUD"),
        (
            audit.contact_verification_status == "UNVERIFIED",
            "UNVERIFIED_RECRUITER_CONTACT",
        ),
    )
    for condition, code in canonical_flags:
        if condition and code not in red_flags:
            red_flags.append(code)

    total_score = 0 if kill_switches else min(MAX_SCORE, maps_score + osint_score + vacancy_score)

    # Aggregate high-signal cited and verified search URLs in priority order
    evidence_candidates: list[str] = []
    evidence_candidates.extend(u for u in direct_citation_urls if _is_relevant_evidence_url(u))
    if _is_valid_http_url(audit.linkedin_company_url):
        evidence_candidates.append(str(audit.linkedin_company_url).strip())
    if _is_valid_http_url(audit.google_maps_url):
        evidence_candidates.append(str(audit.google_maps_url).strip())
    evidence_candidates.extend(
        u.strip() for u in audit.cited_urls if _is_relevant_evidence_url(u)
    )
    evidence_candidates.extend(u for u in search_result_urls if _is_relevant_evidence_url(u))
    evidence_links = list(dict.fromkeys(evidence_candidates))[:MAX_EVIDENCE_LINKS]

    clean_reasoning = re.sub(r"</?cite\b[^>]*>", "", audit.reasoning or "").strip()
    clean_evidence = (
        re.sub(r"</?cite\b[^>]*>", "", audit.contact_verification_evidence).strip()
        if audit.contact_verification_evidence
        else None
    )

    details: dict[str, Any] = {
        "model": model_name,
        "web_search_uses": search_uses,
        "score_breakdown": {
            "maps_reviews": maps_score,
            "web_osint": osint_score,
            "vacancy_audit": vacancy_score,
        },
        "maps_presence_found": audit.maps_presence_found,
        "maps_address_or_website_match": audit.maps_address_or_website_match,
        "maps_rating": audit.maps_rating,
        "maps_reviews_count": audit.maps_reviews_count,
        "maps_reviews_scam_keywords": audit.maps_reviews_scam_keywords,
        "google_maps_url": audit.google_maps_url,
        "linkedin_company_url": audit.linkedin_company_url,
        "negative_osint_reports_found": audit.negative_osint_reports_found,
        "virtual_office_detected": audit.virtual_office_detected,
        "requires_upfront_payment": audit.requires_upfront_payment,
        "unrealistic_salary_or_promises": audit.unrealistic_salary_or_promises,
        "artificial_urgency_detected": audit.artificial_urgency_detected,
        "shady_messenger_only_funnel": audit.shady_messenger_only_funnel,
        "reviews_indicate_fraud": audit.reviews_indicate_fraud,
        "contact_verification_status": audit.contact_verification_status,
        "contact_verification_evidence": clean_evidence,
        "official_verification_contact": audit.official_verification_contact,
        "verification_message_local": audit.verification_message_local,
        "user_action_advice": audit.user_action_advice,
        "reasoning": clean_reasoning,
        "cited_urls_count": len(evidence_links),
    }

    return ClaudeOSINTCheckOutput(
        result=ModuleResult(
            status=ModuleStatus.SUCCESS,
            score=total_score,
            max_score=MAX_SCORE,
            details=details,
        ),
        kill_switches=kill_switches,
        red_flags=list(dict.fromkeys(red_flags)),
        evidence_links=evidence_links,
    )


async def audit_with_claude_osint(
    request: VerificationRequest,
    settings: Settings,
    legal_name: Optional[str] = None,
    official_address: Optional[str] = None,
    deterministic_context: Optional[dict[str, Any]] = None,
    anthropic_client: Optional[AsyncAnthropic] = None,
) -> ClaudeOSINTCheckOutput:
    """Run unified Anthropic web-search OSINT + Maps/Reviews + Vacancy Audit (max 50 pts).

    IMPORTANT: Never passes `temperature`, `top_p`, or `top_k` to `messages.create()`.
    """
    if anthropic_client is None and not settings.ANTHROPIC_API_KEY:
        return ClaudeOSINTCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.UNAVAILABLE,
                score=0,
                max_score=MAX_SCORE,
                details={"reason": "ANTHROPIC_API_KEY not configured"},
            )
        )

    owns_client = anthropic_client is None
    client = anthropic_client or AsyncAnthropic(
        api_key=settings.ANTHROPIC_API_KEY,
        timeout=settings.CLAUDE_TIMEOUT_SECONDS,
        max_retries=1,
    )

    user_prompt = _build_user_prompt(
        request=request,
        legal_name=legal_name,
        official_address=official_address,
        deterministic_context=deterministic_context,
    )
    primary_tool = _build_web_search_tool(
        country_code=request.country_code,
        max_uses=settings.CLAUDE_MAX_SEARCH_USES,
    )

    try:
        try:
            response = await client.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=3072,
                system=SYSTEM_PROMPT,
                tools=[primary_tool],
                messages=[{"role": "user", "content": user_prompt}],
            )
        except BadRequestError as exc:
            err_text = str(exc)
            if "Country code" in err_text and "not supported" in err_text:
                fallback_tool = _build_fallback_web_search_tool(
                    country_code=request.country_code,
                    max_uses=settings.CLAUDE_MAX_SEARCH_USES,
                )
                logger.info(
                    "Anthropic web_search rejected country=%s; retrying with approximate location fallback",
                    request.country_code,
                )
                response = await client.messages.create(
                    model=settings.ANTHROPIC_MODEL,
                    max_tokens=3072,
                    system=SYSTEM_PROMPT,
                    tools=[fallback_tool],
                    messages=[{"role": "user", "content": user_prompt}],
                )
            else:
                raise

        content_blocks = list(getattr(response, "content", None) or [])
        text_chunks, direct_citations, search_urls, search_uses = _extract_web_search_artifacts(
            content_blocks
        )
        try:
            validated = _parse_structured_audit(text_chunks)
        except (ValueError, ValidationError):
            # If Claude spent its turn on web searches / commentary without emitting final JSON,
            # run a fast tool-free formatting pass on its gathered findings.
            logger.info("Claude OSINT response lacked valid JSON; running tool-free JSON finalizer")
            summary_context = "\n".join(text_chunks)[-6000:] or "No text emitted after searches."
            urls_hint = ", ".join(search_urls[:15])
            finalize_resp = await client.messages.create(
                model=settings.ANTHROPIC_MODEL,
                max_tokens=1536,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt},
                    {
                        "role": "user",
                        "content": (
                            "Based on your web search findings below and these discovered URLs "
                            f"[{urls_hint}], output ONLY the single valid JSON object matching the schema:\n\n"
                            f"{summary_context}"
                        ),
                    },
                ],
            )
            fin_blocks = list(getattr(finalize_resp, "content", None) or [])
            fin_texts, fin_cits, _, _ = _extract_web_search_artifacts(fin_blocks)
            direct_citations.extend(fin_cits)
            validated = _parse_structured_audit(fin_texts)
        return evaluate_claude_audit(
            audit=validated,
            direct_citation_urls=direct_citations,
            search_result_urls=search_urls,
            model_name=settings.ANTHROPIC_MODEL,
            search_uses=search_uses,
        )

    except APITimeoutError as exc:
        logger.warning("Claude OSINT auditor timed out: %r", exc)
        return ClaudeOSINTCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.UNAVAILABLE,
                score=0,
                max_score=MAX_SCORE,
                details={"error": "Anthropic Claude OSINT request timed out"},
            )
        )
    except APIError as exc:
        logger.warning("Claude OSINT API error: %s", exc)
        return ClaudeOSINTCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.UNAVAILABLE,
                score=0,
                max_score=MAX_SCORE,
                details={"error": f"Anthropic API error: {type(exc).__name__}"},
            )
        )
    except (ValueError, ValidationError) as exc:
        logger.warning("Claude OSINT structured output validation failed: %s", exc)
        return ClaudeOSINTCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.FAILED,
                score=0,
                max_score=MAX_SCORE,
                details={"error": f"Invalid structured JSON from Claude: {type(exc).__name__}"},
            )
        )
    finally:
        if owns_client:
            await client.close()
