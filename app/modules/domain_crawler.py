"""Module B: Website, WHOIS/RDAP domain age, and lightweight async EU site crawler."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import httpx
import whois
from bs4 import BeautifulSoup
from dateutil import parser as dateutil_parser

from app.config import Settings
from app.schemas import (
    ModuleResult,
    ModuleStatus,
    VerificationRequest,
    extract_root_domain,
    normalize_email,
    normalize_phone_e164,
)

logger = logging.getLogger(__name__)

MAX_SCORE = 25
WHOIS_TIMEOUT_SECONDS = 4.0
RDAP_TIMEOUT_SECONDS = 2.5
PAGE_TIMEOUT_SECONDS = 3.2
MAX_PAGE_BYTES = 1_500_000  # 1.5 MB to handle Wix/Webflow pages with footer links
MAX_SECONDARY_PAGES = 5

FREE_EMAIL_PROVIDERS: frozenset[str] = frozenset(
    {
        "gmail.com",
        "seznam.cz",
        "mail.ru",
        "yahoo.com",
        "outlook.com",
        "yandex.ru",
        "proton.me",
        "hotmail.com",
        "live.com",
        "icloud.com",
        "protonmail.com",
        "email.cz",
        "post.cz",
        "centrum.cz",
    }
)

IGNORED_EMAIL_DOMAINS: frozenset[str] = frozenset(
    {
        "sentry.io",
        "wixpress.com",
        "sentry.wixpress.com",
        "sentry-next.wixpress.com",
        "example.com",
        "example.org",
    }
)

STANDARD_PATHS: tuple[str, ...] = (
    "/kontakt",
    "/kontakty",
    "/contact",
    "/contacts",
    "/kariera",
    "/karriere",
    "/o-nas",
    "/impressum",
    "/datenschutz",
    "/agb",
)

DISCOVERY_PATH_KEYWORDS: tuple[str, ...] = (
    "kontakt",
    "contact",
    "kariera",
    "karriere",
    "career",
    "o-nas",
    "o-firmie",
    "about",
    "uber-uns",
    "ueber-uns",
    "über-uns",
    "impressum",
    "imprint",
    "datenschutz",
    "agb",
    "legal",
    "spolecnost",
    "company",
    "praca",
)

EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")
PHONE_RE = re.compile(
    r"(?:\+|00)\d{1,3}[\s.\-/()]*\d(?:[\s.\-/()]*\d){6,12}"
    r"|\b0\d{2,4}[\s.\-/]\d{3,4}(?:[\s.\-/]\d{2,4})?\b"
    r"|\b\d{3}[\s.\-]\d{3}[\s.\-]\d{3}\b"
)
ICO_PLAIN_RE = re.compile(r"\b(\d{8}|\d{10})\b")
ICO_LABELED_RE = re.compile(
    r"(?:IČO?|ICO|NIP|ID\s*No\.?|Company\s*ID)\s*[:.#]?\s*(\d[\d\s\-]{6,13}\d)",
    flags=re.IGNORECASE,
)
SOCIAL_DOMAINS: tuple[str, ...] = ("linkedin.com", "instagram.com", "facebook.com")
SOCIAL_IGNORE_SUBSTRINGS: tuple[str, ...] = ("/sharer", "/share", "intent/tweet", "/dialog/")


@dataclass
class DomainCheckOutput:
    """Internal result passed from Module B to the orchestrator and Claude context."""

    result: ModuleResult
    kill_switches: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    evidence_links: list[str] = field(default_factory=list)
    social_urls: list[str] = field(default_factory=list)
    extracted_emails: list[str] = field(default_factory=list)
    extracted_phones: list[str] = field(default_factory=list)


def _is_valid_contact_email(addr: str) -> bool:
    cleaned = addr.strip().lower()
    if "@" not in cleaned:
        return False
    domain = cleaned.rsplit("@", 1)[-1]
    if domain in IGNORED_EMAIL_DOMAINS or domain.endswith(".wixpress.com"):
        return False
    return True


def _parse_creation_date(raw_value: Any) -> Optional[datetime]:
    """Normalize python-whois / RDAP creation_date (datetime, list, or str) to earliest UTC datetime."""
    if raw_value is None:
        return None
    candidates = raw_value if isinstance(raw_value, (list, tuple)) else [raw_value]
    parsed_dates: list[datetime] = []
    for item in candidates:
        dt: Optional[datetime] = None
        if isinstance(item, datetime):
            dt = item
        elif isinstance(item, str) and item.strip():
            try:
                dt = dateutil_parser.parse(item.strip())
            except (ValueError, TypeError, OverflowError):
                dt = None
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            parsed_dates.append(dt)
    return min(parsed_dates) if parsed_dates else None


def _whois_lookup_sync(domain: str) -> tuple[Optional[datetime], Optional[str]]:
    """Run synchronous python-whois lookup and return (creation_date, error_message)."""
    try:
        record = whois.whois(domain)
        if not record:
            return None, "Empty WHOIS response"
        creation_dt = _parse_creation_date(record.get("creation_date"))
        if creation_dt is None:
            return None, "WHOIS creation_date unavailable or redacted"
        return creation_dt, None
    except Exception as exc:
        return None, f"WHOIS lookup failed: {type(exc).__name__}"


async def _rdap_creation_date_fallback(
    domain: str, client: httpx.AsyncClient, user_agent: str
) -> Optional[datetime]:
    """Fallback to public RDAP registration event when WHOIS redacts creation_date."""
    try:
        resp = await client.get(
            f"https://rdap.org/domain/{domain}",
            headers={"Accept": "application/rdap+json", "User-Agent": user_agent},
            follow_redirects=True,
            timeout=httpx.Timeout(RDAP_TIMEOUT_SECONDS),
        )
        if resp.status_code != 200:
            return None
        events = resp.json().get("events") or []
        for ev in events:
            if isinstance(ev, dict) and str(ev.get("eventAction", "")).lower() == "registration":
                dt = _parse_creation_date(ev.get("eventDate"))
                if dt is not None:
                    return dt
    except (httpx.HTTPError, ValueError):
        return None
    return None


async def _check_whois_age(
    domain: str, client: httpx.AsyncClient, user_agent: str
) -> tuple[int, list[str], dict[str, Any]]:
    """Compute WHOIS/RDAP domain age score (max 5 pts) without blocking the event loop."""
    if not domain:
        return 0, [], {"available": False, "error": "No domain provided"}

    source = "whois"
    error: Optional[str] = None
    creation_dt: Optional[datetime] = None

    try:
        creation_dt, error = await asyncio.wait_for(
            asyncio.to_thread(_whois_lookup_sync, domain),
            timeout=WHOIS_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("WHOIS lookup timed out for domain=%s", domain)
        error = "WHOIS lookup timed out"

    if creation_dt is None:
        rdap_dt = await _rdap_creation_date_fallback(domain, client, user_agent)
        if rdap_dt is not None:
            creation_dt = rdap_dt
            source = "rdap"
            error = None

    if creation_dt is None:
        return 0, [], {"available": False, "domain": domain, "error": error}

    age_days = max(0, (datetime.now(timezone.utc) - creation_dt).days)
    flags: list[str] = []
    if age_days >= 365:
        score = 5
    elif age_days >= 60:
        score = 2
    else:
        score = 0
        flags.append("FRESH_DOMAIN_UNDER_60_DAYS")

    return (
        score,
        flags,
        {
            "available": True,
            "source": source,
            "domain": domain,
            "creation_date": creation_dt.date().isoformat(),
            "age_days": age_days,
            "score": score,
        },
    )


def _extract_jsonld_artifacts(
    soup: BeautifulSoup, emails: set[str], phones: set[str], social_urls: set[str]
) -> None:
    """Extract emails, phones, and sameAs social URLs from JSON-LD structured data blocks."""
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text() or ""
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except ValueError:
            continue

        def _walk(obj: Any) -> None:
            if isinstance(obj, dict):
                for key, val in obj.items():
                    key_l = str(key).lower()
                    if key_l == "email" and isinstance(val, str):
                        addr = val.replace("mailto:", "").strip().lower()
                        if EMAIL_RE.fullmatch(addr) and _is_valid_contact_email(addr):
                            emails.add(addr)
                    elif key_l == "telephone" and isinstance(val, str):
                        if len(re.sub(r"\D", "", val)) >= 7:
                            phones.add(val.strip())
                    elif key_l == "sameas":
                        urls = val if isinstance(val, list) else [val]
                        for u in urls:
                            if isinstance(u, str) and any(sd in u.lower() for sd in SOCIAL_DOMAINS):
                                if not any(ign in u.lower() for ign in SOCIAL_IGNORE_SUBSTRINGS):
                                    social_urls.add(u.split("#", 1)[0].strip())
                    else:
                        _walk(val)
            elif isinstance(obj, list):
                for item in obj:
                    _walk(item)

        _walk(data)


def _extract_artifacts_from_html(
    html: str, page_url: str, site_root: str
) -> tuple[set[str], set[str], set[str], set[str], list[str]]:
    """Parse HTML + JSON-LD and extract emails, phones, reg IDs, social URLs, and contact links."""
    soup = BeautifulSoup(html, "html.parser")
    emails: set[str] = set()
    phones: set[str] = set()
    social_urls: set[str] = set()

    # Extract JSON-LD before removing <script> tags
    _extract_jsonld_artifacts(soup, emails, phones, social_urls)

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(" ", strip=True)
    for m in EMAIL_RE.findall(text):
        if _is_valid_contact_email(m):
            emails.add(m.lower())

    icos: set[str] = set(ICO_PLAIN_RE.findall(text))
    for match in ICO_LABELED_RE.findall(text):
        digits = re.sub(r"\D", "", match)
        if len(digits) in (8, 10):
            icos.add(digits)

    for raw_phone in PHONE_RE.findall(text):
        digits = re.sub(r"\D", "", raw_phone)
        if 7 <= len(digits) <= 15:
            phones.add(raw_phone.strip())

    discovered_links: list[str] = []

    for anchor in soup.find_all("a", href=True):
        href = str(anchor.get("href") or "").strip()
        if not href:
            continue
        lower_href = href.lower()
        if lower_href.startswith("mailto:"):
            addr = href[7:].split("?", 1)[0].strip().lower()
            if EMAIL_RE.fullmatch(addr) and _is_valid_contact_email(addr):
                emails.add(addr)
            continue
        if lower_href.startswith("tel:"):
            tel = href[4:].split("?", 1)[0].strip()
            if len(re.sub(r"\D", "", tel)) >= 7:
                phones.add(tel)
            continue

        abs_url = urljoin(page_url, href)
        parsed = urlparse(abs_url)
        if parsed.scheme not in {"http", "https"}:
            continue

        host = (parsed.hostname or "").lower()
        if any(sd in host for sd in SOCIAL_DOMAINS):
            if not any(ign in abs_url.lower() for ign in SOCIAL_IGNORE_SUBSTRINGS):
                social_urls.add(abs_url.split("#", 1)[0])
            continue

        if site_root and extract_root_domain(host) == site_root:
            path_lower = parsed.path.lower()
            if any(kw in path_lower for kw in DISCOVERY_PATH_KEYWORDS):
                clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
                if clean_url not in discovered_links:
                    discovered_links.append(clean_url)

    return emails, phones, icos, social_urls, discovered_links


async def _fetch_html(
    client: httpx.AsyncClient, url: str, user_agent: str
) -> Optional[tuple[str, str]]:
    """Fetch a single web page with strict timeout and byte limit; return (final_url, html)."""
    try:
        response = await client.get(
            url,
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
            },
            follow_redirects=True,
            timeout=httpx.Timeout(PAGE_TIMEOUT_SECONDS),
        )
    except httpx.HTTPError:
        return None

    if response.status_code >= 400:
        return None
    content_type = response.headers.get("content-type", "").lower()
    if content_type and "html" not in content_type and "text/" not in content_type:
        return None

    raw_bytes = response.content[:MAX_PAGE_BYTES]
    encoding = response.encoding or "utf-8"
    try:
        text = raw_bytes.decode(encoding, errors="replace")
    except LookupError:
        text = raw_bytes.decode("utf-8", errors="replace")
    return str(response.url), text


def _phone_matches(target_e164: str, extracted_phones: set[str], country_code: str) -> bool:
    """Match target phone against extracted site phones using E.164 and local digit suffixes."""
    if not target_e164 or not extracted_phones:
        return False
    target_digits = re.sub(r"\D", "", target_e164)
    # Use last 8 digits (handles Austrian/German shorter landline numbers like 03865 2247!)
    tail_len = min(len(target_digits), 9 if country_code.upper() in {"CZ", "PL", "SK"} else 8)
    target_tail = target_digits[-tail_len:] if tail_len >= 7 else target_digits

    for raw in extracted_phones:
        norm = normalize_phone_e164(raw, country_code=country_code)
        if norm and norm == target_e164:
            return True
        raw_digits = re.sub(r"\D", "", raw)
        if target_tail and len(raw_digits) >= len(target_tail) and raw_digits.endswith(target_tail):
            return True
    return False


async def _crawl_website(
    request: VerificationRequest,
    site_root: str,
    client: httpx.AsyncClient,
    settings: Settings,
) -> tuple[int, list[str], list[str], list[str], list[str], list[str], dict[str, Any]]:
    """Crawl homepage + up to 5 discovered/standard contact paths (max 10 pts)."""
    if not request.website_url:
        return 0, [], [], [], [], [], {"crawled": False, "reason": "No website_url provided"}

    homepage = await _fetch_html(client, request.website_url, settings.HTTP_USER_AGENT)
    if homepage is None:
        flags: list[str] = []
        if request.contact_email or request.contact_phone:
            flags.append("CONTACT_NOT_LISTED_ON_OFFICIAL_SITE")
        return (
            0,
            flags,
            [],
            [],
            [],
            [],
            {
                "crawled": False,
                "pages_fetched": 0,
                "error": f"Could not fetch homepage {request.website_url}",
            },
        )

    final_home_url, home_html = homepage
    all_emails, all_phones, all_icos, all_socials, discovered = _extract_artifacts_from_html(
        home_html, final_home_url, site_root
    )

    parsed_home = urlparse(final_home_url)
    origin = f"{parsed_home.scheme}://{parsed_home.netloc}"
    home_canonical = final_home_url.rstrip("/")

    candidate_urls: list[str] = []
    seen: set[str] = {home_canonical}

    for url in discovered:
        norm = url.rstrip("/")
        if norm not in seen:
            seen.add(norm)
            candidate_urls.append(url)
        if len(candidate_urls) >= MAX_SECONDARY_PAGES:
            break

    if len(candidate_urls) < MAX_SECONDARY_PAGES:
        for path in STANDARD_PATHS:
            full_url = urljoin(origin + "/", path.lstrip("/"))
            norm = full_url.rstrip("/")
            if norm not in seen:
                seen.add(norm)
                candidate_urls.append(full_url)
            if len(candidate_urls) >= MAX_SECONDARY_PAGES:
                break

    pages_fetched = 1
    if candidate_urls:
        secondary_results = await asyncio.gather(
            *(_fetch_html(client, u, settings.HTTP_USER_AGENT) for u in candidate_urls),
            return_exceptions=True,
        )
        for res in secondary_results:
            if isinstance(res, tuple):
                pages_fetched += 1
                sub_url, sub_html = res
                emails, phones, icos, socials, _ = _extract_artifacts_from_html(
                    sub_html, sub_url, site_root
                )
                all_emails.update(emails)
                all_phones.update(phones)
                all_icos.update(icos)
                all_socials.update(socials)

    crawl_score = 0
    flags = []

    ico_found = request.reg_id in all_icos
    if ico_found:
        crawl_score += 5

    target_email = normalize_email(request.contact_email)
    target_phone = normalize_phone_e164(request.contact_phone, request.country_code)

    email_found = bool(target_email and target_email in all_emails)
    phone_found = _phone_matches(target_phone, all_phones, request.country_code)
    contact_found = email_found or phone_found

    if contact_found:
        crawl_score += 5
    elif target_email or target_phone:
        flags.append("CONTACT_NOT_LISTED_ON_OFFICIAL_SITE")

    sorted_socials = sorted(all_socials)[:10]
    sorted_emails = sorted(all_emails)[:10]
    sorted_phones = sorted(all_phones)[:10]
    evidence_links = [final_home_url] + sorted_socials[:5]

    details = {
        "crawled": True,
        "final_url": final_home_url,
        "pages_fetched": pages_fetched,
        "ico_found_on_site": ico_found,
        "email_found_on_site": email_found,
        "phone_found_on_site": phone_found,
        "extracted_emails_count": len(all_emails),
        "extracted_phones_count": len(all_phones),
        "extracted_icos_sample": sorted(all_icos)[:10],
        "social_urls": sorted_socials,
        "score": crawl_score,
    }
    return (
        crawl_score,
        flags,
        evidence_links,
        sorted_socials,
        sorted_emails,
        sorted_phones,
        details,
    )


async def check_domain_and_site(
    request: VerificationRequest,
    client: httpx.AsyncClient,
    settings: Settings,
) -> DomainCheckOutput:
    """Run Email/Domain match (10), WHOIS/RDAP age (5), and async site crawl (10) in parallel."""
    site_root = extract_root_domain(request.website_url)
    email_root = extract_root_domain(request.contact_email)

    red_flags: list[str] = []
    evidence_links: list[str] = []

    # 1. Email vs Website Domain Match (max 10 pts)
    email_domain_score = 0
    email_match_status = "not_evaluated"
    if email_root and site_root:
        if email_root == site_root:
            email_domain_score = 10
            email_match_status = "exact_match"
        elif email_root in FREE_EMAIL_PROVIDERS:
            email_domain_score = 0
            email_match_status = "free_email_provider"
            red_flags.append("FREE_EMAIL_PROVIDER")
        else:
            email_domain_score = 0
            email_match_status = "domain_mismatch"
            red_flags.append("DOMAIN_MISMATCH")
    elif email_root and email_root in FREE_EMAIL_PROVIDERS:
        email_domain_score = 0
        email_match_status = "free_email_provider"
        red_flags.append("FREE_EMAIL_PROVIDER")
    elif not site_root:
        email_match_status = "missing_website"
        red_flags.append("MISSING_WEBSITE")

    # 2 & 3. Run WHOIS/RDAP lookup and lightweight site crawl concurrently
    whois_task = _check_whois_age(site_root, client=client, user_agent=settings.HTTP_USER_AGENT)
    crawl_task = _crawl_website(request, site_root, client=client, settings=settings)
    (whois_score, whois_flags, whois_details), (
        crawl_score,
        crawl_flags,
        crawl_evidence,
        social_urls,
        extracted_emails,
        extracted_phones,
        crawl_details,
    ) = await asyncio.gather(whois_task, crawl_task)

    red_flags.extend(whois_flags)
    red_flags.extend(crawl_flags)
    evidence_links.extend(crawl_evidence)

    total_score = min(MAX_SCORE, email_domain_score + whois_score + crawl_score)
    details: dict[str, Any] = {
        "website_root_domain": site_root or None,
        "email_root_domain": email_root or None,
        "email_domain_match": email_match_status,
        "whois": whois_details,
        "site_crawl": crawl_details,
        "score_breakdown": {
            "email_domain_match": email_domain_score,
            "whois_age": whois_score,
            "site_crawl": crawl_score,
        },
    }

    return DomainCheckOutput(
        result=ModuleResult(
            status=ModuleStatus.SUCCESS,
            score=total_score,
            max_score=MAX_SCORE,
            details=details,
        ),
        red_flags=list(dict.fromkeys(red_flags)),
        evidence_links=list(dict.fromkeys(evidence_links)),
        social_urls=social_urls,
        extracted_emails=extracted_emails,
        extracted_phones=extracted_phones,
    )
