"""Stage 2 Multimodal Intake Engine: URL, DOCX, PDF, Screenshots & Raw Text -> VerificationRequest."""

from __future__ import annotations

import base64
import io
import ipaddress
import json
import logging
import re
import socket
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import httpx
from anthropic import APIError, APITimeoutError, AsyncAnthropic
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, ValidationError
from pypdf import PdfReader

from app.config import Settings
from app.modules.ares import search_czech_ares_by_name
from app.modules.domain_crawler import FREE_EMAIL_PROVIDERS
from app.schemas import ApplicantType, VerificationRequest, extract_root_domain

logger = logging.getLogger(__name__)

MAX_HTML_TEXT_CHARS = 12_000
MAX_DOC_TEXT_CHARS = 15_000
MAX_IMAGE_ATTACHMENTS = 6
MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_PDF_BINARY_BYTES = 10 * 1024 * 1024

IMAGE_MIME_BY_EXT: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

INTAKE_SYSTEM_PROMPT = """You extract structured employer verification inputs from raw job postings, screenshots, PDF presentations, Word contracts, and scraped job-board web pages.

Return ONLY a single valid JSON object matching this EXACT schema (no markdown prose outside JSON):
{
  "company_name": "string (official company or agency name, poster handle if named, or 'Anonymous Employer' if no company/brand is named anywhere — NEVER output 'UNKNOWN')",
  "country_code": "2-letter ISO code (e.g. CZ, PL, DE, AT, SK, KZ). Use the employer/agency registration country if stated in a contract (e.g. KZ for Kazakh TOO/BIN, CZ for Czech s.r.o./ICO, PL for Polish sp. z o.o./NIP), else the primary job location country.",
  "reg_id": "official company registration ID digits/code (ICO, NIP, BIN, HRB/HRA, FN) or 'UNKNOWN' if absent",
  "website_url": "official employer website URL (https://...) or null. NEVER put job-board URLs (layboard.com, eurabota, jobs.cz, eures, instagram.com, facebook.com) here.",
  "contact_person": "recruiter or contact person name if mentioned (e.g. 'Lav', 'Artjom', 'Kinga Pesl'), or null",
  "contact_email": "contact email address or null",
  "contact_phone": "contact phone number in international format if available, or null",
  "applicant_type": "direct_employer or agency",
  "vacancy_text": "comprehensive, factual summary preserving ALL key details: source URL/social handle if any, job role, location, salary, currency, visa/work-permit terms, ANY requested fees or prepayments (exact amounts and who receives them), housing deductions, contract clauses (e.g. non-refund, NDA/chat bans), and contact channels."
}

Rules:
- Preserve exact monetary amounts, payment terms, official addresses, registration numbers, and suspicious clauses verbatim inside `vacancy_text`.
- If the posting is from an Instagram/Facebook/job-board URL, include that handle/URL inside `vacancy_text` so downstream OSINT can verify whether the social account belongs to the official company website.
- If a field is not present in the materials, use null (or "UNKNOWN" for reg_id). Never fabricate company websites or registration IDs.
"""


class ExtractedIntakePayload(BaseModel):
    """Strict Pydantic schema for Claude multimodal intake extraction."""

    company_name: str = Field(..., min_length=1, max_length=255)
    country_code: str = Field(default="CZ", min_length=2, max_length=2)
    reg_id: str = Field(default="UNKNOWN", min_length=1, max_length=32)
    website_url: Optional[str] = None
    contact_person: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    applicant_type: ApplicantType = ApplicantType.direct_employer
    vacancy_text: str = ""


@dataclass
class UploadedAttachment:
    """In-memory representation of an uploaded file."""

    filename: str
    content_type: str
    data: bytes


@dataclass
class IntakeExtractionResult:
    """Result returned by multimodal intake prior to running the verification pipeline."""

    request: VerificationRequest
    sources: list[str] = field(default_factory=list)
    extraction_method: str = "claude_multimodal"


def _is_safe_public_url(url: str) -> bool:
    """Block non-HTTP(S) schemes and local/private network hosts (SSRF protection)."""
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").strip().lower()
    if not host or host in {"localhost", "0.0.0.0"} or host.endswith(".local"):
        return False
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False
    except ValueError:
        # Hostname rather than literal IP; check resolved address best-effort
        try:
            resolved_ip = socket.gethostbyname(host)
            ip = ipaddress.ip_address(resolved_ip)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
        except OSError:
            # Allow DNS failures to be surfaced cleanly by httpx
            pass
    return True


EURES_JV_ID_RE = re.compile(
    r"europa\.eu/eures/(?:portal/jv-se/jv-details|api/jv-searchengine/public/jv/id)/([A-Za-z0-9_\-=]+)",
    flags=re.IGNORECASE,
)


def _strip_html(raw: str) -> str:
    if not raw:
        return ""
    soup = BeautifulSoup(raw, "html.parser")
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()


async def try_fetch_eures_vacancy(
    url: str,
    client: httpx.AsyncClient,
    settings: Settings,
    country_override: Optional[str] = None,
    company_override: Optional[str] = None,
    reg_id_override: Optional[str] = None,
    force_refresh: bool = False,
) -> Optional[IntakeExtractionResult]:
    """Directly fetch structured job data from the official public EU EURES JSON API."""
    match = EURES_JV_ID_RE.search(url.strip())
    if not match:
        return None

    jv_id = match.group(1)
    api_url = f"https://europa.eu/eures/api/jv-searchengine/public/jv/id/{jv_id}"
    resp = await client.get(
        api_url,
        headers={"Accept": "application/json", "User-Agent": settings.HTTP_USER_AGENT},
        timeout=httpx.Timeout(12.0),
    )
    if resp.status_code == 404:
        raise ValueError("Вакансия EURES не найдена или уже снята с публикации (HTTP 404).")
    if resp.status_code != 200:
        raise ValueError(f"Не удалось получить данные вакансии с API EURES (HTTP {resp.status_code}).")

    data = resp.json()
    profiles = data.get("jvProfiles") if isinstance(data.get("jvProfiles"), dict) else {}
    if not profiles:
        raise ValueError("API EURES вернул пустую карточку вакансии.")

    qs = parse_qs(urlparse(url).query)
    desired_lang = (qs.get("jvDisplayLanguage") or [None])[0] or data.get("preferredLanguage")
    profile = profiles.get(desired_lang) or next(iter(profiles.values()))

    employer = profile.get("employer") if isinstance(profile.get("employer"), dict) else {}
    employer_name = (company_override or employer.get("name") or "").strip() or "EURES Employer"
    legal_id = (reg_id_override or employer.get("legalID") or "UNKNOWN").strip() or "UNKNOWN"

    locations = profile.get("locations") if isinstance(profile.get("locations"), list) else []
    loc_country = ""
    loc_parts: list[str] = []
    if locations and isinstance(locations[0], dict):
        loc_country = str(locations[0].get("countryCode") or "").upper()
        for fld in ("postalCode", "cityName"):
            val = str(locations[0].get(fld) or "").strip()
            if val:
                loc_parts.append(val)

    pes_source = str(data.get("source") or "").upper()
    inferred_country = (
        (country_override or "").strip().upper()
        or loc_country
        or (pes_source[:2] if len(pes_source) >= 2 and pes_source[:2].isalpha() else "EU")
    )

    # Extract contact person, email, phone, and address from personContacts
    contact_person: Optional[str] = None
    contact_email: Optional[str] = None
    contact_phone: Optional[str] = None
    official_address: Optional[str] = None

    contacts = profile.get("personContacts") if isinstance(profile.get("personContacts"), list) else []
    for c in contacts:
        if not isinstance(c, dict):
            continue
        full_name = " ".join(
            p for p in (str(c.get("givenName") or "").strip(), str(c.get("familyName") or "").strip()) if p
        )
        if full_name and not contact_person:
            contact_person = full_name
        comms = c.get("communications") if isinstance(c.get("communications"), dict) else {}
        if not contact_email:
            for em in comms.get("emails") or []:
                uri = str(em.get("uri") or "").replace("mailto:", "").strip() if isinstance(em, dict) else ""
                if "@" in uri:
                    contact_email = uri
                    break
        if not contact_phone:
            for tel_list_key in ("telephoneNumbers", "mobileTelephoneNumbers"):
                for tel in comms.get(tel_list_key) or []:
                    if isinstance(tel, dict):
                        composed = "".join(
                            str(tel.get(k) or "").strip()
                            for k in ("countryDialing", "areaDialing", "dialNumber")
                        )
                        if len(re.sub(r"\D", "", composed)) >= 7:
                            contact_phone = composed
                            break
                if contact_phone:
                    break
        if not official_address:
            for addr in comms.get("addresses") or []:
                if isinstance(addr, dict):
                    addr_lines = [str(a).strip() for a in (addr.get("addressLines") or []) if str(a).strip()]
                    if addr_lines:
                        official_address = ", ".join(addr_lines)
                        break

    if not official_address and loc_parts:
        official_address = " ".join(loc_parts)

    # Determine website_url: prefer employer.website; else infer from non-free corporate email domain
    website_url = str(employer.get("website") or "").strip() or None
    if not website_url and contact_email:
        email_root = extract_root_domain(contact_email)
        if email_root and email_root not in FREE_EMAIL_PROVIDERS:
            website_url = f"https://{email_root}"

    sector_codes = employer.get("sectorCodes") if isinstance(employer.get("sectorCodes"), list) else []
    is_agency = any(str(sc).lower().startswith(("n78", "78")) for sc in sector_codes)

    title = str(profile.get("title") or "").strip()
    desc_clean = _strip_html(str(profile.get("description") or ""))
    app_instr = " ".join(_strip_html(str(x)) for x in (profile.get("applicationInstructions") or []))
    vacancy_lines = [
        f"Official European Commission EURES Portal Vacancy (PES Source: {pes_source or 'EURES'}, Reference: {data.get('reference') or jv_id})",
        f"Title: {title}" if title else "",
        f"Employer: {employer_name}",
        f"Location / Address: {official_address}" if official_address else "",
        f"Description: {desc_clean[:12000]}" if desc_clean else "",
        f"Application Instructions: {app_instr[:2000]}" if app_instr else "",
    ]
    vacancy_text = "\n".join(ln for ln in vacancy_lines if ln)

    req = VerificationRequest(
        company_name=employer_name,
        country_code=inferred_country,
        reg_id=legal_id,
        website_url=website_url,
        contact_person=contact_person,
        contact_email=contact_email,
        contact_phone=contact_phone,
        applicant_type=ApplicantType.agency if is_agency else ApplicantType.direct_employer,
        vacancy_text=vacancy_text[:18000],
        force_refresh=force_refresh,
        eures_meta={
            "verified_official_api": True,
            "pes_source": pes_source,
            "reference": data.get("reference") or jv_id,
            "eures_flag": bool(profile.get("euresFlag")),
            "employer_name": employer_name,
            "official_address": official_address,
            "api_url": api_url,
        },
    )

    if req.country_code == "CZ" and req.reg_id == "UNKNOWN":
        resolved = await search_czech_ares_by_name(req.company_name, client, settings)
        if resolved:
            req = req.model_copy(update={"reg_id": resolved})

    return IntakeExtractionResult(
        request=req,
        sources=[f"eures_api:{jv_id}"],
        extraction_method="eures_official_api",
    )


async def fetch_public_url_text(
    url: str, client: httpx.AsyncClient, settings: Settings
) -> tuple[str, str]:
    """Fetch public unauthenticated job-board HTML or OpenGraph metadata (no cookie scraping)."""
    cleaned_url = url.strip()
    if not _is_safe_public_url(cleaned_url):
        raise ValueError("URL must be a public http(s) address")

    parsed = urlparse(cleaned_url)
    host = (parsed.hostname or "").lower()
    is_social_og = any(d in host for d in ("instagram.com", "facebook.com", "fb.com"))

    headers = (
        {"User-Agent": "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)"}
        if is_social_og
        else {
            "User-Agent": settings.HTTP_USER_AGENT,
            "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        }
    )

    resp = await client.get(
        cleaned_url,
        headers=headers,
        follow_redirects=not is_social_og,
        timeout=httpx.Timeout(12.0),
    )
    if resp.status_code in (403, 429):
        raise ValueError(
            f"Сайт заблокировал автоматическое чтение (HTTP {resp.status_code} / защита от ботов). "
            "Загрузите скриншот вакансии или вставьте её текст в поле ниже."
        )
    if resp.status_code >= 400:
        raise ValueError(f"Не удалось открыть публичную ссылку (HTTP {resp.status_code}).")

    soup = BeautifulSoup(resp.text[:1_500_000], "html.parser")
    og_parts: list[str] = []
    for meta in soup.find_all("meta"):
        prop = (meta.get("property") or meta.get("name") or "").lower()
        content = (meta.get("content") or "").strip()
        if prop in {"og:title", "og:description", "description"} and content:
            if content not in og_parts:
                og_parts.append(content)

    if is_social_og:
        if not og_parts:
            raise ValueError(
                "Social post returned no public OpenGraph metadata (may require login or be private)"
            )
        combined = f"Source URL: {cleaned_url}\n" + "\n".join(og_parts)
        return combined[:MAX_HTML_TEXT_CHARS], f"url:{cleaned_url}"

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    lines = [line.strip() for line in soup.get_text("\n").splitlines() if line.strip()]
    body_text = "\n".join(lines)

    if len(body_text) < 60 and not og_parts:
        raise ValueError(
            "Эта веб-страница рендерится через JavaScript (SPA) и не отдаёт текст вакансии напрямую. "
            "Загрузите скриншот страницы или вставьте текст вакансии в поле ниже."
        )

    parts = [f"Source URL: {str(resp.url)}"]
    if title:
        parts.append(f"Page Title: {title}")
    if og_parts:
        parts.append("Meta: " + " | ".join(og_parts[:2]))
    parts.append(body_text)
    return "\n".join(parts)[:MAX_HTML_TEXT_CHARS], f"url:{str(resp.url)}"


def extract_text_from_docx(data: bytes) -> str:
    """Extract all paragraphs and table cells from a .docx Word file using stdlib zipfile/XML."""
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        xml_bytes = archive.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    paragraphs: list[str] = []
    for para in root.findall(".//w:p", ns):
        chunk = "".join(node.text for node in para.findall(".//w:t", ns) if node.text).strip()
        if chunk:
            paragraphs.append(chunk)
    return "\n".join(paragraphs)[:MAX_DOC_TEXT_CHARS]


def extract_text_from_pdf(data: bytes) -> str:
    """Extract embedded text from all pages of a PDF using pypdf."""
    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for idx, page in enumerate(reader.pages[:30]):
        try:
            txt = (page.extract_text() or "").strip()
        except Exception:
            txt = ""
        if txt:
            pages.append(f"[Page {idx + 1}]\n{txt}")
    return "\n\n".join(pages)[:MAX_DOC_TEXT_CHARS]


def _heuristic_fallback_extract(
    combined_text: str,
    country_override: Optional[str],
    company_override: Optional[str],
    reg_id_override: Optional[str],
    force_refresh: bool,
) -> VerificationRequest:
    """Deterministic regex fallback when ANTHROPIC_API_KEY is absent or LLM extraction fails."""
    email_match = re.search(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b", combined_text)
    phone_match = re.search(r"(?:\+|00)\d{1,3}[\s.\-/()]*\d(?:[\s.\-/()]*\d){6,12}", combined_text)
    url_matches = re.findall(r"https?://[^\s<>\"')]+", combined_text)
    skip_boards = ("layboard.com", "eurabota", "instagram.com", "facebook.com", "linkedin.com", "jobs.cz")
    website = next(
        (u.rstrip(".,;") for u in url_matches if not any(b in u.lower() for b in skip_boards)),
        None,
    )

    ico_match = re.search(r"(?:IČO?|ICO)\s*[:.#]?\s*(\d[\d\s]{6,10}\d)", combined_text, re.IGNORECASE)
    nip_match = re.search(r"(?:NIP)\s*[:.#]?\s*(\d[\d\s\-]{8,13}\d)", combined_text, re.IGNORECASE)
    bin_match = re.search(r"(?:БИН|BIN)\s*[:.#]?\s*(\d{12})", combined_text, re.IGNORECASE)

    inferred_country = (country_override or "").strip().upper()
    inferred_reg_id = (reg_id_override or "").strip()
    if not inferred_reg_id:
        if bin_match:
            inferred_reg_id = bin_match.group(1)
            inferred_country = inferred_country or "KZ"
        elif nip_match:
            inferred_reg_id = re.sub(r"\D", "", nip_match.group(1))
            inferred_country = inferred_country or "PL"
        elif ico_match:
            inferred_reg_id = re.sub(r"\D", "", ico_match.group(1))
            inferred_country = inferred_country or "CZ"
        else:
            inferred_reg_id = "UNKNOWN"

    if not inferred_country:
        lower = combined_text.lower()
        if any(k in lower for k in ("польш", "poland", "warszawa", "zł", "zlecenie", "+48")):
            inferred_country = "PL"
        elif any(k in lower for k in ("герман", "germany", "gmbh", "+49")):
            inferred_country = "DE"
        elif any(k in lower for k in ("austria", "österreich", "steiermark", ".at", "+43")):
            inferred_country = "AT"
        else:
            inferred_country = "CZ"

    inferred_company = (company_override or "").strip()
    if not inferred_company:
        first_line = next((ln.strip() for ln in combined_text.splitlines() if ln.strip()), "")
        inferred_company = first_line[:120] or "Unknown Employer"

    is_agency = any(
        w in combined_text.lower() for w in ("агентств", "agency", "посредник", "zlecenie", "виз")
    )

    return VerificationRequest(
        company_name=inferred_company,
        country_code=inferred_country,
        reg_id=inferred_reg_id,
        website_url=website,
        contact_email=email_match.group(0) if email_match else None,
        contact_phone=phone_match.group(0) if phone_match else None,
        applicant_type=ApplicantType.agency if is_agency else ApplicantType.direct_employer,
        vacancy_text=combined_text[:18_000],
        force_refresh=force_refresh,
    )


def _extract_json_block(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    candidate = fence.group(1) if fence else cleaned
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end > start:
            candidate = candidate[start : end + 1]
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Expected JSON object from intake parser")
    return parsed


async def extract_verification_request(
    settings: Settings,
    url: Optional[str] = None,
    raw_text: Optional[str] = None,
    attachments: Optional[list[UploadedAttachment]] = None,
    country_override: Optional[str] = None,
    company_override: Optional[str] = None,
    reg_id_override: Optional[str] = None,
    force_refresh: bool = False,
    http_client: Optional[httpx.AsyncClient] = None,
    anthropic_client: Optional[AsyncAnthropic] = None,
) -> IntakeExtractionResult:
    """Ingest URL + uploaded DOCX/PDF/images + raw text and produce a VerificationRequest."""
    text_sections: list[str] = []
    sources: list[str] = []
    media_blocks: list[dict[str, Any]] = []
    image_count = 0

    owns_http = http_client is None
    client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(12.0))

    try:
        # 0. Fast-path native EURES Public API (handles Angular SPA URLs with 0 LLM intake tokens!)
        if url and url.strip() and not attachments and not (raw_text and raw_text.strip()):
            eures_result = await try_fetch_eures_vacancy(
                url=url.strip(),
                client=client,
                settings=settings,
                country_override=country_override,
                company_override=company_override,
                reg_id_override=reg_id_override,
                force_refresh=force_refresh,
            )
            if eures_result is not None:
                return eures_result

        # 1. Fetch public URL if provided
        if url and url.strip():
            eures_inline = await try_fetch_eures_vacancy(
                url=url.strip(),
                client=client,
                settings=settings,
                country_override=country_override,
                company_override=company_override,
                reg_id_override=reg_id_override,
                force_refresh=force_refresh,
            )
            if eures_inline is not None:
                text_sections.append(f"=== OFFICIAL EURES VACANCY ===\n{eures_inline.request.vacancy_text}")
                sources.extend(eures_inline.sources)
            else:
                url_text, url_source = await fetch_public_url_text(url.strip(), client, settings)
                text_sections.append(f"=== WEB PAGE ===\n{url_text}")
                sources.append(url_source)

        # 2. Process uploaded attachments (.docx, .pdf, images, .txt)
        for att in attachments or []:
            fname = att.filename or "upload"
            lower_name = fname.lower()
            ext = "." + lower_name.rsplit(".", 1)[-1] if "." in lower_name else ""
            ctype = (att.content_type or "").lower()

            if len(att.data) > settings.INTAKE_MAX_FILE_BYTES:
                raise ValueError(
                    f"File '{fname}' exceeds max allowed size ({settings.INTAKE_MAX_FILE_BYTES // (1024 * 1024)} MB)"
                )

            if ext == ".docx" or "wordprocessingml" in ctype:
                doc_text = extract_text_from_docx(att.data)
                if doc_text:
                    text_sections.append(f"=== DOCX: {fname} ===\n{doc_text}")
                    sources.append(f"docx:{fname}")

            elif ext == ".pdf" or ctype == "application/pdf":
                pdf_text = extract_text_from_pdf(att.data)
                if len(pdf_text) >= 200:
                    text_sections.append(f"=== PDF: {fname} ===\n{pdf_text}")
                    sources.append(f"pdf_text:{fname}")
                elif len(att.data) <= MAX_PDF_BINARY_BYTES:
                    # Slide/scanned PDF with little embedded text -> pass natively to Claude
                    if pdf_text:
                        text_sections.append(f"=== PDF PARTIAL TEXT: {fname} ===\n{pdf_text}")
                    media_blocks.append(
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": base64.standard_b64encode(att.data).decode("ascii"),
                            },
                        }
                    )
                    sources.append(f"pdf_visual:{fname}")

            elif ext in IMAGE_MIME_BY_EXT or ctype.startswith("image/"):
                if image_count >= MAX_IMAGE_ATTACHMENTS:
                    continue
                if len(att.data) > MAX_IMAGE_BYTES:
                    raise ValueError(f"Image '{fname}' exceeds 5 MB limit")
                media_type = IMAGE_MIME_BY_EXT.get(ext) or (
                    ctype if ctype in IMAGE_MIME_BY_EXT.values() else "image/jpeg"
                )
                media_blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64.standard_b64encode(att.data).decode("ascii"),
                        },
                    }
                )
                image_count += 1
                sources.append(f"image:{fname}")

            elif ext in {".txt", ".md", ".csv", ".json"} or ctype.startswith("text/"):
                decoded = att.data.decode("utf-8", errors="replace").strip()
                if decoded:
                    text_sections.append(f"=== TEXT FILE: {fname} ===\n{decoded[:MAX_DOC_TEXT_CHARS]}")
                    sources.append(f"text:{fname}")
            else:
                raise ValueError(
                    f"Unsupported file type '{fname}'. Supported: .docx, .pdf, .png, .jpg, .webp, .txt"
                )

        # 3. Append raw text if provided
        if raw_text and raw_text.strip():
            text_sections.append(f"=== RAW INPUT TEXT ===\n{raw_text.strip()[:MAX_DOC_TEXT_CHARS]}")
            sources.append("raw_text")

        if not text_sections and not media_blocks:
            raise ValueError("Provide at least one input: url, raw_text, or uploaded files")

        combined_text = "\n\n".join(text_sections)
        extraction_method = "regex_fallback"
        req: Optional[VerificationRequest] = None

        # 4. Run Claude multimodal extraction if Anthropic key/client is available
        if anthropic_client is not None or settings.ANTHROPIC_API_KEY:
            owns_anthropic = anthropic_client is None
            ai_client = anthropic_client or AsyncAnthropic(
                api_key=settings.ANTHROPIC_API_KEY,
                timeout=settings.CLAUDE_TIMEOUT_SECONDS,
                max_retries=1,
            )
            try:
                user_content: list[dict[str, Any]] = list(media_blocks)
                prompt_text = (
                    "Extract the structured employer verification fields from all provided materials "
                    "(text, documents, and/or screenshots). Return ONLY the strict JSON object.\n\n"
                    + (combined_text[:18_000] if combined_text else "(See attached images/documents)")
                )
                user_content.append({"type": "text", "text": prompt_text})

                # NEVER pass temperature, top_p, or top_k
                resp = await ai_client.messages.create(
                    model=settings.ANTHROPIC_MODEL,
                    max_tokens=1536,
                    system=INTAKE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
                )
                texts = [
                    getattr(b, "text", "")
                    for b in (getattr(resp, "content", None) or [])
                    if getattr(b, "type", None) == "text"
                ]
                parsed = ExtractedIntakePayload.model_validate(
                    _extract_json_block("\n".join(texts))
                )
                req = VerificationRequest(
                    company_name=(company_override or parsed.company_name).strip(),
                    country_code=(country_override or parsed.country_code).strip().upper(),
                    reg_id=(reg_id_override or parsed.reg_id or "UNKNOWN").strip(),
                    website_url=parsed.website_url,
                    contact_person=parsed.contact_person,
                    contact_email=parsed.contact_email,
                    contact_phone=parsed.contact_phone,
                    applicant_type=parsed.applicant_type,
                    vacancy_text=parsed.vacancy_text or combined_text[:18_000],
                    force_refresh=force_refresh,
                )
                extraction_method = "claude_multimodal"
            except (APITimeoutError, APIError, ValueError, ValidationError) as exc:
                logger.warning("Claude intake extraction failed, falling back to regex: %s", exc)
            finally:
                if owns_anthropic:
                    await ai_client.close()

        if req is None:
            if not combined_text:
                raise ValueError(
                    "Visual/scanned files require ANTHROPIC_API_KEY for multimodal extraction"
                )
            req = _heuristic_fallback_extract(
                combined_text=combined_text,
                country_override=country_override,
                company_override=company_override,
                reg_id_override=reg_id_override,
                force_refresh=force_refresh,
            )

        # 5. Auto-resolve Czech ICO by company name when CZ reg_id is UNKNOWN
        if req.country_code == "CZ" and req.reg_id == "UNKNOWN" and req.company_name:
            resolved_ico = await search_czech_ares_by_name(req.company_name, client, settings)
            if resolved_ico:
                req = req.model_copy(update={"reg_id": resolved_ico})

        return IntakeExtractionResult(
            request=req,
            sources=sources,
            extraction_method=extraction_method,
        )
    finally:
        if owns_http:
            await client.aclose()
