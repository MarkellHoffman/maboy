"""Pydantic v2 API contracts, Claude OSINT structured output schema, and input normalizers."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlparse
from uuid import UUID

import tldextract
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Offline PSL extractor (never performs network calls or writes cache files)
_TLD_EXTRACTOR = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=())


class ApplicantType(str, Enum):
    direct_employer = "direct_employer"
    agency = "agency"


class VerificationStatus(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


class ModuleStatus(str, Enum):
    SUCCESS = "SUCCESS"
    UNAVAILABLE = "UNAVAILABLE"
    FAILED = "FAILED"


class BlacklistEntityType(str, Enum):
    reg_id = "reg_id"
    domain = "domain"
    phone = "phone"
    email = "email"


# ---------------------------------------------------------------------------
# Deterministic Normalization Utilities
# ---------------------------------------------------------------------------


def normalize_reg_id(reg_id: str, country_code: str = "CZ") -> str:
    """Strip whitespace/dashes from registration ID; zero-pad Czech ICOs and clean Polish NIPs."""
    cleaned = re.sub(r"\s+", "", (reg_id or "").strip())
    cc = country_code.upper()
    if cc == "CZ" and cleaned.isdigit() and 1 <= len(cleaned) <= 8:
        return cleaned.zfill(8)
    if cc == "PL":
        stripped_pl = re.sub(r"^PL", "", cleaned.upper()).replace("-", "")
        if stripped_pl.isdigit() and len(stripped_pl) == 10:
            return stripped_pl
    return cleaned.upper()


def extract_root_domain(value: Optional[str]) -> str:
    """Extract canonical lower-case root domain (e.g., 'example.cz') from a URL, host, or email."""
    if not value:
        return ""
    raw = value.strip().lower()
    if not raw:
        return ""
    if "@" in raw and "://" not in raw:
        raw = raw.rsplit("@", 1)[-1]
    if "://" in raw:
        parsed = urlparse(raw)
        host = parsed.hostname or ""
    else:
        host = raw.split("/", 1)[0].split(":", 1)[0]
    host = host.strip(".")
    if not host:
        return ""
    ext = _TLD_EXTRACTOR(host)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}".lower()
    return ext.domain.lower() or host


def normalize_email(email: Optional[str]) -> str:
    """Return lower-cased, whitespace-stripped email address."""
    return (email or "").strip().lower()


def normalize_phone_e164(phone: Optional[str], country_code: str = "CZ") -> str:
    """Normalize a phone number to canonical E.164 format (best effort for CZ and EU numbers)."""
    if not phone:
        return ""
    raw = phone.strip()
    if not raw:
        return ""
    has_plus = raw.startswith("+")
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return ""
    if has_plus:
        return f"+{digits}"
    if digits.startswith("00") and len(digits) > 4:
        return f"+{digits[2:]}"
    if country_code.upper() == "CZ":
        if len(digits) == 9:
            return f"+420{digits}"
        if len(digits) == 12 and digits.startswith("420"):
            return f"+{digits}"
    return f"+{digits}"


def normalize_blacklist_value(
    entity_type: BlacklistEntityType, value: str, country_code: str = "CZ"
) -> str:
    """Normalize a blacklist value according to its entity_type."""
    if entity_type == BlacklistEntityType.reg_id:
        return normalize_reg_id(value, country_code=country_code)
    if entity_type == BlacklistEntityType.domain:
        return extract_root_domain(value)
    if entity_type == BlacklistEntityType.email:
        return normalize_email(value)
    if entity_type == BlacklistEntityType.phone:
        return normalize_phone_e164(value, country_code=country_code)
    return value.strip().lower()


def normalize_website_url(url: Optional[str]) -> Optional[str]:
    """Ensure a non-empty website URL has a valid http(s) scheme and stripped whitespace."""
    if url is None:
        return None
    cleaned = url.strip()
    if not cleaned:
        return None
    if not re.match(r"^https?://", cleaned, flags=re.IGNORECASE):
        cleaned = f"https://{cleaned}"
    return cleaned


def compute_input_hash(request: "VerificationRequest") -> str:
    """Compute deterministic SHA-256 hash of normalized request fields for cache lookup."""
    payload = {
        "company_name": re.sub(r"\s+", " ", request.company_name.strip().lower()),
        "country_code": request.country_code.upper(),
        "reg_id": normalize_reg_id(request.reg_id, request.country_code),
        "website_url": (request.website_url or "").strip().lower().rstrip("/"),
        "contact_email": normalize_email(request.contact_email),
        "contact_phone": normalize_phone_e164(request.contact_phone, request.country_code),
        "applicant_type": request.applicant_type.value,
        "vacancy_text": re.sub(r"\s+", " ", (request.vacancy_text or "").strip()),
    }
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Request / Response Contracts
# ---------------------------------------------------------------------------


class VerificationRequest(BaseModel):
    """Input payload for POST /api/v1/verify."""

    company_name: str = Field(..., min_length=1, max_length=255)
    country_code: str = Field(default="CZ", min_length=2, max_length=2)
    reg_id: str = Field(..., min_length=1, max_length=32, description="8-digit Czech ICO")
    website_url: Optional[str] = Field(default=None, max_length=512)
    contact_person: Optional[str] = Field(default=None, max_length=128)
    contact_email: Optional[str] = Field(default=None, max_length=255)
    contact_phone: Optional[str] = Field(default=None, max_length=64)
    applicant_type: ApplicantType = Field(default=ApplicantType.direct_employer)
    vacancy_text: str = Field(default="", max_length=20000)
    force_refresh: bool = Field(default=False)
    eures_meta: Optional[dict[str, Any]] = Field(default=None, exclude=True)

    @field_validator("company_name", mode="before")
    @classmethod
    def _strip_company_name(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("company_name must be a non-empty string")
        return v.strip()

    @field_validator("country_code", mode="before")
    @classmethod
    def _upper_country(cls, v: Any) -> str:
        if not isinstance(v, str) or len(v.strip()) != 2:
            raise ValueError("country_code must be a 2-letter ISO code")
        return v.strip().upper()

    @field_validator("website_url", mode="before")
    @classmethod
    def _clean_website(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        return normalize_website_url(str(v))

    @field_validator("contact_person", mode="before")
    @classmethod
    def _clean_contact_person(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        cleaned = str(v).strip()
        return cleaned or None

    @field_validator("contact_email", mode="before")
    @classmethod
    def _clean_email(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        cleaned = normalize_email(str(v))
        return cleaned or None

    @field_validator("contact_phone", mode="before")
    @classmethod
    def _clean_phone(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        cleaned = str(v).strip()
        return cleaned or None

    @model_validator(mode="after")
    def _normalize_country_dependent_fields(self) -> "VerificationRequest":
        self.reg_id = normalize_reg_id(self.reg_id, self.country_code)
        if not self.reg_id:
            raise ValueError("reg_id cannot be empty after stripping whitespace")
        if self.contact_phone:
            self.contact_phone = normalize_phone_e164(self.contact_phone, self.country_code)
        return self


class ModuleResult(BaseModel):
    """Per-module verification result returned in the response payload."""

    status: ModuleStatus
    score: int = Field(ge=0)
    max_score: int = Field(ge=0)
    details: dict[str, Any] = Field(default_factory=dict)


class ModulesOutput(BaseModel):
    """3-module map totaling 100 pts (gov_registry 25 + domain_site 25 + claude_osint 50)."""

    gov_registry: ModuleResult
    domain_site: ModuleResult
    claude_osint: ModuleResult


class VerificationResponse(BaseModel):
    """Strict output payload for POST /api/v1/verify and GET /api/v1/verify/{verification_id}."""

    verification_id: UUID
    cached: bool = False
    status: VerificationStatus
    trust_score: int = Field(ge=0, le=100)
    confidence_score: float = Field(ge=0.0, le=1.0)
    kill_switches_triggered: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    evidence_links: list[str] = Field(default_factory=list)
    summary: str
    modules: ModulesOutput


class ClaudeOSINTAuditOutput(BaseModel):
    """Strict Pydantic-validated structured JSON returned by unified Claude OSINT Auditor (max 50 pts).

    Breakdown:
    - maps_reviews_score: 0..20 (physical presence, address/website match, ratings & review safety)
    - web_osint_score:    0..15 (LinkedIn company profile, phone/company scam check, virtual office)
    - vacancy_audit_score: 0..15 (upfront payment check, salary realism, urgency, messenger funnel)
    """

    maps_reviews_score: int
    web_osint_score: int
    vacancy_audit_score: int

    # Physical / Google Maps / Reviews
    maps_presence_found: bool = False
    maps_address_or_website_match: bool = False
    maps_rating: Optional[float] = None
    maps_reviews_count: Optional[int] = None
    maps_reviews_scam_keywords: bool = False
    google_maps_url: Optional[str] = None

    # LinkedIn & Web Scam OSINT
    linkedin_company_url: Optional[str] = None
    negative_osint_reports_found: bool = False
    virtual_office_detected: bool = False

    # Vacancy & Reviews Fraud Audit
    requires_upfront_payment: bool = False
    unrealistic_salary_or_promises: bool = False
    artificial_urgency_detected: bool = False
    shady_messenger_only_funnel: bool = False
    reviews_indicate_fraud: bool = False

    # Conditional Recruiter / Phone Verification (populated only when unverified on site)
    contact_verification_status: str = "NOT_NEEDED"
    contact_verification_evidence: Optional[str] = None
    official_verification_contact: Optional[str] = None
    verification_message_local: Optional[str] = None
    user_action_advice: Optional[str] = None

    # Evidence & Explanations
    cited_urls: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    reasoning: str

    @field_validator("maps_reviews_score", mode="after")
    @classmethod
    def _clamp_maps_score(cls, v: int) -> int:
        return max(0, min(20, int(v)))

    @field_validator("web_osint_score", mode="after")
    @classmethod
    def _clamp_osint_score(cls, v: int) -> int:
        return max(0, min(15, int(v)))

    @field_validator("vacancy_audit_score", mode="after")
    @classmethod
    def _clamp_vacancy_score(cls, v: int) -> int:
        return max(0, min(15, int(v)))


class BlacklistCreateRequest(BaseModel):
    """Input payload for POST /api/v1/blacklist."""

    entity_type: BlacklistEntityType
    value: str = Field(..., min_length=1, max_length=512)
    reason: str = Field(..., min_length=1, max_length=2000)

    @field_validator("reason", mode="before")
    @classmethod
    def _strip_reason(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("reason must be a non-empty string")
        return v.strip()

    @model_validator(mode="after")
    def _normalize_value(self) -> "BlacklistCreateRequest":
        normalized = normalize_blacklist_value(self.entity_type, self.value)
        if not normalized:
            raise ValueError(f"Invalid value for entity_type={self.entity_type.value}")
        self.value = normalized
        return self


class BlacklistResponse(BaseModel):
    """Output payload for POST /api/v1/blacklist."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    entity_type: BlacklistEntityType
    value: str
    reason: str
    created_at: datetime


class HealthResponse(BaseModel):
    """Output payload for GET /health."""

    status: str
    database: str
    database_backend: str
    version: str
    configured_integrations: dict[str, bool]


class VerificationHistoryItem(BaseModel):
    """Summary row for GET /api/v1/verifications."""

    verification_id: UUID
    company_name: Optional[str] = None
    country_code: Optional[str] = None
    reg_id: Optional[str] = None
    status: VerificationStatus
    trust_score: int = Field(ge=0, le=100)
    confidence_score: float = Field(ge=0.0, le=1.0)
    kill_switches_triggered: list[str] = Field(default_factory=list)
    red_flags_count: int = 0
    summary: str
    created_at: datetime


class IntakeVerificationResponse(BaseModel):
    """Output payload for POST /api/v1/verify/intake (URL / files / screenshots / raw text)."""

    extracted_input: VerificationRequest
    intake_sources: list[str] = Field(default_factory=list)
    extraction_method: str
    verification: VerificationResponse
