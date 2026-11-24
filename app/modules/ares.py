"""Module A: EU Government Registry verification client (Czech ARES + Poland MF White List)."""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

import httpx

from app.config import Settings
from app.schemas import ModuleResult, ModuleStatus, VerificationRequest

logger = logging.getLogger(__name__)

MAX_SCORE = 25
LIQUIDATION_NAME_MARKERS = ("v likvidaci", "v konkurzu", "v úpadku", "v upadku")
INSOLVENT_STATUS_VALUES = {"AKTIVNI", "INSOLVENCE", "V_INSOLVENCI"}
UNKNOWN_REG_ID_VALUES = {"", "UNKNOWN", "N/A", "NONE", "NULL", "-"}
ANONYMOUS_COMPANY_NAMES = {
    "",
    "UNKNOWN",
    "UNKNOWN EMPLOYER",
    "ANONYMOUS",
    "ANONYMOUS EMPLOYER",
    "N/A",
    "NA",
    "NONE",
    "NULL",
    "NOT PROVIDED",
    "НЕИЗВЕСТНО",
    "НЕИЗВЕСТНЫЙ РАБОТОДАТЕЛЬ",
    "ЧАСТНОЕ ЛИЦО",
    "-",
}


@dataclass
class RegistryCheckOutput:
    """Internal result passed from Module A to the orchestrator and downstream modules."""

    result: ModuleResult
    kill_switches: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    evidence_links: list[str] = field(default_factory=list)
    legal_name: Optional[str] = None
    official_address: Optional[str] = None
    resolved_reg_id: Optional[str] = None
    raw_data: Optional[dict[str, Any]] = None
    registered: bool = False


def _fold_name(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9\s]", " ", stripped.lower()).strip()


def _is_truthy_liquidation(value: Any) -> bool:
    """Detect whether an ARES liquidation field indicates active liquidation."""
    if value is None or value is False:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().upper() not in {"", "NE", "NO", "FALSE", "NEEXISTUJICI", "0"}
    if isinstance(value, (dict, list)):
        return len(value) > 0
    return bool(value)


def _score_company_age(inception_raw: Optional[str]) -> tuple[int, Optional[int], list[str]]:
    """Compute age score (+10 / +5 / 0) and flags from an ISO YYYY-MM-DD inception date."""
    if not isinstance(inception_raw, str) or len(inception_raw) < 10:
        return 0, None, []
    try:
        inception_date = date.fromisoformat(inception_raw[:10])
    except ValueError:
        return 0, None, []

    age_days = max(0, (date.today() - inception_date).days)
    if age_days >= 730:
        return 10, age_days, []
    if age_days >= 180:
        return 5, age_days, ["YOUNG_COMPANY"]
    return 0, age_days, ["VERY_NEW_COMPANY"]


def _evaluate_czech_ares_payload(
    ico: str, data: dict[str, Any], evidence_url: str
) -> RegistryCheckOutput:
    """Deterministically score a parsed Czech ARES v3 JSON payload (max 25 pts)."""
    legal_name = str(data.get("obchodniJmeno") or "").strip() or None
    sidlo = data.get("sidlo") if isinstance(data.get("sidlo"), dict) else {}
    official_address = str(sidlo.get("textovaAdresa") or "").strip() or None
    registrations = (
        data.get("seznamRegistraci") if isinstance(data.get("seznamRegistraci"), dict) else {}
    )

    stav_ir = str(registrations.get("stavZdrojeIr") or "").strip().upper()
    stav_ros = str(registrations.get("stavZdrojeRos") or "").strip().upper()
    stav_vr = str(registrations.get("stavZdrojeVr") or "").strip().upper()
    datum_zaniku = data.get("datumZaniku")

    is_insolvent = stav_ir in INSOLVENT_STATUS_VALUES
    is_liquidated_field = _is_truthy_liquidation(data.get("likvidace")) or _is_truthy_liquidation(
        registrations.get("likvidace")
    )
    name_lower = (legal_name or "").lower()
    is_liquidated_name = any(marker in name_lower for marker in LIQUIDATION_NAME_MARKERS)
    is_dissolved = bool(datum_zaniku) or (
        stav_ros == "ZANIKLY" or (stav_vr == "ZANIKLY" and stav_ros != "AKTIVNI")
    )

    if is_insolvent or is_liquidated_field or is_liquidated_name or is_dissolved:
        details: dict[str, Any] = {
            "registry": "CZ_ARES",
            "ico": ico,
            "legal_name": legal_name,
            "official_address": official_address,
            "datum_vzniku": data.get("datumVzniku"),
            "datum_zaniku": datum_zaniku,
            "is_active": False,
            "is_insolvent": is_insolvent,
            "is_liquidated": is_liquidated_field or is_liquidated_name or is_dissolved,
            "stav_zdroje_ir": stav_ir or None,
            "stav_zdroje_ros": stav_ros or None,
            "stav_zdroje_vr": stav_vr or None,
            "score_breakdown": {"active": 0, "age": 0, "cz_nace": 0},
        }
        return RegistryCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.SUCCESS,
                score=0,
                max_score=MAX_SCORE,
                details=details,
            ),
            kill_switches=["COMPANY_LIQUIDATED_OR_INSOLVENT"],
            evidence_links=[evidence_url],
            legal_name=legal_name,
            official_address=official_address,
            resolved_reg_id=ico,
            raw_data=data,
            registered=True,
        )

    # 1. Active: +10 pts
    active_score = 10

    # 2. Company age from datumVzniku: +10 / +5 / 0
    datum_vzniku_raw = data.get("datumVzniku")
    age_score, age_days, red_flags = _score_company_age(datum_vzniku_raw)

    # 3. Valid czNace business activities present: +5 pts
    raw_nace = data.get("czNace") or data.get("czNace2008") or []
    valid_nace = (
        [str(code).strip() for code in raw_nace if str(code).strip()]
        if isinstance(raw_nace, list)
        else []
    )
    nace_score = 5 if len(valid_nace) > 0 else 0

    total_score = min(MAX_SCORE, active_score + age_score + nace_score)
    details = {
        "registry": "CZ_ARES",
        "ico": ico,
        "legal_name": legal_name,
        "official_address": official_address,
        "datum_vzniku": datum_vzniku_raw,
        "age_days": age_days,
        "is_active": True,
        "is_insolvent": False,
        "is_liquidated": False,
        "cz_nace_count": len(valid_nace),
        "cz_nace_sample": valid_nace[:10],
        "dic": data.get("dic"),
        "stav_zdroje_ros": stav_ros or None,
        "stav_zdroje_vr": stav_vr or None,
        "stav_zdroje_ir": stav_ir or None,
        "score_breakdown": {"active": active_score, "age": age_score, "cz_nace": nace_score},
    }
    return RegistryCheckOutput(
        result=ModuleResult(
            status=ModuleStatus.SUCCESS,
            score=total_score,
            max_score=MAX_SCORE,
            details=details,
        ),
        red_flags=red_flags,
        evidence_links=[evidence_url],
        legal_name=legal_name,
        official_address=official_address,
        resolved_reg_id=ico,
        raw_data=data,
        registered=True,
    )


async def search_czech_ares_by_name(
    company_name: str,
    client: httpx.AsyncClient,
    settings: Settings,
) -> Optional[str]:
    """Resolve an 8-digit Czech ICO from a company name using ARES public search."""
    cleaned = company_name.strip()
    if len(cleaned) < 3 or cleaned.upper() in ANONYMOUS_COMPANY_NAMES:
        return None
    search_url = f"{settings.ARES_BASE_URL.rstrip('/')}/vyhledat"
    # Strip Czech legal suffixes/country fluff for cleaner ARES search hits
    query_name = re.sub(
        r"\b(česká republika|ceska republika|czech republic|czechia)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" ,.-")
    query_name = query_name or cleaned

    try:
        response = await client.post(
            search_url,
            json={"obchodniJmeno": query_name, "pocet": 10},
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": settings.HTTP_USER_AGENT,
            },
            timeout=httpx.Timeout(min(5.0, settings.MODULE_TIMEOUT_SECONDS)),
        )
    except httpx.HTTPError as exc:
        logger.debug("ARES name search error for %r: %s", query_name, exc)
        return None

    if response.status_code != 200:
        return None

    try:
        subjects = response.json().get("ekonomickeSubjekty") or []
    except ValueError:
        return None

    if not isinstance(subjects, list) or not subjects:
        return None

    query_folded = _fold_name(query_name)
    target_tokens = {t for t in query_folded.split() if len(t) >= 3}
    best_ico: Optional[str] = None
    best_rank: tuple[int, int, int, str] = (-1, -1, -1, "9999-99-99")

    for item in subjects:
        if not isinstance(item, dict):
            continue
        ico = str(item.get("ico") or "").strip()
        name = str(item.get("obchodniJmeno") or "")
        if not re.fullmatch(r"\d{8}", ico):
            continue
        regs = item.get("seznamRegistraci") if isinstance(item.get("seznamRegistraci"), dict) else {}
        if str(regs.get("stavZdrojeRos", "")).upper() == "ZANIKLY":
            continue
        name_folded = _fold_name(name)
        name_tokens = {t for t in name_folded.split() if len(t) >= 3}
        overlap = len(target_tokens & name_tokens)
        if not overlap and target_tokens:
            if any(tok in name_folded for tok in target_tokens):
                overlap = 1
        if overlap <= 0:
            continue

        # Require brand prefix match so searching "Manpower" never falsely resolves an unrelated
        # company like "NARA manpower s.r.o." that merely contains the word in the middle.
        first_q_tok = query_folded.split()[0] if query_folded else ""
        starts_with_brand = 1 if (first_q_tok and name_folded.startswith(first_q_tok)) else 0
        if starts_with_brand == 0:
            continue
        exact_token_match = 1 if (first_q_tok and first_q_tok in name_tokens) else 0
        inception = str(item.get("datumVzniku") or "9999-99-99")[:10]
        candidate_rank = (starts_with_brand, exact_token_match, overlap, inception)
        if best_ico is None or (
            candidate_rank[:3] > best_rank[:3]
            or (candidate_rank[:3] == best_rank[:3] and candidate_rank[3] < best_rank[3])
        ):
            best_rank = candidate_rank
            best_ico = ico

    return best_ico


async def _check_czech_ares(
    request: VerificationRequest,
    client: httpx.AsyncClient,
    settings: Settings,
) -> RegistryCheckOutput:
    """Query the public Czech ARES REST API for an 8-digit ICO (with optional name auto-resolve)."""
    ico = request.reg_id.strip().upper()
    if ico in UNKNOWN_REG_ID_VALUES:
        resolved = await search_czech_ares_by_name(request.company_name, client, settings)
        if resolved:
            logger.info("Auto-resolved Czech ICO=%s for company=%r", resolved, request.company_name)
            ico = resolved
        else:
            return RegistryCheckOutput(
                result=ModuleResult(
                    status=ModuleStatus.UNAVAILABLE,
                    score=0,
                    max_score=MAX_SCORE,
                    details={
                        "registry": "CZ_ARES",
                        "reason": "No 8-digit Czech ICO provided and ARES name search found no match",
                    },
                )
            )

    url = f"{settings.ARES_BASE_URL.rstrip('/')}/{ico}"
    if not re.fullmatch(r"\d{8}", ico):
        logger.info("Invalid Czech ICO format reg_id=%s -> UNREGISTERED_ENTITY", ico)
        return RegistryCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.SUCCESS,
                score=0,
                max_score=MAX_SCORE,
                details={
                    "registry": "CZ_ARES",
                    "ico": ico,
                    "found": False,
                    "reason": "Invalid Czech ICO format (expected 8 digits)",
                },
            ),
            kill_switches=["UNREGISTERED_ENTITY"],
            registered=False,
        )

    try:
        response = await client.get(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": settings.HTTP_USER_AGENT,
            },
            timeout=httpx.Timeout(settings.MODULE_TIMEOUT_SECONDS),
        )
    except httpx.TimeoutException:
        logger.warning("ARES request timed out for ico=%s", ico)
        return RegistryCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.UNAVAILABLE,
                score=0,
                max_score=MAX_SCORE,
                details={"registry": "CZ_ARES", "ico": ico, "error": "Czech ARES API request timed out"},
            )
        )
    except httpx.HTTPError as exc:
        logger.warning("ARES HTTP error for ico=%s: %s", ico, exc)
        return RegistryCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.UNAVAILABLE,
                score=0,
                max_score=MAX_SCORE,
                details={"registry": "CZ_ARES", "ico": ico, "error": f"ARES network error: {type(exc).__name__}"},
            )
        )

    if response.status_code in (400, 404):
        logger.info("ARES returned %s for ico=%s -> UNREGISTERED_ENTITY", response.status_code, ico)
        return RegistryCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.SUCCESS,
                score=0,
                max_score=MAX_SCORE,
                details={
                    "registry": "CZ_ARES",
                    "ico": ico,
                    "found": False,
                    "http_status": response.status_code,
                    "reason": "Entity not found in Czech ARES registry",
                },
            ),
            kill_switches=["UNREGISTERED_ENTITY"],
            evidence_links=[url] if response.status_code == 404 else [],
            registered=False,
        )

    if response.status_code != 200:
        logger.warning("ARES returned HTTP %s for ico=%s", response.status_code, ico)
        return RegistryCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.UNAVAILABLE,
                score=0,
                max_score=MAX_SCORE,
                details={
                    "registry": "CZ_ARES",
                    "ico": ico,
                    "http_status": response.status_code,
                    "error": f"Unexpected ARES status code {response.status_code}",
                },
            )
        )

    try:
        payload = response.json()
    except ValueError:
        return RegistryCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.FAILED,
                score=0,
                max_score=MAX_SCORE,
                details={"registry": "CZ_ARES", "ico": ico, "error": "Invalid JSON returned by ARES"},
            )
        )

    if not isinstance(payload, dict):
        return RegistryCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.FAILED,
                score=0,
                max_score=MAX_SCORE,
                details={"registry": "CZ_ARES", "ico": ico, "error": "Unexpected non-object ARES payload"},
            )
        )

    return _evaluate_czech_ares_payload(ico=ico, data=payload, evidence_url=url)


def _evaluate_poland_mf_payload(
    nip: str, subject: dict[str, Any], evidence_url: str
) -> RegistryCheckOutput:
    """Deterministically score a Polish Ministry of Finance White List subject (max 25 pts)."""
    legal_name = str(subject.get("name") or "").strip() or None
    official_address = (
        str(subject.get("residenceAddress") or subject.get("workingAddress") or "").strip() or None
    )
    status_vat = str(subject.get("statusVat") or "").strip()
    removal_date = subject.get("removalDate")
    restoration_date = subject.get("restorationDate")

    # Removed from register without subsequent restoration -> dissolved/removed
    is_removed = bool(removal_date) and (
        not restoration_date or str(restoration_date) < str(removal_date)
    )
    name_lower = (legal_name or "").lower()
    is_liquidated_name = "w likwidacji" in name_lower or "w upadłości" in name_lower

    if is_removed or is_liquidated_name:
        return RegistryCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.SUCCESS,
                score=0,
                max_score=MAX_SCORE,
                details={
                    "registry": "PL_MF_WHITELIST",
                    "nip": nip,
                    "legal_name": legal_name,
                    "official_address": official_address,
                    "status_vat": status_vat or None,
                    "removal_date": removal_date,
                    "is_active": False,
                    "is_liquidated": True,
                    "score_breakdown": {"active": 0, "age": 0, "krs_regon": 0},
                },
            ),
            kill_switches=["COMPANY_LIQUIDATED_OR_INSOLVENT"],
            evidence_links=[evidence_url],
            legal_name=legal_name,
            official_address=official_address,
            resolved_reg_id=nip,
            raw_data=subject,
            registered=True,
        )

    active_score = 10
    reg_date = subject.get("registrationLegalDate")
    age_score, age_days, red_flags = _score_company_age(reg_date)
    krs = str(subject.get("krs") or "").strip() or None
    regon = str(subject.get("regon") or "").strip() or None
    id_score = 5 if (krs or regon) else 0

    total_score = min(MAX_SCORE, active_score + age_score + id_score)
    details = {
        "registry": "PL_MF_WHITELIST",
        "nip": nip,
        "krs": krs,
        "regon": regon,
        "legal_name": legal_name,
        "official_address": official_address,
        "status_vat": status_vat or None,
        "registration_legal_date": reg_date,
        "age_days": age_days,
        "is_active": True,
        "score_breakdown": {"active": active_score, "age": age_score, "krs_regon": id_score},
    }
    return RegistryCheckOutput(
        result=ModuleResult(
            status=ModuleStatus.SUCCESS,
            score=total_score,
            max_score=MAX_SCORE,
            details=details,
        ),
        red_flags=red_flags,
        evidence_links=[evidence_url],
        legal_name=legal_name,
        official_address=official_address,
        resolved_reg_id=nip,
        raw_data=subject,
        registered=True,
    )


async def _check_poland_mf_nip(
    request: VerificationRequest,
    client: httpx.AsyncClient,
    settings: Settings,
) -> RegistryCheckOutput:
    """Query Poland's official free Ministry of Finance White List REST API by 10-digit NIP."""
    nip = request.reg_id.strip().upper()
    if nip in UNKNOWN_REG_ID_VALUES:
        return RegistryCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.UNAVAILABLE,
                score=0,
                max_score=MAX_SCORE,
                details={
                    "registry": "PL_MF_WHITELIST",
                    "reason": "No 10-digit Polish NIP provided",
                },
            )
        )

    if not re.fullmatch(r"\d{10}", nip):
        return RegistryCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.SUCCESS,
                score=0,
                max_score=MAX_SCORE,
                details={
                    "registry": "PL_MF_WHITELIST",
                    "nip": nip,
                    "found": False,
                    "reason": "Invalid Polish NIP format (expected 10 digits)",
                },
            ),
            kill_switches=["UNREGISTERED_ENTITY"],
            registered=False,
        )

    today_str = date.today().isoformat()
    url = f"{settings.POLAND_WL_BASE_URL.rstrip('/')}/{nip}?date={today_str}"
    try:
        response = await client.get(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": settings.HTTP_USER_AGENT,
            },
            timeout=httpx.Timeout(settings.MODULE_TIMEOUT_SECONDS),
        )
    except httpx.TimeoutException:
        return RegistryCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.UNAVAILABLE,
                score=0,
                max_score=MAX_SCORE,
                details={"registry": "PL_MF_WHITELIST", "nip": nip, "error": "Poland MF API timed out"},
            )
        )
    except httpx.HTTPError as exc:
        return RegistryCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.UNAVAILABLE,
                score=0,
                max_score=MAX_SCORE,
                details={
                    "registry": "PL_MF_WHITELIST",
                    "nip": nip,
                    "error": f"Poland MF network error: {type(exc).__name__}",
                },
            )
        )

    if response.status_code in (400, 404):
        return RegistryCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.SUCCESS,
                score=0,
                max_score=MAX_SCORE,
                details={
                    "registry": "PL_MF_WHITELIST",
                    "nip": nip,
                    "found": False,
                    "http_status": response.status_code,
                    "reason": "Entity not found in Poland Ministry of Finance registry",
                },
            ),
            kill_switches=["UNREGISTERED_ENTITY"],
            evidence_links=[url],
            registered=False,
        )

    if response.status_code != 200:
        return RegistryCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.UNAVAILABLE,
                score=0,
                max_score=MAX_SCORE,
                details={
                    "registry": "PL_MF_WHITELIST",
                    "nip": nip,
                    "http_status": response.status_code,
                    "error": f"Unexpected Poland MF status {response.status_code}",
                },
            )
        )

    try:
        payload = response.json()
    except ValueError:
        return RegistryCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.FAILED,
                score=0,
                max_score=MAX_SCORE,
                details={"registry": "PL_MF_WHITELIST", "nip": nip, "error": "Invalid JSON from Poland MF"},
            )
        )

    subject = (payload.get("result") or {}).get("subject") if isinstance(payload, dict) else None
    if not isinstance(subject, dict):
        return RegistryCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.SUCCESS,
                score=0,
                max_score=MAX_SCORE,
                details={
                    "registry": "PL_MF_WHITELIST",
                    "nip": nip,
                    "found": False,
                    "reason": "No registered entity found for Polish NIP",
                },
            ),
            kill_switches=["UNREGISTERED_ENTITY"],
            evidence_links=[url],
            registered=False,
        )

    return _evaluate_poland_mf_payload(nip=nip, subject=subject, evidence_url=url)


async def check_gov_registry(
    request: VerificationRequest,
    client: httpx.AsyncClient,
    settings: Settings,
) -> RegistryCheckOutput:
    """Dispatch government registry verification by ISO country_code (CZ ARES, PL MF White List, EU EURES PES)."""
    country = request.country_code.upper()
    if country == "CZ":
        return await _check_czech_ares(request, client=client, settings=settings)
    if country == "PL":
        return await _check_poland_mf_nip(request, client=client, settings=settings)

    # Verified directly from the official European Commission EURES Public Employment Service API
    if isinstance(request.eures_meta, dict) and request.eures_meta.get("verified_official_api"):
        pes_source = str(request.eures_meta.get("pes_source") or "EURES")
        reference = str(request.eures_meta.get("reference") or "")
        legal_name = str(request.eures_meta.get("employer_name") or request.company_name).strip()
        official_address = str(request.eures_meta.get("official_address") or "").strip() or None
        api_url = str(request.eures_meta.get("api_url") or "")
        return RegistryCheckOutput(
            result=ModuleResult(
                status=ModuleStatus.SUCCESS,
                score=20,
                max_score=MAX_SCORE,
                details={
                    "registry": "EU_EURES_PES",
                    "country_code": country,
                    "pes_source": pes_source,
                    "eures_reference": reference,
                    "legal_name": legal_name,
                    "official_address": official_address,
                    "is_active": True,
                    "note": (
                        "Verified directly via official European Commission EURES Public Employment "
                        "Service API (20/25; direct national commercial register insolvency check N/A)."
                    ),
                    "score_breakdown": {"pes_verified_employer": 20},
                },
            ),
            evidence_links=[api_url] if api_url.startswith("http") else [],
            legal_name=legal_name,
            official_address=official_address,
            registered=False,
        )

    logger.info("Government registry check unavailable for country=%s", country)
    return RegistryCheckOutput(
        result=ModuleResult(
            status=ModuleStatus.UNAVAILABLE,
            score=0,
            max_score=MAX_SCORE,
            details={
                "country_code": country,
                "reg_id": request.reg_id,
                "reason": f"Government registry adapter not yet implemented for country '{country}'",
            },
        )
    )
