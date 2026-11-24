"""FastAPI application entrypoint, Stage 2 Multimodal Intake, History, Blacklist & Web UI."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import desc, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import configure_logging, get_settings
from app.database import check_db_health, engine, get_db, init_db
from app.models import Blacklist, VerificationLog
from app.modules.intake import UploadedAttachment, extract_verification_request
from app.orchestrator import log_to_response, run_verification_pipeline
from app.schemas import (
    BlacklistCreateRequest,
    BlacklistResponse,
    HealthResponse,
    IntakeVerificationResponse,
    VerificationHistoryItem,
    VerificationRequest,
    VerificationResponse,
    VerificationStatus,
)

configure_logging()
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Initialize SQL schema on startup and dispose DB connections on shutdown."""
    cfg = get_settings()
    logger.info(
        "Starting %s v%s (db_backend=%s, model=%s, integrations=%s)",
        cfg.APP_NAME,
        cfg.APP_VERSION,
        cfg.database_backend,
        cfg.ANTHROPIC_MODEL,
        cfg.configured_integrations,
    )
    await init_db()
    yield
    await engine.dispose()
    logger.info("Shutdown complete")


app = FastAPI(
    title=get_settings().APP_NAME,
    version=get_settings().APP_VERSION,
    description=(
        "Fault-tolerant anti-fraud OSINT & EU government registry verification engine "
        "(Stage 2: Czech ARES + Poland MF NIP + WHOIS/Crawler + Claude Web Search OSINT + "
        "Multimodal URL/DOCX/PDF/Screenshot Intake + Web Dashboard)."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Structured request/response latency logging."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "HTTP %s %s -> %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.get("/", include_in_schema=False)
async def serve_dashboard() -> FileResponse:
    """Serve the built-in Stage 2 Anti-Fraud Verification Dashboard UI."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dashboard UI not found")
    return FileResponse(index_path, media_type="text/html; charset=utf-8")


@app.post(
    "/api/v1/verify",
    response_model=VerificationResponse,
    summary="Verify an employer or recruiting agency (structured JSON)",
    tags=["Verification"],
)
async def verify_employer(
    payload: VerificationRequest,
    db: AsyncSession = Depends(get_db),
) -> VerificationResponse:
    """Run pre-flight blacklist/cache, free Modules A & B, and unified Claude OSINT audit."""
    return await run_verification_pipeline(request=payload, db=db, settings=get_settings())


@app.post(
    "/api/v1/verify/intake",
    response_model=IntakeVerificationResponse,
    summary="Multimodal intake & verification (URL, DOCX, PDF, Screenshots, or raw text)",
    tags=["Verification"],
)
async def verify_employer_intake(
    url: Optional[str] = Form(default=None),
    raw_text: Optional[str] = Form(default=None),
    country_code: Optional[str] = Form(default=None),
    company_name: Optional[str] = Form(default=None),
    reg_id: Optional[str] = Form(default=None),
    force_refresh: bool = Form(default=False),
    files: Optional[list[UploadFile]] = File(default=None),
    db: AsyncSession = Depends(get_db),
) -> IntakeVerificationResponse:
    """Extract structured employer data from URLs/files/screenshots and run full verification."""
    cfg = get_settings()
    attachments: list[UploadedAttachment] = []
    for upload in files or []:
        data = await upload.read()
        if not data:
            continue
        attachments.append(
            UploadedAttachment(
                filename=upload.filename or "upload",
                content_type=upload.content_type or "application/octet-stream",
                data=data,
            )
        )

    try:
        intake_result = await extract_verification_request(
            settings=cfg,
            url=url,
            raw_text=raw_text,
            attachments=attachments,
            country_override=country_code,
            company_override=company_name,
            reg_id_override=reg_id,
            force_refresh=force_refresh,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    verification = await run_verification_pipeline(
        request=intake_result.request,
        db=db,
        settings=cfg,
    )
    return IntakeVerificationResponse(
        extracted_input=intake_result.request,
        intake_sources=intake_result.sources,
        extraction_method=intake_result.extraction_method,
        verification=verification,
    )


@app.get(
    "/api/v1/verifications",
    response_model=list[VerificationHistoryItem],
    summary="List recent verification reports",
    tags=["Verification"],
)
async def list_verifications(
    limit: int = Query(default=25, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status_filter: Optional[VerificationStatus] = Query(default=None, alias="status"),
    db: AsyncSession = Depends(get_db),
) -> list[VerificationHistoryItem]:
    """Return recent verification logs ordered newest-first for dashboard history."""
    stmt = (
        select(VerificationLog)
        .options(selectinload(VerificationLog.company))
        .order_by(desc(VerificationLog.created_at))
        .offset(offset)
        .limit(limit)
    )
    if status_filter is not None:
        stmt = stmt.where(VerificationLog.status == status_filter)

    rows = (await db.execute(stmt)).scalars().all()
    items: list[VerificationHistoryItem] = []
    for log in rows:
        company = log.company
        items.append(
            VerificationHistoryItem(
                verification_id=log.id,
                company_name=company.legal_name if company else None,
                country_code=company.country_code if company else None,
                reg_id=company.reg_id if company else None,
                status=log.status,
                trust_score=log.trust_score,
                confidence_score=log.confidence_score,
                kill_switches_triggered=list(log.kill_switches or []),
                red_flags_count=len(log.red_flags or []),
                summary=log.summary or "",
                created_at=log.created_at,
            )
        )
    return items


@app.get(
    "/api/v1/verify/{verification_id}",
    response_model=VerificationResponse,
    summary="Fetch a stored verification report by ID",
    tags=["Verification"],
)
async def get_verification_report(
    verification_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> VerificationResponse:
    """Retrieve a previously computed verification report from the database."""
    stmt = select(VerificationLog).where(VerificationLog.id == verification_id)
    result = await db.execute(stmt)
    log = result.scalar_one_or_none()
    if log is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Verification report '{verification_id}' not found",
        )
    return log_to_response(log, cached=True)


@app.get(
    "/api/v1/blacklist",
    response_model=list[BlacklistResponse],
    summary="List all blacklisted entities",
    tags=["Blacklist"],
)
async def list_blacklist(
    db: AsyncSession = Depends(get_db),
) -> list[BlacklistResponse]:
    """Return all entries currently in the anti-fraud blacklist."""
    stmt = select(Blacklist).order_by(desc(Blacklist.created_at))
    rows = (await db.execute(stmt)).scalars().all()
    return [BlacklistResponse.model_validate(row) for row in rows]


@app.post(
    "/api/v1/blacklist",
    response_model=BlacklistResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Add an entity to the anti-fraud blacklist",
    tags=["Blacklist"],
)
async def add_to_blacklist(
    payload: BlacklistCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> BlacklistResponse:
    """Normalize and insert a reg_id, domain, phone, or email into the blacklist table."""
    entry = Blacklist(
        entity_type=payload.entity_type,
        value=payload.value,
        reason=payload.reason,
    )
    db.add(entry)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Entity '{payload.value}' is already blacklisted",
        )
    await db.refresh(entry)
    logger.warning(
        "Blacklisted entity type=%s value=%s reason=%s",
        entry.entity_type.value,
        entry.value,
        entry.reason,
    )
    return BlacklistResponse.model_validate(entry)


@app.delete(
    "/api/v1/blacklist/{entry_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove an entity from the blacklist",
    tags=["Blacklist"],
)
async def delete_from_blacklist(
    entry_id: UUID,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a blacklist entry by UUID."""
    stmt = select(Blacklist).where(Blacklist.id == entry_id)
    entry = (await db.execute(stmt)).scalar_one_or_none()
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Blacklist entry '{entry_id}' not found",
        )
    await db.delete(entry)
    await db.commit()
    logger.info("Removed blacklist entry id=%s value=%s", entry_id, entry.value)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health and database status",
    tags=["System"],
)
async def health_check(
    db: AsyncSession = Depends(get_db),
) -> HealthResponse | JSONResponse:
    """Check service liveness and database connectivity."""
    cfg = get_settings()
    is_healthy, db_status = await check_db_health(db)
    body = HealthResponse(
        status="ok" if is_healthy else "degraded",
        database=db_status,
        database_backend=cfg.database_backend,
        version=cfg.APP_VERSION,
        configured_integrations=cfg.configured_integrations,
    )
    if not is_healthy:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=body.model_dump(),
        )
    return body
