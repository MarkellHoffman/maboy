"""SQLAlchemy 2.0 ORM models for companies, verification_logs, and blacklist."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import settings
from app.database import Base
from app.schemas import BlacklistEntityType, VerificationStatus


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _default_expires_at() -> datetime:
    return _utcnow() + timedelta(days=settings.CACHE_TTL_DAYS)


# Portable JSON column: uses JSONB on PostgreSQL (Supabase) and JSON on SQLite
PortableJSON = JSON().with_variant(JSONB(), "postgresql")


class Company(Base):
    """Verified or observed corporate entity indexed by (country_code, reg_id)."""

    __tablename__ = "companies"
    __table_args__ = (
        UniqueConstraint("country_code", "reg_id", name="uq_companies_country_reg_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    country_code: Mapped[str] = mapped_column(String(2), index=True, nullable=False)
    reg_id: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    legal_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    website_domain: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ares_data: Mapped[Optional[dict[str, Any]]] = mapped_column(PortableJSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    verification_logs: Mapped[list["VerificationLog"]] = relationship(
        "VerificationLog", back_populates="company"
    )


class VerificationLog(Base):
    """Persisted verification audit report with 14-day cache TTL."""

    __tablename__ = "verification_logs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("companies.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    input_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    status: Mapped[VerificationStatus] = mapped_column(
        SAEnum(VerificationStatus, name="verification_status_enum", native_enum=False),
        nullable=False,
    )
    trust_score: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    kill_switches: Mapped[list[str]] = mapped_column(PortableJSON, default=list, nullable=False)
    red_flags: Mapped[list[str]] = mapped_column(PortableJSON, default=list, nullable=False)
    evidence_links: Mapped[list[str]] = mapped_column(PortableJSON, default=list, nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    module_results: Mapped[dict[str, Any]] = mapped_column(
        PortableJSON, default=dict, nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_default_expires_at, nullable=False, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )

    company: Mapped[Optional[Company]] = relationship(
        "Company", back_populates="verification_logs"
    )


class Blacklist(Base):
    """Known fraudulent entities triggering immediate pre-flight kill-switch."""

    __tablename__ = "blacklist"

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_type: Mapped[BlacklistEntityType] = mapped_column(
        SAEnum(BlacklistEntityType, name="blacklist_entity_type_enum", native_enum=False),
        nullable=False,
    )
    value: Mapped[str] = mapped_column(Text, unique=True, index=True, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, nullable=False
    )
