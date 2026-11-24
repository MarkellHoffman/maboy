"""Async SQLAlchemy 2.0 database engine, session factory, and health helpers."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import StaticPool

from app.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def create_async_db_engine(database_url: str) -> AsyncEngine:
    """Create an async SQLAlchemy engine configured for SQLite or PostgreSQL."""
    is_sqlite = database_url.startswith("sqlite")
    kwargs: dict[str, Any] = {
        "echo": False,
        "future": True,
    }
    if is_sqlite:
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in database_url:
            kwargs["poolclass"] = StaticPool
    else:
        kwargs["pool_pre_ping"] = True
        kwargs["pool_size"] = 10
        kwargs["max_overflow"] = 20

    db_engine = create_async_engine(database_url, **kwargs)

    if is_sqlite:
        @event.listens_for(db_engine.sync_engine, "connect")
        def _set_sqlite_pragmas(dbapi_connection: Any, _connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA foreign_keys=ON")
                if ":memory:" not in database_url:
                    cursor.execute("PRAGMA journal_mode=WAL")
            finally:
                cursor.close()

    return db_engine


engine: AsyncEngine = create_async_db_engine(settings.DATABASE_URL)
async_session_maker: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def init_db(target_engine: AsyncEngine | None = None) -> None:
    """Create database tables if they do not exist (zero-config local startup)."""
    # Ensure ORM models are imported and registered on Base.metadata
    import app.models  # noqa: F401

    active_engine = target_engine or engine
    async with active_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema initialized (backend=%s)", settings.database_backend)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency providing an AsyncSession per request."""
    async with async_session_maker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def check_db_health(session: AsyncSession) -> tuple[bool, str]:
    """Execute a lightweight connectivity probe against the active DB."""
    try:
        await session.execute(text("SELECT 1"))
        return True, "connected"
    except Exception as exc:  # pragma: no cover - defensive runtime check
        logger.exception("Database health check failed: %s", exc)
        return False, f"error: {type(exc).__name__}"
