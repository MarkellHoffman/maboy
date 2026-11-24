"""Application configuration loaded via Pydantic BaseSettings (Anthropic-only stack)."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SQLITE_URL = "sqlite+aiosqlite:///./dev.db"


class Settings(BaseSettings):
    """Typed environment settings with zero-config SQLite fallback."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_NAME: str = "Employer Anti-Fraud OSINT Verification Engine"
    APP_VERSION: str = "2.0.0"
    LOG_LEVEL: str = "INFO"

    # Database (PostgreSQL / Supabase via asyncpg, or automatic local SQLite fallback)
    DATABASE_URL: str = DEFAULT_SQLITE_URL

    # Anthropic API (Unified Web Search OSINT + Maps/Reviews + Vacancy Audit + Multimodal Intake)
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-haiku-4-5-20251001"
    CLAUDE_MAX_SEARCH_USES: int = 5

    # Free public EU government registries
    ARES_BASE_URL: str = "https://ares.gov.cz/ekonomicke-subjekty-v-be/rest/ekonomicke-subjekty"
    POLAND_WL_BASE_URL: str = "https://wl-api.mf.gov.pl/api/search/nip"

    # Pipeline execution controls
    MODULE_TIMEOUT_SECONDS: float = 10.0
    CLAUDE_TIMEOUT_SECONDS: float = 35.0
    INTAKE_MAX_FILE_BYTES: int = 10 * 1024 * 1024  # 10 MB per uploaded file
    CACHE_TTL_DAYS: int = 14
    HTTP_USER_AGENT: str = (
        "EmployerVerificationEngine/1.0 (+https://employer-verify.local; anti-fraud-osint)"
    )

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def _normalize_database_url(cls, value: Optional[str]) -> str:
        if not value or not str(value).strip():
            return DEFAULT_SQLITE_URL
        raw = str(value).strip()
        if raw.startswith("postgres://"):
            return "postgresql+asyncpg://" + raw[len("postgres://") :]
        if raw.startswith("postgresql://") and "+asyncpg" not in raw:
            return "postgresql+asyncpg://" + raw[len("postgresql://") :]
        return raw

    @field_validator("ANTHROPIC_API_KEY", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned if cleaned else None

    @property
    def is_sqlite(self) -> bool:
        return self.DATABASE_URL.startswith("sqlite")

    @property
    def database_backend(self) -> str:
        return "sqlite" if self.is_sqlite else "postgresql"

    @property
    def configured_integrations(self) -> dict[str, bool]:
        return {
            "ares_cz": True,
            "poland_mf_nip": True,
            "domain_crawler": True,
            "claude_osint": bool(self.ANTHROPIC_API_KEY),
            "multimodal_intake": bool(self.ANTHROPIC_API_KEY),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()


def configure_logging(level: Optional[str] = None) -> None:
    """Configure structured application logging."""
    log_level_name = (level or get_settings().LOG_LEVEL).upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        force=False,
    )
    logging.getLogger("app").setLevel(log_level)


settings = get_settings()
