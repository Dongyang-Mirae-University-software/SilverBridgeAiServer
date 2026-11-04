from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings


class CompanionBase(DeclarativeBase):
    """말벗 AI 개선 DB 전용 Base (D11). main의 Base와 섞지 않는다."""


def get_companion_log_engine() -> Engine | None:
    """COMPANION_LOG_DATABASE_URL 이 있을 때만 엔진을 만든다. 없으면 None (적재 끔)."""
    url = get_settings().companion_log_database_url.strip()
    if not url:
        return None
    return create_engine(url, future=True, pool_pre_ping=True)
