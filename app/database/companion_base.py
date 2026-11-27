from __future__ import annotations

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings


class CompanionBase(DeclarativeBase):
    """말벗 AI 개선 DB 전용 Base (D11). main의 Base와 섞지 않는다."""


# 11/27 AI #364 점검 W-1: 검수 DB(sqlite)를 검수 목록·고쳐 쓰기 횟수 두 엔진이 턴마다 쓰면서 동시 12건 이상에서
#   'database is locked'로 검수 쓰기가 실패했다. sqlite 면 WAL(읽기·쓰기 동시)과 잠금 대기를 켠다. PostgreSQL 은 그대로
SQLITE_BUSY_TIMEOUT_MS = 5000


def create_review_engine(url: str) -> Engine:
    """검수 DB 엔진. sqlite 면 연결마다 WAL·busy_timeout 을 켠다."""
    engine = create_engine(url, future=True, pool_pre_ping=True)
    if engine.dialect.name == "sqlite":
        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _record):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
            # WAL 의 권장 값. 쓰기마다 디스크 동기화를 기다리지 않는다 - 전원이 끊기는 순간 마지막 몇 건(검수 항목·횟수)을 잃을 수 있지만 DB 는 깨지지 않는다
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()
    return engine


def get_companion_log_engine() -> Engine | None:
    """COMPANION_LOG_DATABASE_URL 이 있을 때만 엔진을 만든다. 없으면 None (적재 끔)."""
    url = get_settings().companion_log_database_url.strip()
    if not url:
        return None
    return create_engine(url, future=True, pool_pre_ping=True)
