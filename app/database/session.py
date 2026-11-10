from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()

if settings.database_url.startswith("sqlite:///"):
    sqlite_path = settings.database_url.removeprefix("sqlite:///")
    Path(sqlite_path).parent.mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False}
    engine = create_engine(settings.database_url, future=True, connect_args=connect_args)
else:
    # PostgreSQL 만: DB 가 응답하지 않을 때 요청 스레드가 무한정 붙들리지 않게 한다(2차 점검 R-3).
    # statement_timeout 은 기동 때 스키마 보정이 길 수 있어 넣지 않는다.
    engine = create_engine(
        settings.database_url,
        future=True,
        pool_pre_ping=True,
        pool_timeout=settings.db_pool_timeout_sec,
        connect_args={"connect_timeout": settings.db_connect_timeout_sec},
    )
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=Session)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
