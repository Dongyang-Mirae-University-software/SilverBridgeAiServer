"""챗 기록 보관 기간 정리 (2026-11-17 개인정보 점검 H-2, 90일)."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core import config
from app.database.base import Base
from app.models.chat_log import ChatLog
from app.services import chat_log_retention as r

NOW = datetime(2026, 11, 17, 8, 0, 0)  # UTC naive (created_at 과 같은 기준)
SECRET = "상담-본문-원문"


@pytest.fixture()
def factory():
    engine = create_engine("sqlite+pysqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def _add(db, no: str, created_at: datetime, user: str = "user01") -> None:
    db.add(ChatLog(chat_no=no, user_id=user, message=SECRET, reply=SECRET, context_json="{}", created_at=created_at))
    db.commit()


def _nos(db) -> set[str]:
    return {row.chat_no for row in db.query(ChatLog).all()}


def test_deletes_only_rows_older_than_retention(factory) -> None:
    db = factory()
    _add(db, "old", NOW - timedelta(days=91))
    _add(db, "edge-old", NOW - timedelta(days=90, seconds=1))
    _add(db, "edge-keep", NOW - timedelta(days=90) + timedelta(seconds=1))
    _add(db, "new", NOW - timedelta(days=1))
    assert r.purge_old_chat_logs(db, 90, now=NOW) == 2
    assert _nos(db) == {"edge-keep", "new"}


def test_zero_or_negative_days_is_off(factory) -> None:
    db = factory()
    _add(db, "old", NOW - timedelta(days=400))
    assert r.purge_old_chat_logs(db, 0, now=NOW) == 0
    assert r.purge_old_chat_logs(db, -5, now=NOW) == 0
    assert _nos(db) == {"old"}


def test_deletes_in_batches(factory, monkeypatch) -> None:
    monkeypatch.setattr(r, "_BATCH", 3)
    db = factory()
    for i in range(7):
        _add(db, f"old-{i}", NOW - timedelta(days=200))
    _add(db, "new", NOW)
    assert r.purge_old_chat_logs(db, 90, now=NOW) == 7
    assert _nos(db) == {"new"}


def test_run_once_logs_count_only(factory, caplog) -> None:
    db = factory()
    _add(db, "old", datetime.utcnow() - timedelta(days=120), user="secretUser")
    caplog.set_level(logging.INFO, logger=r.__name__)
    assert r.run_once(SimpleNamespace(chat_log_retention_days=90), session_factory=factory) == 1
    text = caplog.text
    assert "[CHAT-RETENTION] deleted=1 days=90" in text
    assert SECRET not in text and "secretUser" not in text


def test_run_once_swallows_errors(caplog) -> None:
    class _Broken:
        def query(self, *a, **k):
            raise RuntimeError(SECRET)

        def rollback(self):
            pass

        def close(self):
            pass

    caplog.set_level(logging.WARNING, logger=r.__name__)
    assert r.run_once(SimpleNamespace(chat_log_retention_days=90), session_factory=_Broken) == -1
    assert "type=RuntimeError" in caplog.text and SECRET not in caplog.text


def test_start_loop_disabled_when_off() -> None:
    assert r.start_loop(SimpleNamespace(chat_log_retention_days=0)) is False


def test_default_is_90_days(monkeypatch) -> None:
    monkeypatch.delenv("CHAT_LOG_RETENTION_DAYS", raising=False)
    assert config._as_int("CHAT_LOG_RETENTION_DAYS", 90) == 90
    monkeypatch.setenv("CHAT_LOG_RETENTION_DAYS", "abc")
    assert config._as_int("CHAT_LOG_RETENTION_DAYS", 90) == 90


def test_main_starts_retention_loop_and_default_is_90() -> None:
    # app.main 은 모델 패키지를 불러와 소스 글자로 본다(기동 연결·기본값을 지우면 잡힌다).
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert "chat_log_retention.start_loop()" in (root / "app/main.py").read_text(encoding="utf-8")
    assert '_as_int("CHAT_LOG_RETENTION_DAYS", 90)' in (root / "app/core/config.py").read_text(encoding="utf-8")
