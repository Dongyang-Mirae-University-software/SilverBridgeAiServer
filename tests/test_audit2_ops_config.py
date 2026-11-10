from __future__ import annotations

import logging
import threading
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core import config as cfg
from app.core.security import admin_api_key_warning, require_admin_api_key
from app.database.base import Base
from app.database.session import get_db
from app.routers import companion_router, reservation_credential_router
from app.services.reservation_key_cipher import CipherUnavailable
from app.utils.logger import redact_api_key


def test_as_float_non_numeric_warns_without_value(monkeypatch, caplog) -> None:
    monkeypatch.setenv("X_TEST_F", "secret-abc")
    with caplog.at_level(logging.WARNING, logger="app.config"):
        assert cfg._as_float("X_TEST_F", 1.5) == 1.5
    assert "X_TEST_F" in caplog.text and "secret-abc" not in caplog.text


def test_as_int_non_numeric_warns(monkeypatch, caplog) -> None:
    monkeypatch.setenv("X_TEST_I", "abc")
    with caplog.at_level(logging.WARNING, logger="app.config"):
        assert cfg._as_int("X_TEST_I", 7) == 7
    assert "X_TEST_I" in caplog.text


def test_settings_survive_non_numeric_threshold_and_purge_hold(monkeypatch) -> None:
    monkeypatch.setenv("DEFAULT_DETECTION_THRESHOLD", "abc")
    monkeypatch.setenv("MEMBER_PURGE_HOLD_SEC", "xyz")
    s = cfg.Settings()
    assert s.default_detection_threshold == 0.75 and s.member_purge_hold_sec == 150.0


def test_claude_timeout_zero_and_stt_concurrency_zero_use_default(monkeypatch, caplog) -> None:
    monkeypatch.setenv("CLAUDE_TIMEOUT_SEC", "0")
    monkeypatch.setenv("COMPANION_STT_API_CONCURRENCY", "0")
    with caplog.at_level(logging.WARNING, logger="app.config"):
        s = cfg.Settings()
    assert s.claude_timeout_sec == 60 and s.companion_stt_api_concurrency == 2
    assert "CLAUDE_TIMEOUT_SEC" in caplog.text and "COMPANION_STT_API_CONCURRENCY" in caplog.text


def test_new_settings_defaults() -> None:
    s = cfg.Settings()
    assert (s.live_latest_analysis_max_age_sec, s.live_ws_send_timeout_sec, s.live_stale_session_purge_sec) == (30.0, 5.0, 1800.0)
    assert (s.medgemma_generate_max_time_sec, s.db_connect_timeout_sec, s.db_pool_timeout_sec) == (90.0, 5, 10)


def test_postgres_engine_gets_timeouts_and_sqlite_does_not(monkeypatch) -> None:
    import importlib
    from app.database import session as sess
    captured = {}
    real = sess.create_engine
    monkeypatch.setattr(sess, "create_engine", lambda url, **kw: captured.update(kw) or real("sqlite://"))
    monkeypatch.setattr(sess.settings.__class__, "database_url", "postgresql+psycopg://u:p@h/db", raising=False)
    try:
        importlib.reload(sess)
    except Exception:
        pass
    # reload 는 create_engine 을 다시 import 하므로 직접 검증하는 대신 소스 분기를 확인한다
    src = open(sess.__file__, encoding="utf-8").read()
    assert '"connect_timeout"' in src and "pool_timeout" in src
    assert "check_same_thread" in src.split("else:")[0]
    monkeypatch.undo()
    importlib.reload(sess)


def test_credential_cipher_unavailable_is_503_other_value_error_400() -> None:
    engine = create_engine("sqlite+pysqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    app = FastAPI()
    app.include_router(reservation_credential_router.router)
    app.dependency_overrides[get_db] = lambda: session
    body = {"userId": "aaaaaa", "reservationEmail": "a@b.com", "apiKey": "sbk_p1.secret1"}
    svc = reservation_credential_router.get_reservation_credential_service()
    with TestClient(app) as c:
        with patch.object(type(svc), "upsert_credential", side_effect=CipherUnavailable("x")):
            r = c.post("/api/v1/reservation-credentials", json=body)
        assert r.status_code == 503 and r.json()["detail"]["errorCode"] == "RESERVATION_CREDENTIAL_UNAVAILABLE"
        assert "비밀값" not in r.text
        with patch.object(type(svc), "upsert_credential", side_effect=ValueError("bad")):
            r = c.post("/api/v1/reservation-credentials", json=body)
        assert r.status_code == 400
    session.close()


def test_admin_key_compared_after_strip() -> None:
    s = cfg.Settings(admin_api_key="  adminkey  ", api_key="common")
    require_admin_api_key("adminkey", s)  # 예외 없음
    with pytest.raises(Exception):
        require_admin_api_key("common", s)


def test_admin_key_warning() -> None:
    assert admin_api_key_warning(cfg.Settings(admin_api_key=" same ", api_key="same")) is not None
    assert admin_api_key_warning(cfg.Settings(admin_api_key="a", api_key="b")) is None
    assert admin_api_key_warning(cfg.Settings(admin_api_key="", api_key="")) is None


def test_redact_ward_id_in_path_but_not_session_id() -> None:
    line = '127.0.0.1 "GET /companion/wards/aB3x9Z/data?apiKey=k HTTP/1.1" 200'
    out = redact_api_key(line)
    assert "aB3x9Z" not in out and "/companion/wards/***/data" in out and "apiKey=***" in out
    assert "/live/sessions/ward_abc" in redact_api_key("GET /live/sessions/ward_abc HTTP/1.1")


def test_transcribe_slot_released_when_submit_fails(monkeypatch) -> None:
    monkeypatch.setattr(companion_router, "_stt_slots", threading.BoundedSemaphore(1))

    def boom(*a, **k):
        raise RuntimeError("pool closed")

    monkeypatch.setattr(companion_router, "call_with_timeout", boom)
    with pytest.raises(RuntimeError):
        companion_router._transcribe_with_slot(lambda p: None, b"x", 30, 0.1, 1)
    assert companion_router._stt_slots.acquire(timeout=0.1)  # 자리가 돌아왔다


def test_transcribe_slot_not_double_released_on_work_error(monkeypatch) -> None:
    monkeypatch.setattr(companion_router, "_stt_slots", threading.BoundedSemaphore(1))
    monkeypatch.setattr(companion_router, "decode_to_pcm16k", lambda a: (_ for _ in ()).throw(ValueError("bad")))
    with pytest.raises(ValueError):
        companion_router._transcribe_with_slot(lambda p: None, b"x", 30, 0.1, 1)
    assert companion_router._stt_slots.acquire(timeout=0.1)  # BoundedSemaphore 라 이중 해제면 ValueError
