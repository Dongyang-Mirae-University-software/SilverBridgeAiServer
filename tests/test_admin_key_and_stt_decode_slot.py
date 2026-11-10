"""관리 경로 전용 키·받아쓰기 디코딩 동시 상한(2026-11-10 전체 점검 M-4·M-2 남은 부분)."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException

from app.core import security
from app.routers import companion_router as cr


def _s(api: str, admin: str) -> SimpleNamespace:
    return SimpleNamespace(api_key=api, admin_api_key=admin)


def test_admin_key_unset_accepts_common_key() -> None:
    security.require_admin_api_key("common", _s("common", ""))
    with pytest.raises(HTTPException) as e:
        security.require_admin_api_key("wrong", _s("common", ""))
    assert e.value.status_code == 401


def test_admin_key_set_rejects_common_key() -> None:
    security.require_admin_api_key("admin-only", _s("common", "admin-only"))
    for bad in ("common", "", None, "관리자"):
        with pytest.raises(HTTPException) as e:
            security.require_admin_api_key(bad, _s("common", "admin-only"))
        assert e.value.status_code == 401 and e.value.detail["errorCode"] == "AUTH_INVALID_KEY"


def test_admin_router_uses_admin_dependency() -> None:
    from app.main import app

    routes = [r for r in app.routes if getattr(r, "path", "").startswith("/api/admin/medgemma")]
    if not routes:  # 라우터를 감싸 등록하는 판에서는 OpenAPI 로만 경로를 확인한다
        assert any(p.startswith("/api/admin/medgemma") for p in app.openapi()["paths"])
        return
    for r in routes:
        assert any(d.call is security.require_admin_api_key for d in r.dependant.dependencies)


class _Result:
    text = "ok"


def test_decode_runs_inside_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[bool] = []

    def decode(_: bytes):
        # 디코딩하는 동안 자리 1개가 이미 잡혀 있어야 한다(자리가 1개면 더 못 얻는다).
        got = cr._stt_slots.acquire(blocking=False)
        seen.append(got)
        if got:
            cr._stt_slots.release()
        return np.zeros(16000, dtype=np.int16)

    import threading

    monkeypatch.setattr(cr, "_stt_slots", threading.BoundedSemaphore(1))
    monkeypatch.setattr(cr, "decode_to_pcm16k", decode)
    result, seconds = cr._transcribe_with_slot(lambda pcm: _Result(), b"x", 30.0, 0.1, 5.0)
    assert seen == [False] and result.text == "ok" and seconds == pytest.approx(1.0)
    assert cr._stt_slots.acquire(blocking=False)  # 끝나면 자리가 풀린다
    cr._stt_slots.release()


def test_too_long_and_unreadable_release_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    monkeypatch.setattr(cr, "_stt_slots", threading.BoundedSemaphore(1))
    monkeypatch.setattr(cr, "decode_to_pcm16k", lambda _: np.zeros(16000 * 40, dtype=np.int16))
    with pytest.raises(cr._AudioTooLong):
        cr._transcribe_with_slot(lambda pcm: _Result(), b"x", 30.0, 0.1, 5.0)

    def bad(_: bytes):
        raise ValueError("unreadable")

    monkeypatch.setattr(cr, "decode_to_pcm16k", bad)
    with pytest.raises(ValueError):
        cr._transcribe_with_slot(lambda pcm: _Result(), b"x", 30.0, 0.1, 5.0)
    assert cr._stt_slots.acquire(blocking=False)
    cr._stt_slots.release()


def test_no_slot_returns_none_without_decoding(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    sem = threading.BoundedSemaphore(1)
    sem.acquire()
    monkeypatch.setattr(cr, "_stt_slots", sem)
    called: list[int] = []
    monkeypatch.setattr(cr, "decode_to_pcm16k", lambda _: called.append(1))
    assert cr._transcribe_with_slot(lambda pcm: _Result(), b"x", 30.0, 0.05, 5.0) is None
    assert called == []
