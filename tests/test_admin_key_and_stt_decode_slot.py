"""받아쓰기 디코딩 동시 상한(2026-11-10 전체 점검 M-2). 관리 키 테스트는 2026-11-11 스위치 제거로 삭제."""
from __future__ import annotations

import numpy as np
import pytest

from app.routers import companion_router as cr


def test_medgemma_admin_switch_removed() -> None:
    from fastapi.testclient import TestClient

    from app.main import app

    # 스위치 제거(2026-11-11): 어떤 키를 실어도 경로가 없다
    r = TestClient(app).get("/api/admin/medgemma/status", headers={"X-API-Key": "x"})
    assert r.status_code == 404


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
