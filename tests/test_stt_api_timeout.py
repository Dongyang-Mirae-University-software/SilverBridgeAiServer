"""/companion/stt 받아쓰기 제한 시간 (백엔드 점검 L-3): 오래 걸리면 503 STT_UNAVAILABLE, 자리는 끝나야 풀린다."""

import io
import time
import wave

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.routers.companion_router import _stt_slots, get_pcm_transcriber
from app.services.companion.stt_service import SttResult
from scripts.companion.dev_app import app

KEY = "test-stt-timeout-key"
HDR = {"X-API-Key": KEY}


def _wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes(b"\x10\x00" * 16000)  # 1초짜리 작은 소리
    return buf.getvalue()


def _client(transcriber, timeout_sec: float):
    settings = get_settings().model_copy(update={"companion_api_key": KEY, "companion_stt_api_timeout_sec": timeout_sec})
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_pcm_transcriber] = lambda: transcriber
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_slow_transcription_returns_503_stt_unavailable():
    def slow(_pcm):
        time.sleep(1.0)
        return SttResult(text="늦은 글자", confidence=1.0, elapsed_ms=1000)

    r = _client(slow, timeout_sec=0.2).post("/companion/stt", files={"audio": ("a.wav", _wav(), "audio/wav")}, headers=HDR)
    assert r.status_code == 503
    assert r.json()["code"] == "STT_UNAVAILABLE"


def test_fast_transcription_still_returns_text():
    def fast(_pcm):
        return SttResult(text="잘 잤어요", confidence=1.0, elapsed_ms=1)

    r = _client(fast, timeout_sec=5).post("/companion/stt", files={"audio": ("a.wav", _wav(), "audio/wav")}, headers=HDR)
    assert r.status_code == 200
    assert r.json()["data"]["text"] == "잘 잤어요"


def test_slot_is_released_after_slow_work_finishes():
    def slow(_pcm):
        time.sleep(0.6)
        return SttResult(text="", confidence=0.0, elapsed_ms=600)

    _client(slow, timeout_sec=0.1).post("/companion/stt", files={"audio": ("a.wav", _wav(), "audio/wav")}, headers=HDR)
    time.sleep(1.0)  # 돌고 있던 작업이 끝날 때까지 기다린다
    assert _stt_slots.acquire(timeout=0.5)  # 자리가 돌아와 있어야 한다
    _stt_slots.release()
