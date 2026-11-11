"""챗봇 받아쓰기 API(/companion/stt)의 엔진 스위치 COMPANION_STT_API_ENGINE 을 HTTP 로 확인한다.

turbo: Silero 말소리 감지 → turbo → 반복 정리·자막 환각 제거 (무음에서 "감사합니다"를 지어내지 않는다).
sensevoice(되돌리기용, 2026-11-11까지 기본): 예전 SenseVoice 경로 그대로. 기본은 turbo.
모델은 가짜로 바꿔 끼운다 (실제 모델·API 키를 쓰지 않는다). 상한(30초·10MB·동시 2)과 응답 형식은 그대로다.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.services.companion import vad
from app.services.companion.sensevoice_service import sensevoice_service
from app.services.companion.stt_service import SttResult
from app.services.companion.turbo_service import turbo_service
from scripts.companion.dev_app import app

KEY = "test-companion-key"
HDR = {"X-API-Key": KEY}
SR = 16000


def _wav(seconds: float, amp: float = 0.0) -> bytes:
    pcm = np.random.default_rng(0).standard_normal(int(SR * seconds)) * amp
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


def _post(client, data: bytes):
    boundary = "----sttswitch"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"a.wav\"\r\n"
            "Content-Type: audio/wav\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return client.post("/companion/stt", content=body,
                       headers={**HDR, "content-type": f"multipart/form-data; boundary={boundary}"})


@pytest.fixture
def client(monkeypatch):
    s = get_settings()  # 엔진 선택은 캐시된 설정을 직접 읽으므로 그 객체를 바꾼다
    monkeypatch.setattr(s, "companion_api_key", KEY)
    monkeypatch.setattr(s, "companion_vad", "silero")
    monkeypatch.setattr(s, "companion_stt_collapse_repeats", True)
    yield TestClient(app), s
    app.dependency_overrides.clear()


@pytest.fixture
def fake_turbo(monkeypatch):
    """turbo 의 GPU 추론만 가짜로. 불린 횟수와 돌려줄 글을 바꿀 수 있다."""
    state = {"calls": 0, "text": "내일 병원 예약해 줘"}

    def gen(pcm):
        state["calls"] += 1
        return state["text"], [-0.1, -0.1, -0.2]

    monkeypatch.setattr(turbo_service, "_generate", gen)
    return state


def test_turbo_silence_returns_empty_without_calling_the_model(client, fake_turbo, monkeypatch):
    c, s = client
    monkeypatch.setattr(s, "companion_stt_api_engine", "turbo")
    monkeypatch.setattr(vad.silero_vad, "speech_seconds", lambda _p: 0.0)
    fake_turbo["text"] = "시청해 주셔서 감사합니다."  # 모델이 불리면 지어냈을 말
    for data in (_wav(2.0), _wav(2.0, amp=0.05)):  # 무음, 생활 소음
        r = _post(c, data)
        assert r.status_code == 200
        assert r.json()["data"] == {"text": ""}
    assert fake_turbo["calls"] == 0


def test_turbo_speech_is_transcribed_and_cleaned(client, fake_turbo, monkeypatch):
    c, s = client
    monkeypatch.setattr(s, "companion_stt_api_engine", "turbo")
    monkeypatch.setattr(vad.silero_vad, "speech_seconds", lambda _p: 1.5)
    r = _post(c, _wav(2.0, amp=0.05))
    assert r.status_code == 200 and r.json()["data"] == {"text": "내일 병원 예약해 줘"}
    fake_turbo["text"] = "병원 병원 병원 병원 예약해 줘 자막 제공 배달의민족"
    assert _post(c, _wav(2.0, amp=0.05)).json()["data"] == {"text": "병원 예약해 줘"}


def test_turbo_hallucination_only_is_empty(client, fake_turbo, monkeypatch):
    c, s = client
    monkeypatch.setattr(s, "companion_stt_api_engine", "turbo")
    monkeypatch.setattr(vad.silero_vad, "speech_seconds", lambda _p: 1.0)  # 말소리로 잘못 들어와도
    fake_turbo["text"] = "시청해 주셔서 감사합니다."
    assert _post(c, _wav(2.0, amp=0.05)).json()["data"] == {"text": ""}


def test_sensevoice_switch_keeps_the_current_path(client, fake_turbo, monkeypatch):
    """되돌리면(sensevoice) 지금처럼 SenseVoice transcribe_pcm 결과가 그대로 나가고 turbo·Silero 는 부르지 않는다."""
    c, s = client
    monkeypatch.setattr(s, "companion_stt_api_engine", "sensevoice")
    seen = []
    monkeypatch.setattr(sensevoice_service, "transcribe_pcm",
                        lambda pcm: seen.append(pcm.size) or SttResult("네 네 네 네 약 먹었어요", 1.0, 3))

    def no_silero(_p):
        raise AssertionError("SenseVoice 경로는 Silero 를 부르지 않는다")

    monkeypatch.setattr(vad.silero_vad, "speech_seconds", no_silero)
    r = _post(c, _wav(2.0, amp=0.05))
    assert r.status_code == 200
    assert r.json()["data"] == {"text": "네 네 네 네 약 먹었어요"}  # 반복 정리도 하지 않는다 (지금과 같음)
    assert seen == [2 * SR] and fake_turbo["calls"] == 0


def test_limits_and_format_are_unchanged_with_turbo(client, fake_turbo, monkeypatch):
    c, s = client
    monkeypatch.setattr(s, "companion_stt_api_engine", "turbo")
    assert (s.companion_stt_api_max_sec, s.companion_stt_api_max_mb, s.companion_stt_api_concurrency) == (30.0, 10, 2)
    r = _post(c, _wav(31.0))
    assert r.status_code == 413 and r.json()["code"] == "AUDIO_TOO_LONG"
    assert fake_turbo["calls"] == 0
