from __future__ import annotations

import io
import logging
import threading
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.routers import companion_router
from app.routers.companion_router import get_pcm_transcriber
from scripts.companion.dev_app import app

KEY = "test-companion-key"
HDR = {"X-API-Key": KEY}
SECRET_TEXT = "머리가 아파요"


def _wav(seconds: float) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.zeros(int(16000 * seconds), dtype=np.int16)).tobytes())
    return buf.getvalue()


def _multipart(data: bytes, name: str = "audio", filename: str = "a.wav", ctype: str = "audio/wav"):
    boundary = "----sttboundary"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


@pytest.fixture
def client():
    settings = get_settings().model_copy(update={"companion_api_key": KEY, "companion_stt_api_wait_sec": 0.05})
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_pcm_transcriber] = lambda: (lambda pcm: SimpleNamespace(text=SECRET_TEXT))
    yield TestClient(app)
    app.dependency_overrides.clear()


def _post(client, data: bytes, **kw):
    body, ctype = _multipart(data, **kw)
    return client.post("/companion/stt", content=body, headers={**HDR, "content-type": ctype})


def test_stt_returns_text_only(client):
    r = _post(client, _wav(2.0))
    assert r.status_code == 200
    assert r.json()["data"] == {"text": SECRET_TEXT}


def test_stt_needs_the_key(client):
    body, ctype = _multipart(_wav(1.0))
    r = client.post("/companion/stt", content=body, headers={"content-type": ctype})
    assert r.status_code == 401


def test_stt_missing_audio_is_422(client):
    r = _post(client, _wav(1.0), name="other")
    assert r.status_code == 422
    assert r.json()["code"] == "AUDIO_MISSING"


def test_stt_unsupported_format_is_422(client):
    r = _post(client, b"not audio at all", filename="a.txt", ctype="text/plain")
    assert r.status_code == 422
    assert r.json()["code"] == "AUDIO_FORMAT_UNSUPPORTED"


def test_stt_longer_than_30_seconds_is_413(client):
    r = _post(client, _wav(31.0))
    assert r.status_code == 413
    assert r.json()["code"] == "AUDIO_TOO_LONG"


def test_stt_busy_when_no_slot_is_503(client, monkeypatch):
    slots = threading.BoundedSemaphore(1)
    slots.acquire()  # 자리를 미리 차지해 둔다
    monkeypatch.setattr(companion_router, "_stt_slots", slots)
    r = _post(client, _wav(1.0))
    assert r.status_code == 503
    assert r.json()["code"] == "STT_BUSY"


def test_stt_log_has_no_text_and_no_audio(client, caplog):
    caplog.set_level(logging.INFO, logger="companion.stt_api")
    _post(client, _wav(1.0))
    messages = " ".join(rec.getMessage() for rec in caplog.records)
    assert "stt status=200" in messages
    assert SECRET_TEXT not in messages
