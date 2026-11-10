"""업로드 하드닝(2026-11-10 점검 H-1·M-1·M-2): 여는 단계 형식 제한, 프레임 상한, STT 사전 거절."""
from __future__ import annotations

import io
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from PIL import Image

import app.models.ai_model  # noqa: F401
import app.models.camera  # noqa: F401
from app.core.config import get_settings
from app.core.response import error_response
from app.core.security import require_api_key
from app.routers import companion_router
from app.routers import live_stream_router as live_router
from app.routers.companion_router import get_pcm_transcriber
from app.services import medication_ocr_image
from app.services.companion import face_service
from app.services.medication_ocr_image import InvalidImageError, normalize_image
from scripts.companion.dev_app import app as companion_app
from tests.clip_helpers import make_jpeg

API_KEY = "test-upload-key"
HEADERS = {"X-API-Key": API_KEY}


def _img_bytes(fmt: str, size=(40, 30)) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", size, (10, 120, 200)).save(out, format=fmt)
    return out.getvalue()


# ---- H-1: Image.open formats= -------------------------------------------------

@pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF"])
def test_ocr_rejects_other_formats(fmt):
    with pytest.raises(InvalidImageError):
        normalize_image(_img_bytes(fmt))


def test_ocr_open_stage_blocks_format_even_if_allowed_set_widens(monkeypatch):
    # 선행 검사(ALLOWED_FORMATS)가 넓어져도 디코더는 formats= 밖 형식을 열지 않는다.
    monkeypatch.setattr(medication_ocr_image, "ALLOWED_FORMATS", frozenset({"JPEG", "PNG", "WEBP", "GIF"}))
    with pytest.raises(InvalidImageError):
        normalize_image(_img_bytes("GIF"))


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_ocr_accepts_allowed_formats(fmt):
    assert normalize_image(_img_bytes(fmt)).jpeg[:3] == b"\xff\xd8\xff"


def test_face_head3_opens_only_jpeg_png(monkeypatch):
    seen = {}
    real_open = Image.open

    def spy(fp, mode="r", formats=None):
        seen["formats"] = formats
        return real_open(fp, mode, formats)

    monkeypatch.setattr(Image, "open", spy)
    svc = object.__new__(face_service.FaceService)
    svc._h3 = (SimpleNamespace(), SimpleNamespace(), None)
    assert svc._head3_frame(_img_bytes("GIF")) is None  # 허용 밖 형식 = 얼굴 없음
    assert seen["formats"] == ("JPEG", "PNG")


# ---- M-1: 프레임 업로드 -------------------------------------------------------

def _frame_app(**overrides) -> FastAPI:
    app = FastAPI()

    @app.exception_handler(HTTPException)
    async def _handler(_: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "success" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(status_code=exc.status_code, content=error_response(str(exc.detail), "HTTP_ERROR", None))

    app.include_router(live_router.router, dependencies=[Depends(require_api_key)])
    settings = get_settings().model_copy(update={"api_key": API_KEY, **overrides})
    app.dependency_overrides[get_settings] = lambda: settings
    return app


@pytest.fixture
def frame_client():
    from app.database.base import Base
    from app.database.session import engine

    Base.metadata.create_all(bind=engine)
    with TestClient(_frame_app(stream_frame_max_bytes=200_000, stream_frame_max_side=1000)) as c:
        r = c.post("/api/v1/stream-sessions", json={"sessionId": "uh_sess", "cameraIdentifier": "cam"}, headers=HEADERS)
        assert r.status_code == 200, r.text
        yield c


def _post_frame(c, data: bytes, mime="image/jpeg"):
    return c.post("/api/v1/stream-sessions/uh_sess/frame", files={"frame": ("f.jpg", data, mime)}, headers=HEADERS)


def test_frame_normal_ok(frame_client):
    r = _post_frame(frame_client, make_jpeg(640, 360))
    assert r.status_code == 200, r.text
    assert r.json()["data"]["sessionId"] == "uh_sess"


def test_frame_over_bytes_is_413(frame_client):
    r = _post_frame(frame_client, b"\xff\xd8\xff" + b"0" * 300_000)
    assert r.status_code == 413
    assert r.json()["errorCode"] == "STREAM_FRAME_TOO_LARGE"


def test_frame_far_over_limit_is_413(frame_client):
    r = _post_frame(frame_client, b"\xff\xd8\xff" + b"0" * 1_000_000)
    assert r.status_code == 413


def test_frame_over_resolution_is_413(frame_client):
    r = _post_frame(frame_client, make_jpeg(1200, 200))
    assert r.status_code == 413
    assert r.json()["errorCode"] == "STREAM_FRAME_TOO_LARGE"


def test_frame_png_bytes_with_jpeg_mime_is_400(frame_client):
    r = _post_frame(frame_client, _img_bytes("PNG"))
    assert r.status_code == 400
    assert r.json()["errorCode"] == "STREAM_FRAME_INVALID_TYPE"


def test_frame_wrong_content_type_is_400(frame_client):
    r = _post_frame(frame_client, make_jpeg(64, 48), mime="image/png")
    assert r.status_code == 400
    assert r.json()["errorCode"] == "STREAM_FRAME_INVALID_TYPE"


# ---- M-2: /companion/stt ------------------------------------------------------

def _wav(seconds: float) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.zeros(int(16000 * seconds), dtype=np.int16).tobytes())
    return buf.getvalue()


def _mp(data: bytes):
    b = "----uhb"
    body = (f'--{b}\r\nContent-Disposition: form-data; name="audio"; filename="a.wav"\r\n'
            f"Content-Type: audio/wav\r\n\r\n").encode() + data + f"\r\n--{b}--\r\n".encode()
    return body, f"multipart/form-data; boundary={b}"


@pytest.fixture
def stt_client():
    s = get_settings().model_copy(update={"companion_api_key": "k", "companion_stt_api_max_mb": 1})
    companion_app.dependency_overrides[get_settings] = lambda: s
    companion_app.dependency_overrides[get_pcm_transcriber] = lambda: (lambda pcm: SimpleNamespace(text="ok"))
    yield TestClient(companion_app)
    companion_app.dependency_overrides.clear()


def test_stt_normal_ok(stt_client):
    body, ct = _mp(_wav(1.0))
    r = stt_client.post("/companion/stt", content=body, headers={"X-API-Key": "k", "content-type": ct})
    assert r.status_code == 200 and r.json()["data"] == {"text": "ok"}


def test_stt_over_limit_is_413_without_full_read(stt_client):
    body, ct = _mp(b"0" * (1024 * 1024 + 10_000))
    r = stt_client.post("/companion/stt", content=body, headers={"X-API-Key": "k", "content-type": ct})
    assert r.status_code == 413 and r.json()["code"] == "AUDIO_TOO_LARGE"


def test_stt_content_length_precheck_does_not_stream():
    # Content-Length 가 상한을 넘으면 request.stream() 을 한 번도 읽지 않고 거절한다.
    class _Req:
        headers = {"content-length": "999999999"}

        async def stream(self):
            raise AssertionError("본문을 읽으면 안 된다")
            yield b""

    import asyncio
    from app.core.companion_errors import CompanionError

    err = CompanionError(413, "AUDIO_TOO_LARGE", "x")
    with pytest.raises(CompanionError) as ei:
        asyncio.run(companion_router._read_limited_body(_Req(), 1000, err))
    assert ei.value.code == "AUDIO_TOO_LARGE"
