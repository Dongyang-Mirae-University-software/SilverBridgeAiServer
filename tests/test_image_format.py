"""얼굴 사진 형식 검사 (JPEG·PNG만). 파일 머리(매직 바이트)로 판별한다."""

import io
import wave

from fastapi.testclient import TestClient

from app.core.config import get_settings
from scripts.companion.dev_app import app
from app.services.companion.image_format import detect_image_format

KEY = "test-key-image"
HDR = {"X-API-Key": KEY}

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def _wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


def test_detects_jpeg_and_png_by_header():
    assert detect_image_format(JPEG) == "jpeg"
    assert detect_image_format(PNG) == "png"


def test_rejects_other_files_even_with_image_name():
    assert detect_image_format(b"GIF89a" + b"\x00" * 20) is None
    assert detect_image_format(b"%PDF-1.4 hello") is None
    assert detect_image_format(b"not an image") is None
    assert detect_image_format(b"") is None


def test_text_file_as_frame_is_rejected_with_422():
    app.dependency_overrides[get_settings] = lambda: get_settings().model_copy(update={"companion_api_key": KEY})
    try:
        client = TestClient(app)
        opened = client.post("/companion/dialogues",
                             json={"wardId": "u1", "sessionId": "s1", "slot": "09:00", "question": "잘 주무셨어요?"},
                             headers=HDR)
        did = opened.json()["data"]["dialogueId"]
        r = client.post(
            f"/companion/dialogues/{did}/reply/audio",
            files={"audio": ("answer.wav", _wav(), "audio/wav"),
                   "frame": ("photo.jpg", b"this is not a picture", "image/jpeg")},
            headers=HDR,
        )
        assert r.status_code == 422
        assert r.json()["code"] == "FRAME_FORMAT_UNSUPPORTED"
    finally:
        app.dependency_overrides.clear()
