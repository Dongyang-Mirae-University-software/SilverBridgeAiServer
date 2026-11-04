from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.routers.companion_router import get_face_analyzer, get_text_classifier, get_transcriber
from app.services.companion.face_service import FaceResult, FaceUnavailable, summarize
from app.services.companion.stt_service import SttResult
from app.services.companion.text_emotion_service import TextEmotionResult
from scripts.companion.dev_app import app

KEY = "test-companion-key"
HDR = {"X-API-Key": KEY}


def test_summarize_no_faces_is_unusable():
    r = summarize([], total=5, min_rate=0.5)
    assert r == FaceResult(None, 0.0, 0, False, 5, 0)


def test_summarize_mode_and_confidence():
    det = [("sadness", 0.8), ("sadness", 0.6), ("happiness", 0.9)]
    r = summarize(det, total=4, min_rate=0.5)
    assert r.label == "sadness"
    assert r.frameCount == 2
    assert r.confidence == pytest.approx(0.7)
    assert r.facesFound == 3 and r.totalFrames == 4
    assert r.usable is True  # 3/4 = 0.75 >= 0.5


def test_summarize_below_min_rate_is_unusable_but_keeps_label():
    det = [("neutral", 0.7)]
    r = summarize(det, total=5, min_rate=0.5)  # 1/5 = 0.2 < 0.5
    assert r.usable is False
    assert r.label == "neutral"


def _jpeg_noise(seed: int = 0) -> bytes:
    import cv2

    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _multipart(audio: bytes, frames: list[bytes]) -> tuple[bytes, str]:
    boundary = "----faceboundary"
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="audio"; filename="a.wav"\r\n'
        f"Content-Type: audio/wav\r\n\r\n".encode() + audio + b"\r\n"
    ]
    for f in frames:
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="frame"; filename="f.jpg"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n".encode() + f + b"\r\n"
        )
    body = b"".join(parts) + f"--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def _wav(size: int = 64) -> bytes:
    return b"RIFF" + size.to_bytes(4, "little") + b"WAVE" + b"\x00" * size


@pytest.fixture
def client():
    settings = get_settings().model_copy(update={
        "companion_api_key": KEY,
        "companion_backend_base_url": "http://backend.test",
    })
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_transcriber] = lambda: (lambda a: SttResult("테스트 대답입니다", 0.9, 1))
    app.dependency_overrides[get_text_classifier] = lambda: (lambda t: TextEmotionResult("기쁨", 0.9, True, {}))
    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()


def _open_dialogue(c) -> str:
    r = c.post("/companion/dialogues", json={"wardId": "u1", "sessionId": "s1", "slot": "09:00", "question": "잘 주무셨어요?", "questionId": "q-test"}, headers=HDR)
    return r.json()["data"]["dialogueId"]


def test_no_frames_gives_face_unusable(client):
    did = _open_dialogue(client)
    body, ctype = _multipart(_wav(), [])
    r = client.post(f"/companion/dialogues/{did}/reply/audio", content=body, headers={**HDR, "content-type": ctype})
    assert r.status_code == 200
    face = r.json()["data"]["face"]
    assert face == {"label": None, "confidence": 0.0, "frameCount": 0, "usable": False, "totalFrames": 0, "facesFound": 0}


def test_frames_without_a_face_are_unusable(client):
    # 실제 모델을 쓴다: 노이즈 사진에는 얼굴이 없으므로 usable 이 False 여야 한다
    did = _open_dialogue(client)
    frames = [_jpeg_noise(i) for i in range(3)]
    body, ctype = _multipart(_wav(), frames)
    r = client.post(f"/companion/dialogues/{did}/reply/audio", content=body, headers={**HDR, "content-type": ctype})
    assert r.status_code == 200
    face = r.json()["data"]["face"]
    assert face["usable"] is False


def test_face_model_missing_does_not_stop_the_dialogue(client):
    def unavailable(_frames):
        raise FaceUnavailable("표정 모델 파일이 없습니다")

    app.dependency_overrides[get_face_analyzer] = lambda: unavailable
    did = _open_dialogue(client)
    body, ctype = _multipart(_wav(), [_jpeg_noise(1)])
    r = client.post(f"/companion/dialogues/{did}/reply/audio", content=body, headers={**HDR, "content-type": ctype})
    assert r.status_code == 200  # 대화는 계속
    assert r.json()["data"]["face"]["usable"] is False


def test_face_result_is_passed_through(client):
    seen = {}

    def fake(frames):
        seen["n"] = len(frames)
        return FaceResult("happiness", 0.8, 3, True, 3, 3)

    app.dependency_overrides[get_face_analyzer] = lambda: fake
    did = _open_dialogue(client)
    body, ctype = _multipart(_wav(), [_jpeg_noise(i) for i in range(3)])
    r = client.post(f"/companion/dialogues/{did}/reply/audio", content=body, headers={**HDR, "content-type": ctype})
    assert seen["n"] == 3
    assert r.json()["data"]["face"] == {"label": "happiness", "confidence": 0.8, "frameCount": 3, "usable": True, "totalFrames": 3, "facesFound": 3}


def test_more_than_five_frames_is_413(client):
    did = _open_dialogue(client)
    body, ctype = _multipart(_wav(), [_jpeg_noise(i) for i in range(6)])
    r = client.post(f"/companion/dialogues/{did}/reply/audio", content=body, headers={**HDR, "content-type": ctype})
    assert r.status_code == 413
    assert r.json()["code"] == "FRAMES_TOO_MANY"
