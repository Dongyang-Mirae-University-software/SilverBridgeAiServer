"""대답 제출 API(api_mock.py)에 얼굴 확인이 연결되었는지 테스트. 실제 텍스트 모델·AI 서버·데이터 폴더를 쓰지 않는다.

실행 (저장소 루트에서):
    venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

COMP = Path(__file__).resolve().parents[2] / "app" / "companion"
sys.path.insert(0, str(COMP))
import api_mock  # noqa: E402
import stream_client  # noqa: E402

KEY = "test-key"
HEADERS = {"X-API-Key": KEY}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("COMPANION_API_KEY", KEY)
    monkeypatch.setattr(api_mock, "ANSWERS_FILE", tmp_path / "answers.json")
    monkeypatch.setattr(api_mock, "QUESTIONS_FILE", tmp_path / "guardian_questions.json")
    monkeypatch.setattr(api_mock, "text_label", lambda text: {
        "textEmotion": "기쁨", "textScore": 0.9, "label": "기쁨", "reason": "테스트"})
    monkeypatch.setenv("COMPANION_STREAM_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("COMPANION_STREAM_API_KEY", "stream-test-key")
    return TestClient(api_mock.app)


def blank_jpeg() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((240, 320, 3), 120, np.uint8))
    assert ok
    return buf.tobytes()


def post(client, **extra):
    body = {"wardId": "W1", "sessionId": "sess-1", "slot": "09:00", "text": "잘 잤어요, 기분이 좋아요", **extra}
    return client.post("/companion/answers", json=body, headers=HEADERS)


def test_answer_without_face_check_has_no_face_field(client):
    r = post(client)
    assert r.status_code == 200
    assert r.json()["data"]["face"] is None


def test_answer_with_face_check_records_unknown_when_no_face(client, monkeypatch):
    monkeypatch.setattr(stream_client, "fetch_latest_frame", lambda session_id: blank_jpeg())
    r = post(client, faceCheckSec=1)
    assert r.status_code == 200
    face = r.json()["data"]["face"]
    assert face["faceStatus"].startswith("판단 불가")
    assert face["frames"] >= 1


def test_face_result_is_stored_with_the_answer(client, monkeypatch, tmp_path):
    monkeypatch.setattr(stream_client, "fetch_latest_frame", lambda session_id: blank_jpeg())
    post(client, faceCheckSec=1)
    rows = json.loads((tmp_path / "answers.json").read_text(encoding="utf-8"))
    assert rows[-1]["face"]["faceStatus"].startswith("판단 불가")


def test_face_check_unreachable_does_not_fail_the_answer(client, monkeypatch):
    def down(session_id):
        raise stream_client.StreamError("UNREACHABLE", "연결 실패")

    monkeypatch.setattr(stream_client, "fetch_latest_frame", down)
    r = post(client, faceCheckSec=1)
    assert r.status_code == 200
    assert r.json()["data"]["face"]["faceStatus"].startswith("얼굴 확인 불가")
    assert r.json()["data"]["label"] == "기쁨"


def test_face_check_seconds_out_of_range_is_rejected(client):
    assert post(client, faceCheckSec=11).status_code == 422
    assert post(client, faceCheckSec=-1).status_code == 422
