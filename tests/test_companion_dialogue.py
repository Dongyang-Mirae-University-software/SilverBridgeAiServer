from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.routers.companion_router import get_text_classifier, get_transcriber
from app.services.companion.dialogue_store import store
from app.services.companion.review_store import review_store
from app.services.companion.stt_service import SttResult, SttUnavailable, confidence_from_logprobs
from app.services.companion.text_emotion_service import TextEmotionResult
from app.services.companion.timeouts import call_with_timeout
from app.services.companion.turn_record import send_dialogue_result
from scripts.companion.dev_app import app

KEY = "test-companion-key"
HDR = {"X-API-Key": KEY}


def _wav_bytes(size: int = 64) -> bytes:
    return b"RIFF" + (size).to_bytes(4, "little") + b"WAVE" + b"\x00" * size


def _multipart(name: str, filename: str, content_type: str, data: bytes) -> tuple[bytes, str]:
    boundary = "----testboundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def _fake_transcriber(audio: bytes) -> SttResult:
    # 시험용 가짜 STT: 소리가 있으면 고정 문장 (실제 모델은 test_stt_integration.py 에서 본다)
    return SttResult(text="테스트 대답입니다" if audio else "", confidence=0.9, elapsed_ms=1)


@pytest.fixture
def make_client():
    def _make(transcriber=_fake_transcriber):
        settings = get_settings().model_copy(update={
            "companion_api_key": KEY,
            "companion_backend_base_url": "",  # 턴 기록 전송은 끈다 (실제 전송 시험은 아래에서 따로 한다)
        })
        app.dependency_overrides[get_settings] = lambda: settings
        app.dependency_overrides[get_transcriber] = lambda: transcriber
        app.dependency_overrides[get_text_classifier] = lambda: (lambda t: TextEmotionResult("기쁨", 0.9, True, {}))
        return TestClient(app)

    yield _make
    app.dependency_overrides.clear()


def _open(client, slot="09:00"):
    return client.post("/companion/dialogues", json={"wardId": "u1", "sessionId": "s1", "slot": slot}, headers=HDR)


def test_open_requires_key(make_client):
    client = make_client()
    r = client.post("/companion/dialogues", json={"wardId": "u1", "sessionId": "s1", "slot": "09:00", "question": "잘 주무셨어요?", "questionId": "q-test"})
    assert r.status_code == 401
    assert r.json()["code"] == "AUTH_INVALID_KEY"


def test_open_rejects_unknown_slot(make_client):
    client = make_client()
    r = _open(client, slot="10:00")
    assert r.status_code == 422
    assert r.json()["errorCode"] == "SLOT_INVALID"


def test_open_uses_question_from_request(make_client):
    client = make_client()
    r = _open(client, slot="09:00")
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["question"] == "잘 주무셨어요?"
    assert data["maxTurns"] == 8


def test_dialogue_result_carries_question_id_and_screen_mood():
    # 노션 요구: 질문 ID를 되돌려 주기, 기분 값은 화면 4개(기쁨·평온·걱정·우울)로 묶기
    from app.services.companion.dialogue_store import Dialogue, TurnRecord
    from app.services.companion.turn_record import build_dialogue_result

    d = Dialogue("dlg-test-1", "ward-x", "s1", "09:00", "질문", 8, questionId="q-0900")
    d.turns = [TurnRecord(1, "슬픔", "TEXT"), TurnRecord(2, "화남", "TEXT"), TurnRecord(3, "기쁨", "TEXT"),
               TurnRecord(4, None, "NO_RESPONSE")]
    result = build_dialogue_result(d)
    assert result["questionId"] == "q-0900"
    assert result["mood"] == "걱정"  # 슬픔·화남 두 번 → 걱정이 최빈
    assert result["noResponseCount"] == 1
    assert result["turnCount"] == 4


def test_undecided_only_dialogue_has_null_mood_and_screen_moods_for_others():
    from app.services.companion.dialogue_store import Dialogue, TurnRecord
    from app.services.companion.turn_record import build_dialogue_result

    only_undecided = Dialogue("dlg-test-2", "ward-x", "s1", "09:00", "질문", 8)
    only_undecided.turns = [TurnRecord(1, "판단 보류", "CONFLICT")]
    assert build_dialogue_result(only_undecided)["mood"] is None
    sad = Dialogue("dlg-test-3", "ward-x", "s1", "09:00", "질문", 8)
    sad.turns = [TurnRecord(1, "우울", "TEXT")]
    assert build_dialogue_result(sad)["mood"] == "우울"


def test_ward_data_deletion_removes_only_that_ward(make_client):
    # 탈퇴 시 말벗 데이터 삭제: 그 피보호자의 대화와 검수 항목만 지운다
    from app.services.companion.review_store import review_store

    client = make_client()
    keep = _dialogue_id(client)
    client.post(f"/companion/dialogues/{keep}/reply", json={"text": "오늘 아침에 공원을 한 바퀴 걷고 왔어요"}, headers=HDR)
    review_store.add("other-dialogue", 1, "다른 피보호자 응답", "reply-v1", ward_id="ward-other")
    r = client.delete("/companion/wards/ward-demo-test-delete/data", headers=HDR)
    assert r.status_code == 200
    assert r.json()["data"] == {"wardId": "ward-demo-test-delete", "dialogues": 0, "reviewItems": 0}
    r2 = client.delete("/companion/wards/ward-demo-test-delete/data", headers=HDR)
    assert r2.status_code == 200
    assert r2.json()["data"]["dialogues"] == 0
    assert any(i["dialogueId"] == "other-dialogue" for i in review_store.list_items())
