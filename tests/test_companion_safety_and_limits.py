"""2026-11-09 BE 맞춤: 위험 턴 안내 응답(14번), 대답 500자 상한(13번), 프레임 1MB·음성 5MB(12번)."""

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.routers.companion_router import FRAME_MAX_BYTES, get_text_classifier, get_transcriber
from app.services.companion import pipeline
from app.services.companion.reply_service import SAFETY_REPLY
from app.services.companion.llm_client import LlmUnavailable
from app.services.companion.stt_service import SttResult
from app.services.companion.text_emotion_service import TextEmotionResult
from scripts.companion.dev_app import app

KEY = "test-companion-key"
HDR = {"X-API-Key": KEY}
GOOD = "그러셨군요. 요즘 입맛은 좀 어떠세요?"


class GoodLlm:
    """답은 주지만 신호(JSON)는 주지 않는 모델."""

    def generate(self, prompt, **_kwargs):
        return GOOD


class OffLlm:
    def generate(self, *_args, **_kwargs):
        raise LlmUnavailable("off")


def _run(llm, transcript):
    return pipeline.run_turn(transcript=transcript, face={},
                             classify_text=lambda _t: TextEmotionResult("우울", 0.9, True, {}),
                             llm=llm, question="오늘 하루는 어떠세요?", prev_signals=None,
                             is_last_turn=False, turn_no=1)


def test_risk_turn_gets_safety_reply_even_when_model_is_off():
    result = _run(OffLlm(), "숨이 안 쉬어져서 쓰러졌어요")
    assert result["riskFlag"] is True
    assert result["reply"]["text"] == SAFETY_REPLY
    assert result["replySource"] == "SAFETY"


def test_safety_reply_does_not_ask_about_meals():
    assert "식사" not in SAFETY_REPLY and "?" not in SAFETY_REPLY


def test_normal_turn_keeps_normal_reply():
    result = _run(GoodLlm(), "입맛이 별로 없어요")
    assert result["riskFlag"] is False
    assert result["reply"]["text"] != SAFETY_REPLY
    assert result["replySource"] != "SAFETY"


def test_size_limits_follow_be_standard():
    assert FRAME_MAX_BYTES == 1 * 1024 * 1024
    assert get_settings().companion_upload_max_mb == 5


def _client():
    settings = get_settings().model_copy(update={"companion_api_key": KEY, "companion_backend_base_url": ""})
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_transcriber] = lambda: (lambda audio: SttResult(text="", confidence=0.0, elapsed_ms=1))
    app.dependency_overrides[get_text_classifier] = lambda: (lambda t: TextEmotionResult("기쁨", 0.9, True, {}))
    return TestClient(app)


def _open_dialogue(client):
    body = {"wardId": "u-len", "sessionId": "s-len", "slot": "09:00", "question": "잘 주무셨어요?", "questionId": "q-len"}
    return client.post("/companion/dialogues", json=body, headers=HDR).json()["data"]["dialogueId"]


def test_answer_longer_than_500_chars_is_rejected():
    try:
        client = _client()
        dialogue_id = _open_dialogue(client)
        r = client.post(f"/companion/dialogues/{dialogue_id}/reply", json={"text": "가" * 501}, headers=HDR)
        assert r.status_code == 422
        assert r.json()["code"] == "ANSWER_TOO_LONG"
    finally:
        app.dependency_overrides.clear()


def test_answer_of_exactly_500_chars_is_accepted():
    try:
        client = _client()
        dialogue_id = _open_dialogue(client)
        r = client.post(f"/companion/dialogues/{dialogue_id}/reply", json={"text": "가" * 500}, headers=HDR)
        assert r.status_code == 200
    finally:
        app.dependency_overrides.clear()
