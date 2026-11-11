"""턴 응답의 v9 위험 확률 `riskProbability` (백엔드 여러 날 위험 누적용, 2026-11-11).

모델 확률만 싣는다(낱말 규칙 kw4·signals 대체 판정은 섞지 않는다). 계산 안 했거나 시간 초과·v9 아님이면 null.
riskFlag 판정 규칙·기준값은 그대로이고, 저장(검수 목록·턴 기록)에는 남기지 않는다.
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.routers.companion_router import _risk_probability_out, get_llm, get_text_classifier, get_transcriber
from app.services.companion import pipeline
from app.services.companion.dialogue_store import store
from app.services.companion.llm_client import LlmUnavailable
from app.services.companion.review_store import review_store
from app.services.companion.text_emotion_service import TextEmotionResult
from scripts.companion.dev_app import app

KEY = "test-companion-key"
HDR = {"X-API-Key": KEY}
CLASSIFY = lambda _t: TextEmotionResult("평온", 0.9, True, {})  # noqa: E731


@pytest.fixture(autouse=True)
def _v9(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "companion_risk_prompt", "v9")
    monkeypatch.setattr(s, "companion_risk_threshold", None)


class _RiskOnly:
    """대답·신호는 만들지 않고(연결 실패) v9 위험 확률만 준다."""

    def __init__(self, p, delay: float = 0.0):
        self.p, self.delay, self.calls = p, delay, 0

    def generate(self, *a, **k):
        raise LlmUnavailable("off")

    def message_token_probs(self, *a, **k):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return {"있": self.p, "없": 1 - self.p}


class _Broken(_RiskOnly):
    def message_token_probs(self, *a, **k):
        raise RuntimeError("down")


def _analyzed(llm, text="오늘은 공원에 다녀와서 기분이 좋아요"):
    return pipeline.run_turn(text, {}, CLASSIFY, llm, "q", None, False, 1, stt_logprob=-0.1)


# --- 파이프라인 경로별 -----------------------------------------------------------------------

@pytest.mark.parametrize("risk_first", [True, False])
def test_analyzed_turn_carries_model_probability(monkeypatch, risk_first):
    monkeypatch.setattr(get_settings(), "companion_risk_first", risk_first)
    r = _analyzed(_RiskOnly(0.123456))
    assert r["riskProbability"] == 0.123456 and r["riskFlag"] is False


def test_analyzed_turn_probability_failure_is_none():
    r = _analyzed(_Broken(0.5))
    assert r["riskProbability"] is None


def test_analyzed_turn_timeout_is_none(monkeypatch):
    monkeypatch.setattr(get_settings(), "companion_turn_cap_sec", 0.2)
    r = _analyzed(_RiskOnly(0.9999, delay=1.0))
    assert r["riskProbability"] is None


def test_keyword_risk_does_not_change_probability():
    """낱말 규칙(kw4)으로 위험이어도 확률은 모델 값 그대로다 (규칙은 확률이 아니다)."""
    r = _analyzed(_RiskOnly(0.02), text="요즘은 그냥 죽고 싶어요")
    assert r["riskFlag"] is True and r["riskProbability"] == 0.02


def test_low_confidence_turn_carries_probability(monkeypatch):
    monkeypatch.setattr(get_settings(), "companion_confirm_mode", "logprob")
    r = pipeline.run_turn("음 그게 뭐더라", {}, CLASSIFY, _RiskOnly(0.3), "q", None, False, 1, stt_logprob=-9.0)
    assert r["replySource"] == "CONFIRM" and r["riskProbability"] == 0.3


CONFIRM_TEXT = "삼기탕 먹고 나니 다 내려놓고 이제 그만 가고 싶어"


def test_lexicon_confirm_turn_with_model_carries_probability():
    llm = _RiskOnly(0.9999)
    r = pipeline.run_turn(CONFIRM_TEXT, {}, CLASSIFY, llm, "q", None, False, 1, stt_logprob=None)
    assert llm.calls == 1 and r["riskFlag"] is True and r["riskProbability"] == 0.9999


def test_lexicon_confirm_turn_without_model_is_none(monkeypatch):
    monkeypatch.setattr(get_settings(), "companion_confirm_risk_model", False)
    llm = _RiskOnly(0.9999)
    r = pipeline.run_turn(CONFIRM_TEXT, {}, CLASSIFY, llm, "q", None, False, 1, stt_logprob=None)
    assert llm.calls == 0 and r["riskProbability"] is None


def test_no_response_turn_is_none():
    assert pipeline.run_no_response_turn(1, False)["riskProbability"] is None


# --- 응답 값 ---------------------------------------------------------------------------------

def test_out_rounds_to_four_digits():
    assert _risk_probability_out(0.987654321) == 0.9877
    assert _risk_probability_out(1.0) == 1.0
    assert _risk_probability_out(None) is None


def test_out_is_null_when_not_v9(monkeypatch):
    """백엔드 누적 기준(0.97)은 v9 척도라 v4 확률은 내보내지 않는다."""
    monkeypatch.setattr(get_settings(), "companion_risk_prompt", "v4")
    assert _risk_probability_out(0.99) is None


# --- API: 응답 최상위에 실리고 저장되지 않는다 ---------------------------------------------------

@pytest.fixture
def client():
    settings = get_settings().model_copy(update={"companion_api_key": KEY, "companion_backend_base_url": ""})
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_transcriber] = lambda: None
    app.dependency_overrides[get_text_classifier] = lambda: CLASSIFY
    app.dependency_overrides[get_llm] = lambda: _RiskOnly(0.98765)
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_text_reply_response_has_rounded_probability_and_nothing_stored(client):
    body = {"wardId": "u-risk-prob", "sessionId": "s1", "slot": "09:00", "question": "잘 주무셨어요?", "questionId": "q"}
    did = client.post("/companion/dialogues", json=body, headers=HDR).json()["data"]["dialogueId"]
    r = client.post(f"/companion/dialogues/{did}/reply", json={"text": "오늘 아침에 공원을 한 바퀴 걷고 왔어요"}, headers=HDR)
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["riskProbability"] == 0.9877
    assert data["riskFlag"] is False  # 기준 0.99 그대로
    # 확률은 응답으로만 나간다: 검수 목록·대화 저장소에 남지 않는다
    assert "0.98765" not in repr(review_store.list_items(None))
    assert "0.98765" not in repr(store.get(did)) and "0.9877" not in repr(store.get(did))
