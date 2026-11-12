"""몸 응급 안내 문장·세 턴 경로·종료 이유 집합 (2026-11-12 영향 범위 점검 Q-1·L-3 반영).

위험 턴 안내(SAFETY_REPLY)는 "마음이 많이 무거우신 것 같아요"라 "혀가 굳어서 말을 못 하겠어" 같은 몸 응급에 맞지 않았다.
몸 응급 낱말만 걸린 턴은 BODY_SAFETY_REPLY, 자해·죽음 낱말이 같이 있거나 모델 확률로만 위험이면 기존 안내.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

import pytest

from app.core.config import get_settings
from app.routers.companion_router import _avoid_repeat
from app.services.companion import pipeline
from app.services.companion.dialogue_store import store
from app.services.companion.llm_client import LlmUnavailable
from app.services.companion.reply_service import BODY_SAFETY_REPLY, SAFETY_REPLY, ReplyResult, safety_reply
from app.services.companion.risk import body_emergency_only, keyword_risk
from app.services.companion.text_emotion_service import TextEmotionResult

BODY = ["혀가 굳어서 말을 못 하겠어", "입이 안 돌아가서 말을 못 하겠네", "숨이 안 쉬어져", "가슴이 너무 아파",
        "영감이 쓰러졌어", "몸 한쪽이 마비가 왔어"]
MIXED = ["숨이 안 쉬어져서 죽고 싶어", "혀가 굳었는데 그냥 죽고 싶다", "쓰러졌으면 좋겠어 사라지고 싶어"]
MIND = ["죽고 싶어", "자살하고 싶어", "다 끝내고 싶어"]


@pytest.fixture
def settings(monkeypatch):
    s = get_settings()

    def set_(**kw):
        for k, v in kw.items():
            monkeypatch.setattr(s, k, v)
    return set_


@pytest.fixture(autouse=True)
def _v9(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "companion_risk_prompt", "v9")
    monkeypatch.setattr(s, "companion_risk_threshold", None)


class _RiskOnly:
    """대답·신호는 만들지 않고(연결 실패) v9 위험 확률만 준다."""

    def __init__(self, p):
        self.p, self.calls = p, 0

    def generate(self, *a, **k):
        raise LlmUnavailable("off")

    def message_token_probs(self, *a, **k):
        self.calls += 1
        return {"있": self.p, "없": 1 - self.p}


# --- 판정 ----------------------------------------------------------------------------------

@pytest.mark.parametrize("text", BODY)
def test_body_words_only_get_body_reply(text):
    assert keyword_risk(text) and body_emergency_only(text)
    assert safety_reply(text) == BODY_SAFETY_REPLY


@pytest.mark.parametrize("text", MIXED + MIND)
def test_self_harm_words_keep_mind_reply(text):
    assert keyword_risk(text) and not body_emergency_only(text)
    assert safety_reply(text) == SAFETY_REPLY


def test_model_only_risk_keeps_mind_reply():
    # 낱말이 없고 v9 확률로만 위험이면 무슨 위험인지 모르므로 기존 안내
    assert safety_reply("요즘 다 내려놓고 싶어") == SAFETY_REPLY


def test_switch_off_keeps_mind_reply(settings):
    settings(companion_body_safety_reply=False)
    assert safety_reply("혀가 굳어서 말을 못 하겠어") == SAFETY_REPLY


def test_jaehee_rule_keeps_mind_reply(settings):
    settings(companion_risk_keyword_rule="jaehee")
    assert safety_reply("숨이 안 쉬어져") == SAFETY_REPLY


def test_body_reply_says_119_and_does_not_mention_mind():
    assert "119" in BODY_SAFETY_REPLY and "마음" not in BODY_SAFETY_REPLY


# --- 세 턴 경로 (L-3) --------------------------------------------------------------------------

def _stub_analyzed(monkeypatch):
    monkeypatch.setattr(pipeline, "generate_reply",
                        lambda *a, **k: ReplyResult(text="그러셨군요?", isFollowUp=True, source="LLM", failures=[]))
    monkeypatch.setattr(pipeline, "extract_signals", lambda *a, **k: {"riskFlag": False})
    monkeypatch.setattr(pipeline, "summarize_answer", lambda *a, **k: "요약")
    monkeypatch.setattr(pipeline, "_risk_prob_capped", lambda *a, **k: 0.75)


def test_analyzed_turn_stroke_gets_body_reply(monkeypatch):
    _stub_analyzed(monkeypatch)
    r = pipeline._analyzed_turn("혀가 굳어서 말을 못 하겠어", {"label": None, "usable": False},
                                lambda t: TextEmotionResult("우울", 0.9, True, {}),
                                llm=None, question="q", prev_signals=None, is_last_turn=False, turn_no=2)
    assert r["riskFlag"] is True and r["replySource"] == "SAFETY"
    assert r["reply"]["text"] == BODY_SAFETY_REPLY and r["avoided"] is False


def test_analyzed_turn_self_harm_keeps_mind_reply(monkeypatch):
    _stub_analyzed(monkeypatch)
    r = pipeline._analyzed_turn("죽고 싶어", {"label": None, "usable": False},
                                lambda t: TextEmotionResult("우울", 0.9, True, {}),
                                llm=None, question="q", prev_signals=None, is_last_turn=False, turn_no=2)
    assert r["reply"]["text"] == SAFETY_REPLY


def test_low_confidence_turn_stroke_gets_body_reply():
    # 확신도가 낮아 되묻는 턴이어도 낱말 위험이면 되묻지 않고 안내한다 (모델 확률은 위험 아래)
    r = pipeline.run_turn("혀가 굳어서 말을 못 하겠어", {}, lambda t: None, _RiskOnly(0.1), "q", None, False, 1,
                          stt_logprob=-9.0)
    assert r["riskFlag"] is True and r["replySource"] == "SAFETY"
    assert r["reply"]["text"] == BODY_SAFETY_REPLY
    assert r["lowConfidence"] is False  # 위험 턴은 연속 확인 종료 횟수에 세지 않는다


def test_lexicon_confirm_turn_stroke_gets_body_reply():
    # 실제 단어가 아닐 수 있는 말(삼기탕)이 있어 다시 확인하는 턴이어도 낱말 위험이면 안내한다
    r = pipeline.run_turn("삼기탕 먹다가 혀가 굳어서 말을 못 하겠어", {}, lambda t: None, _RiskOnly(0.1), "q", None,
                          False, 1, stt_logprob=None)
    assert r["needsCheck"] is True
    assert r["riskFlag"] is True and r["replySource"] == "SAFETY"
    assert r["reply"]["text"] == BODY_SAFETY_REPLY


def test_repeated_body_reply_is_kept():
    # 안전 안내는 반복돼도 다른 문장으로 바꾸지 않는다 (COMPANION_KEEP_FIXED_REPLIES)
    d = store.create("ward-body", "s1", "09:00", "잘 주무셨어요?", 8, "q1")
    d.replies.append(BODY_SAFETY_REPLY)
    result = {"reply": {"text": BODY_SAFETY_REPLY, "isFollowUp": True}, "replySource": "SAFETY",
              "turnResult": {"label": None, "source": "TEXT"}, "textMood": None}
    assert _avoid_repeat(d, result, False)["reply"]["text"] == BODY_SAFETY_REPLY


# --- 종료 이유 집합 (L-3: AI 가 내는 값 ⊆ 백엔드 CompanionEndReason) ------------------------------

BACKEND_END_REASONS = {"MAX_TURNS", "NO_RESPONSE", "USER_STOP", "AVOIDED"}  # BE V72·CompanionEndReason


def test_ai_end_reasons_are_known_to_backend():
    root = Path(__file__).resolve().parents[1] / "app"
    src = (root / "services/companion/dialogue_store.py").read_text() + (root / "routers/companion_router.py").read_text()
    emitted = set(re.findall(r'd\.endReason = True, "([A-Z_]+)"', src))
    emitted |= set(re.findall(r'\["endReason"\] = "([A-Z_]+)"', src))
    emitted |= set(re.findall(r'store\.finish\(dialogue_id, "([A-Z_]+)"\)', src))
    assert emitted == BACKEND_END_REASONS, emitted  # 새 값을 내려면 백엔드 enum·CHECK 를 먼저 바꾼다
