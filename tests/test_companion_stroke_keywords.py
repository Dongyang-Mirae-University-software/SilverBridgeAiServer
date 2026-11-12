"""뇌졸중 징후 낱말 규칙 (2026-11-12).

DGX(서버 코드 사본·int8·v9, 기준 0.99): "혀가 굳어서 말을 못 하겠어" 0.75 · "입이 안 돌아가서 말을 못 하겠네" 0.91 -
v9도 낱말 규칙도 못 잡아 보호자 알림이 가지 않았다. 오탐 측정(낱말만): CareCall 어르신 대답 24,248 중 0 · 웰니스 32,570 중 3.
"""
from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.services.companion import pipeline
from app.services.companion.avoidance import is_avoidance
from app.services.companion.reply_service import SAFETY_REPLY, ReplyResult
from app.services.companion.risk import STROKE_KEYWORDS, keyword_risk
from app.services.companion.text_emotion_service import TextEmotionResult

STROKE = [
    "혀가 굳어서 말을 못 하겠어",
    "입이 안 돌아가서 말을 못 하겠네",
    "갑자기 말이 어눌해졌어",
    "발음이 꼬여서 말이 잘 안 돼",
    "한쪽 팔에 힘이 하나도 없어",
    "얼굴 한쪽이 처진 것 같아",
    "몸 한쪽이 마비가 왔어",
    "아까부터 한쪽 다리에 힘이 안 들어가",
    "입이 한쪽으로 돌아갔어",
    "혀가꼬여서말이안나와",  # 띄어쓰기 없음
]
# 넣지 않은 꼴 - 감정·지난 일·틀니·어깨 통증 (웰니스·직접 작성)
NOT_STROKE = [
    "감정이 마비됐어",
    "발음이 이상하지는 않을까",
    "틀니를 빼서 발음이 안 돼",
    "한쪽 팔이 좀 쑤셔",
    "얼굴이 좀 부었어",
    "기가 막혀서 말이 안 나오네",
    "잘 잤어요. 아침에 산책도 했어",
]


@pytest.mark.parametrize("text", STROKE)
def test_stroke_signs_are_keyword_risk(text):
    assert keyword_risk(text), text


@pytest.mark.parametrize("text", NOT_STROKE)
def test_similar_words_are_not_stroke_risk(text):
    assert not keyword_risk(text), text


def test_jaehee_rule_is_unchanged(monkeypatch):
    # 재희 규칙(COMPANION_RISK_KEYWORD_RULE=jaehee)은 목록을 바꾸지 않는다 (kw4 에만 더함)
    monkeypatch.setattr(get_settings(), "companion_risk_keyword_rule", "jaehee")
    assert not keyword_risk("혀가 굳어서 말을 못 하겠어")


def test_stroke_keywords_have_no_duplicates():
    assert len(STROKE_KEYWORDS) == len(set(STROKE_KEYWORDS))


def test_stroke_turn_gets_safety_reply_not_avoid_reply(monkeypatch):
    # 말 피함 감지보다 위험 판정이 우선한다 - 몸 증상 말은 말 피함으로도 보지 않는다 (#274)
    monkeypatch.setattr(pipeline, "generate_reply",
                        lambda *a, **k: ReplyResult(text="그러셨군요?", isFollowUp=True, source="LLM", failures=[]))
    monkeypatch.setattr(pipeline, "extract_signals", lambda *a, **k: {"riskFlag": False})
    monkeypatch.setattr(pipeline, "summarize_answer", lambda *a, **k: "요약")
    monkeypatch.setattr(pipeline, "_risk_prob_capped", lambda *a, **k: 0.75)  # v9 는 위험 아래
    text = "혀가 굳어서 말을 못 하겠어"
    assert not is_avoidance(text)
    r = pipeline._analyzed_turn(text, {"label": None, "usable": False},
                                lambda t: TextEmotionResult("우울", 0.9, True, {}),
                                llm=None, question="잘 주무셨어요?", prev_signals=None, is_last_turn=False, turn_no=2)
    assert r["riskFlag"] is True
    assert r["reply"]["text"] == SAFETY_REPLY
    assert r["avoided"] is False
