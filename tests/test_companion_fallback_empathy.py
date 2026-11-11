"""말벗 대체 문장이 화제를 바꾸지 않는다 (2026-11-11).

실기기 시험에서 어르신이 힘든 이야기를 하는데, '대답과 무관'에 걸린 모델 대답이
"말씀해 주셔서 고맙습니다. 요즘 식사는 잘 챙겨 드시고 계세요?"(대체 문장)로 바뀌어 화제가 넘어갔다.
대체 문장은 어르신 말을 받아 주기만 하고, 그 턴의 글 기분이 부정이면 공감 문장을 쓴다.
"""
from __future__ import annotations

import pytest

from app.routers.companion_router import _avoid_repeat
from app.services.companion import pipeline, reply_service
from app.services.companion.dialogue_store import DialogueStore
from app.services.companion.reply_service import (
    FALLBACK_EMPATHY,
    FALLBACK_FOLLOWUP,
    FALLBACK_LAST,
    SAFETY_REPLY,
    ReplyResult,
    check_reply,
)
from app.services.companion.text_emotion_service import TextEmotionResult

# 화제를 바꾸는 말 (식사·하루 일과·시기 질문). 대체 문장에는 하나도 없어야 한다
TOPIC_CHANGE_WORDS = ("식사", "드시", "하루", "오늘", "요즘", "아침", "점심", "저녁", "주무")
TRANSCRIPT = "아들이 통 연락이 없어서 서운하고 속이 상해요"


def test_empathy_fallbacks_pass_their_own_rules():
    for s in FALLBACK_EMPATHY:
        assert check_reply(s, is_last_turn=False) == [], s


@pytest.mark.parametrize("sentence", FALLBACK_FOLLOWUP + FALLBACK_EMPATHY)
def test_fallbacks_do_not_change_the_topic(sentence):
    assert not any(w in sentence for w in TOPIC_CHANGE_WORDS), sentence


def test_fallback_lists_do_not_overlap():
    assert not set(FALLBACK_FOLLOWUP) & set(FALLBACK_EMPATHY)


@pytest.mark.parametrize("label", ["슬픔", "화남", "우울"])
def test_negative_usable_mood_is_negative(label):
    assert reply_service.is_negative_mood({"turnLabel": label, "usable": True})


@pytest.mark.parametrize("mood", [
    None,
    {},
    {"turnLabel": "평온", "usable": True},
    {"turnLabel": "기쁨", "usable": True},
    {"turnLabel": "우울", "usable": False},  # 쓸 수 없는 결과(짧은 말·낮은 확신)는 기분으로 보지 않는다
])
def test_other_moods_are_not_negative(mood):
    assert not reply_service.is_negative_mood(mood)


def test_last_turn_keeps_closing_sentence_even_when_negative():
    assert reply_service.fallback_options(is_last_turn=True, negative=True) == FALLBACK_LAST


def _stub_turn(monkeypatch, reply: ReplyResult, risk: bool = False):
    monkeypatch.setattr(pipeline, "generate_reply", lambda *a, **k: reply)
    monkeypatch.setattr(pipeline, "extract_signals", lambda *a, **k: {"riskFlag": False})
    monkeypatch.setattr(pipeline, "summarize_answer", lambda *a, **k: "요약")
    monkeypatch.setattr(pipeline, "_risk_prob_capped", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "keyword_risk", lambda t: risk)


def _run(label: str, turn_no: int = 2, is_last_turn: bool = False):
    return pipeline._analyzed_turn(
        TRANSCRIPT, {"label": None, "usable": False},
        lambda t: TextEmotionResult(label, 0.95, True, {}),
        llm=None, question="요즘 어떻게 지내세요?", prev_signals=None,
        is_last_turn=is_last_turn, turn_no=turn_no,
    )


def _fallback_reply(turn_no: int = 2) -> ReplyResult:
    return ReplyResult(text=reply_service.fallback_text(turn_no, False), isFollowUp=True,
                       source="FALLBACK", failures=["대답과 무관"])


@pytest.mark.parametrize("risk_first", [True, False])
@pytest.mark.parametrize("label", ["WORRY", "DEPRESSED", "슬픔", "불안"])
def test_negative_mood_fallback_turn_gets_empathy_sentence(monkeypatch, label, risk_first):
    settings = pipeline.get_settings().model_copy(update={"companion_risk_first": risk_first})
    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)
    _stub_turn(monkeypatch, _fallback_reply())
    out = _run(label)
    assert out["replySource"] == "FALLBACK"
    assert out["reply"]["text"] in FALLBACK_EMPATHY
    assert out["reply"]["isFollowUp"] is True


@pytest.mark.parametrize("label", ["CALM", "JOY"])
def test_calm_mood_fallback_turn_keeps_plain_sentence(monkeypatch, label):
    _stub_turn(monkeypatch, _fallback_reply())
    out = _run(label)
    assert out["reply"]["text"] in FALLBACK_FOLLOWUP


def test_model_reply_is_not_replaced_on_negative_turn(monkeypatch):
    good = ReplyResult(text="서운하셨겠어요. 아드님께 먼저 연락해 보시는 건 어떠세요?", isFollowUp=True,
                       source="LLM", failures=[])
    _stub_turn(monkeypatch, good)
    assert _run("WORRY")["reply"]["text"] == good.text


def test_risk_turn_still_gets_safety_reply(monkeypatch):
    _stub_turn(monkeypatch, _fallback_reply(), risk=True)
    out = _run("DEPRESSED")
    assert out["reply"]["text"] == SAFETY_REPLY
    assert out["replySource"] == "SAFETY"


def test_avoid_repeat_on_negative_turn_picks_an_unused_empathy_sentence():
    d = DialogueStore().create("w", "s", "09:00", "요즘 어떻게 지내세요?", 8, "q-1")
    d.replies.append(FALLBACK_EMPATHY[0])
    result = {"reply": {"text": FALLBACK_EMPATHY[0], "isFollowUp": True}, "replySource": "FALLBACK",
              "textMood": {"turnLabel": "우울", "usable": True}}
    out = _avoid_repeat(d, result, is_last_turn=False)
    assert out["reply"]["text"] in FALLBACK_EMPATHY[1:]
