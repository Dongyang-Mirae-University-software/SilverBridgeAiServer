"""별 내용 없는 짧은 대답이 이어지면 아직 묻지 않은 일상 화제로 넘어간다 (2026-11-13).

11/13 실기기(BE sessionId=212)·DGX 재생: "그냥 그래"·"몰라"·"계획 없어"가 이어지면 꼬리질문 모델(cc7)이 "좋아하시는 활동"을
30턴 중 15번 물었다. 생성 뒤 거르기·재학습(cc8, 54% → 37%)으로는 기준에 못 미쳐, 대화 흐름에서 화제를 바꾼다.
11/11 결정(힘든 이야기에는 화제를 바꾸지 않는다)에 맞춰 글 기분이 부정인 턴은 바꾸지 않는다.
LLM·위험·신호·요약은 스텁이다 (API 키를 쓰지 않는다).
"""
from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.routers import companion_router
from app.services.companion import pipeline
from app.services.companion.dialogue_store import store
from app.services.companion.pipeline import LOW_CONFIDENCE_REPLY
from app.services.companion.reply_service import (
    FALLBACK_AVOID,
    SAFETY_REPLY,
    TOPIC_SWITCH,
    check_reply,
    is_flat_answer,
    topic_switch_reply,
)
from app.services.companion.text_emotion_service import TextEmotionResult

OPENING = "좋은 아침이에요. 아침에 일어나셨을 때 몸은 좀 어떠셨어요?"
ACTIVITY = "그렇군요. 혹시 특별히 좋아하시는 활동이 있으신가요?"


@pytest.fixture
def settings(monkeypatch):
    s = get_settings()

    def set_(**kw):
        for k, v in kw.items():
            monkeypatch.setattr(s, k, v)
    set_(companion_flat_topic_switch=2, companion_confirm_mode="logprob", companion_confirm_logprob=-0.3,
         companion_reply_similar_ratio=0.65, companion_llm_retry=0, companion_risk_first=True,
         companion_reply_relax_calm=True, companion_avoid_reply=True, companion_reply_lora=False,
         companion_avoid_end_count=0)
    return set_


class Llm:
    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    def generate(self, prompt, **kw):
        self.calls += 1
        return self.reply


@pytest.fixture
def stubs(monkeypatch):
    monkeypatch.setattr(pipeline, "extract_signals", lambda *a, **k: {"riskFlag": False})
    monkeypatch.setattr(pipeline, "summarize_answer", lambda *a, **k: "요약")
    monkeypatch.setattr(pipeline, "_risk_prob_capped", lambda *a, **k: 0.01)
    monkeypatch.setattr(pipeline, "risk_probability", lambda *a, **k: 0.01)


def _mood(label):
    return lambda _t: TextEmotionResult(label, 0.9, True, {})


NO_FACE = {"label": None, "usable": False}


# 스텁 대답 - 서로 다른 문장이어야 #279 글자 되풀이 검사에 걸리지 않는다
_OTHERS = iter(["오늘 오후에는 무엇을 하며 지내셨어요?", "요즘 즐겨 보시는 텔레비전 방송이 있으세요?",
                "아침에 일어나시면 제일 먼저 무엇을 하세요?", "요즘 날마다 가시는 곳이 있으세요?",
                "요즘 마음 쓰이는 일은 없으세요?", "오늘 저녁에는 무엇을 하실 생각이세요?"] * 20)


def _say(d, text, reply=None, mood="CALM", logprob=-0.1):
    reply = reply or (ACTIVITY if ACTIVITY not in store.get(d.dialogueId).replies else next(_OTHERS))
    result = companion_router._run_turn(d, text, NO_FACE, _mood(mood), Llm(reply), stt_logprob=logprob)
    companion_router._turn_response(store.get(d.dialogueId), result, face=NO_FACE)
    return result


def _dialogue(name="flat", q=OPENING, max_turns=8):
    return store.create("ward-" + name, "s-" + name, "09:00", q, max_turns, None)


# --- 판정 ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["그냥 그래", "몰라", "없어", "계획 없어", "할 거 없어", "글쎄", "별로", "아니 없어",
                                  "알 거 없잖아", "없다니까", "그냥 있지 뭐.", "별일 없어", "그저 그래"])
def test_flat_answers(text):
    assert is_flat_answer(text)


@pytest.mark.parametrize("text", ["", "김치찌개 먹었지", "그냥 집에 있어", "손주가 와서 좋았어", "아침에 일어났을 때 너무 피곤했어",
                                  "오늘은 하루 보낼 거 계획 없다니까", "몰라 몰라 몰라 몰라 몰라 몰라 몰라 몰라"])
def test_not_flat_answers(text):
    assert not is_flat_answer(text)


@pytest.mark.parametrize("pattern,sentence", TOPIC_SWITCH)
def test_switch_sentences_pass_reply_rules(pattern, sentence):
    assert check_reply(sentence, False) == []
    assert "활동" not in sentence and "취미" not in sentence


def test_switch_picks_topic_not_yet_asked():
    assert topic_switch_reply(["점심은 드셨어요?"]) == TOPIC_SWITCH[1][1]  # 식사는 이미 물었다 → 잠
    assert topic_switch_reply([OPENING]) == TOPIC_SWITCH[0][1]  # "몸"은 물었지만 식사는 아직
    assert topic_switch_reply([s for _, s in TOPIC_SWITCH]) is None


# --- 대화 흐름 ------------------------------------------------------------------------------------

def test_session_212_second_flat_answer_switches_topic(settings, stubs):
    d = _dialogue("212")
    _say(d, "아침에 일어났을 때 너무 피곤했어", "아침부터 피곤하시다니 안타깝네요. 혹시 특별히 힘든 일이 있으셨나요?")
    r2 = _say(d, "알 거 없잖아", "그런 날도 있으시죠. 오늘 하루는 좀 더 편안하게 보내셨나요?")
    assert r2["replySource"] == "LLM"  # 짧은 대답 1번째는 그대로
    r3 = _say(d, "아니 없어")
    assert r3["replySource"] == "TOPIC" and r3["reply"]["text"] == TOPIC_SWITCH[0][1]  # 식사 (몸은 여는 질문에서 물음)
    assert r3["reply"]["isFollowUp"] is True
    r4 = _say(d, "없다고")
    assert r4["replySource"] == "LLM"  # 화제를 바꾼 뒤에는 다시 2번 이어져야 바꾼다
    r5 = _say(d, "계획 없어")
    assert r5["reply"]["text"] == TOPIC_SWITCH[1][1]  # 잠
    texts = [t for t in store.get(d.dialogueId).replies]
    assert texts.count(ACTIVITY) == 1  # 활동 질문은 한 번만 나갔다


def test_content_answer_resets_streak(settings, stubs):
    d = _dialogue("reset")
    _say(d, "몰라")
    _say(d, "김치찌개 먹었지")
    r = _say(d, "그냥 그래")
    assert r["replySource"] == "LLM" and store.get(d.dialogueId).flatStreak == 1


def test_negative_mood_keeps_model_reply(settings, stubs):
    d = _dialogue("neg")
    _say(d, "몰라", mood="DEPRESSED")
    r = _say(d, "그냥 그래", mood="DEPRESSED")
    assert r["replySource"] != "TOPIC"  # 11/11: 힘든 이야기에는 화제를 바꾸지 않는다


def test_risk_turn_keeps_safety_reply(settings, stubs, monkeypatch):
    d = _dialogue("risk")
    _say(d, "몰라")
    monkeypatch.setattr(pipeline, "keyword_risk", lambda t: True)
    r = _say(d, "그냥 그래")
    assert r["replySource"] == "SAFETY" and r["reply"]["text"] == SAFETY_REPLY


def test_avoid_turn_keeps_avoid_reply(settings, stubs):
    d = _dialogue("avoid")
    _say(d, "몰라")
    r = _say(d, "할 말 없어")
    assert r["replySource"] == "AVOID" and r["reply"]["text"] in FALLBACK_AVOID


def test_low_confidence_turn_keeps_reasking(settings, stubs):
    d = _dialogue("conf")
    _say(d, "몰라")
    r = _say(d, "뭐 그렇지", logprob=-0.9)
    assert r["reply"]["text"] == LOW_CONFIDENCE_REPLY


def test_last_turn_is_not_switched(settings, stubs):
    d = _dialogue("last", max_turns=2)
    _say(d, "몰라")
    r = _say(d, "그냥 그래", reply="오늘 이야기 나눠 주셔서 고마워요. 편안한 하루 보내세요.")
    assert r["replySource"] != "TOPIC"


def test_all_topics_used_keeps_model_reply(settings, stubs):
    d = _dialogue("used")
    d.replies.extend(s for _, s in TOPIC_SWITCH)
    _say(d, "몰라")
    r = _say(d, "그냥 그래")
    assert r["replySource"] == "LLM"


def test_switch_off(settings, stubs):
    settings(companion_flat_topic_switch=0)
    d = _dialogue("off")
    _say(d, "몰라")
    r = _say(d, "그냥 그래")
    assert r["replySource"] == "LLM"


def test_review_item_uses_fallback_version(settings, stubs):
    from app.services.companion.reply_service import FALLBACK_VERSION
    d = _dialogue("ver")
    _say(d, "몰라")
    _say(d, "그냥 그래")
    assert FALLBACK_VERSION == "fallback-v4"
