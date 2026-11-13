"""말벗 대답 경로 기능 점검(2026-11-14, docs/audits/2026-11-14-companion-reply-path-audit.md) 반영.

M-1 이어 가기용 대화 상태(마지막 말벗 문장) · M-2 화제 판정 오탐 · M-3 위험·되묻기 턴과 짧은 대답 횟수 ·
L-1 표정까지 합친 부정 판정 · L-2 무응답 턴은 횟수 유지 · L-4 다시 생성 한 번 · L-5 글 대답 화제 전환·전환 다음 턴 질문.
LLM·위험·신호·요약은 스텁이다 (API 키를 쓰지 않는다).
"""
from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.routers import companion_router
from app.services.companion import pipeline
from app.services.companion.dialogue_store import store
from app.services.companion.reply_service import (
    SAFETY_REPLY,
    TOPIC_SWITCH,
    generate_reply,
    topic_switch_reply,
)
from app.services.companion.text_emotion_service import TextEmotionResult

OPENING = "좋은 아침이에요. 아침에 일어나셨을 때 몸은 좀 어떠셨어요?"
FOOD, SLEEP, BODY, FAMILY, NEIGHBOR, OUTSIDE = (s for _, s in TOPIC_SWITCH)


@pytest.fixture
def settings(monkeypatch):
    s = get_settings()

    def set_(**kw):
        for k, v in kw.items():
            monkeypatch.setattr(s, k, v)
    set_(companion_flat_topic_switch=2, companion_confirm_mode="logprob", companion_confirm_logprob=-0.3,
         companion_reply_similar_ratio=0.65, companion_reply_redo_temperature=0.7, companion_llm_retry=0,
         companion_risk_first=True, companion_reply_relax_calm=True, companion_avoid_reply=True,
         companion_reply_lora=False, companion_avoid_end_count=0)
    return set_


class Llm:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt, **kw):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else "오늘 오후에는 무엇을 하셨어요?"


@pytest.fixture
def stubs(monkeypatch):
    monkeypatch.setattr(pipeline, "extract_signals", lambda *a, **k: {"riskFlag": False})
    monkeypatch.setattr(pipeline, "summarize_answer", lambda *a, **k: "요약")
    monkeypatch.setattr(pipeline, "_risk_prob_capped", lambda *a, **k: 0.01)
    monkeypatch.setattr(pipeline, "risk_probability", lambda *a, **k: 0.01)


def _calm(_t):
    return TextEmotionResult("CALM", 0.9, True, {})


NO_FACE = {"label": None, "usable": False}
_OTHERS = iter([f"요즘 {w} 어떻게 지내세요?" for w in ("텔레비전은", "화초는", "아침마다", "날마다", "마음은", "저녁마다",
                                                      "주말에는", "오후에는", "요일마다", "장날에는")] * 10)


def _say(d, text, llm=None, logprob=-0.1, face=NO_FACE, kind="voice"):
    llm = llm or Llm([next(_OTHERS)])
    result = companion_router._run_turn(d, text, face, _calm, llm, stt_logprob=logprob, input_kind=kind)
    companion_router._turn_response(store.get(d.dialogueId), result, face=face)
    return result


def _dialogue(name, q=OPENING):
    return store.create("ward-" + name, "s-" + name, "09:00", q, 8, None)


# --- M-2 화제 판정 -------------------------------------------------------------------------------

@pytest.mark.parametrize("said,expected", [
    ([SAFETY_REPLY], FOOD),  # 안전 안내(물음 아님)의 "힘드셨"·"가족"은 화제가 아니다
    (["많이 힘드셨겠어요. 오늘은 어떤 일이 있으셨어요?"], FOOD),
    (["잠깐이라도 쉬셨군요. 오늘은 어떠셨어요?", "다행이에요. 식사는 하셨어요?"], SLEEP),
    (["잠시 쉬셨군요. 점심은 드셨어요?"], SLEEP),
    (["저녁노을이 예뻤겠네요. 누구와 보셨어요?"], FOOD),
    (["마음이 불편하셨군요. 무슨 일이 있으셨어요?", "그러셨군요. 저녁은 드셨어요?", "주무시긴 하셨어요?"], BODY),
])
def test_topic_detection_false_positives(said, expected):
    assert topic_switch_reply(said) == expected


@pytest.mark.parametrize("said,expected", [
    (["점심은 맛있게 드셨나요?"], SLEEP),
    (["어젯밤 잘 주무셨어요?", "식사는 하셨어요?"], BODY),
    ([OPENING, "밥은 드셨어요?", "잠은 잘 주무셨어요?"], FAMILY),  # 여는 질문의 "몸"은 물은 것
    (["손주분은 자주 오세요?", "점심은 드셨어요?", "푹 주무셨어요?", "아프신 데는 없으세요?"], NEIGHBOR),
    (["친구분들과 자주 보세요?", "점심은 드셨어요?", "푹 주무셨어요?", "아프신 데는 없으세요?", "가족분들은 잘 지내세요?"], OUTSIDE),
])
def test_topic_detection_true_positives(said, expected):
    assert topic_switch_reply(said) == expected


# --- M-3 · L-2 짧은 대답 횟수 ---------------------------------------------------------------------

def test_risk_turn_resets_streak_so_no_switch_right_after_safety(settings, stubs, monkeypatch):
    d = _dialogue("m3risk")
    _say(d, "몰라")
    monkeypatch.setattr(pipeline, "keyword_risk", lambda t: True)
    r = _say(d, "그냥 그래")
    assert r["replySource"] == "SAFETY" and store.get(d.dialogueId).flatStreak == 0
    monkeypatch.setattr(pipeline, "keyword_risk", lambda t: False)
    r = _say(d, "몰라")
    assert r["replySource"] != "TOPIC"  # 안전 안내 바로 다음에 가벼운 화제로 넘어가지 않는다


def test_confirm_turn_is_not_counted(settings, stubs):
    d = _dialogue("m3conf")
    _say(d, "몰라")
    _say(d, "뭐 그렇지", logprob=-0.9)  # 확신도 낮아 되묻기
    assert store.get(d.dialogueId).flatStreak == 1
    r = _say(d, "그냥 그래")
    assert r["replySource"] == "TOPIC"  # 되묻기 턴을 건너 2번째 짧은 대답


def test_no_response_turn_keeps_streak(settings, stubs):
    d = _dialogue("l2")
    _say(d, "몰라")
    _say(d, "")
    assert store.get(d.dialogueId).flatStreak == 1
    assert _say(d, "그냥 그래")["replySource"] == "TOPIC"


# --- L-1 표정까지 합친 부정 ------------------------------------------------------------------------

def test_sad_turn_result_keeps_model_reply(settings, stubs, monkeypatch):
    d = _dialogue("l1")
    _say(d, "몰라")
    real = pipeline.analyze_turn_result
    monkeypatch.setattr(pipeline, "analyze_turn_result",
                        lambda *a, **k: (real(*a, **k), {"label": "슬픔", "source": "FACE"})[1])
    r = _say(d, "그냥 그래")
    assert r["turnResult"]["label"] == "슬픔" and r["replySource"] != "TOPIC"


# --- L-5 글 대답 · 전환 다음 턴 질문 -----------------------------------------------------------------

def test_text_answer_also_switches_and_next_question_is_switch_sentence(settings, stubs):
    d = _dialogue("l5", q="오늘 오후는 어떻게 보내고 계세요?")
    _say(d, "그냥 그래", kind="text")
    r = _say(d, "몰라", kind="text")
    assert r["replySource"] == "TOPIC" and r["reply"]["text"] == FOOD
    llm = Llm(["김치찌개 드셨군요. 누구랑 같이 드셨어요?"])
    _say(d, "김치찌개 먹었어", llm=llm, kind="text")
    assert re.search(r"^질문: (.*)$", llm.prompts[0], re.M).group(1) == FOOD


# --- L-4 다시 생성 한 번 --------------------------------------------------------------------------

def test_redo_happens_once_even_with_retry(settings, stubs):
    settings(companion_llm_retry=2)
    past = ("아침에 몸 상태가 좋지 않으셨군요. 혹시 특별히 불편하신 곳이 있으셨나요?",)
    same = "아침에 몸 상태가 좀 불편하셨군요. 특별히 불편하신 곳은 없으셨나요?"
    llm = Llm([same] * 10)
    r = generate_reply(llm, "질문?", "계획 없어", None, False, 5, {"usable": True, "turnLabel": "평온"}, past_replies=past)
    assert r.source == "FALLBACK" and r.failures == ["반복"]
    assert len(llm.prompts) == 4  # 처음 1 + 다시 1 + 재시도 2 (다시 생성은 한 번뿐)


# --- M-1 이어 가기용 대화 상태 ----------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    from app.main import app
    monkeypatch.setattr(get_settings(), "companion_api_key", "k" * 32)
    return TestClient(app), {"X-API-Key": "k" * 32}


def test_dialogue_state_returns_last_question(settings, stubs, client):
    c, h = client
    d = _dialogue("m1")
    r0 = c.get(f"/companion/dialogues/{d.dialogueId}", headers=h).json()["data"]
    assert r0["lastQuestion"] == OPENING and r0["question"] == OPENING and r0["turnNo"] == 0
    reply = _say(d, "아침에 일어났을 때 너무 피곤했어")["reply"]["text"]
    r1 = c.get(f"/companion/dialogues/{d.dialogueId}", headers=h).json()["data"]
    assert r1["lastQuestion"] == reply and r1["turnNo"] == 1 and r1["ended"] is False
    assert set(r1) == {"dialogueId", "question", "lastQuestion", "turnNo", "ended"}  # 어르신 대답 원문 없음


def test_dialogue_state_404_and_key(client):
    c, h = client
    assert c.get("/companion/dialogues/nope", headers=h).status_code == 404
    assert c.get("/companion/dialogues/nope").status_code in (401, 403)
