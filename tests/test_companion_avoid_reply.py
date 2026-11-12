"""말을 피하는 대답에는 캐묻지 않고 물러나는 문장으로 답한다 (2026-11-12).

11/11 실기기 시험: "말하지 않고 있어…", "너무 힘들어서 얘기를 못 하겠어"에 말벗이 질문 주제를 갖다 붙이거나 더 캐물었다.
DGX(11/12): 피하는 말 40개 중 30개가 '대답과 무관' → "조금 더 들려주실 수 있을까요?"(캐묻는 대체 문장).
"""
from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.routers import companion_router
from app.services.companion import pipeline
from app.services.companion.avoidance import is_avoidance
from app.services.companion.dialogue_store import store
from app.services.companion.reply_service import (
    FALLBACK_AVOID,
    FALLBACK_EMPATHY,
    FALLBACK_FOLLOWUP,
    FALLBACK_LAST,
    FALLBACK_NO_RESPONSE,
    SAFETY_REPLY,
    ReplyResult,
    check_reply,
    fallback_options,
)
from app.services.companion.text_emotion_service import TextEmotionResult

AVOIDING = [
    "말하지 않고 있어…",
    "너무 힘들어서 얘기를 못 하겠어",
    "할 말 없어",
    "말하기 싫어",
    "그 얘기는 하기 싫어요",
    "묻지 마.",
    "왜 자꾸 물어봐요?",
    "말해 봤자 뭐해",
    "나중에 얘기할게",
    "그냥 넘어가자",
    "냅둬",
    # 11/12 점검 L-1
    "그만 물어봐",
    "대답하기 싫어",
    "몰라 묻지 마요",
    "할 얘기 없어",
    # 11/11 실기기 원문 전체 (DGX 피하는 말 40에 있음)
    "말하지 않고 있어 너무 말하면 신경 쓸 일 것 같아서 그냥 가만히 있어",
]
NOT_AVOIDING = [
    "말도 마 손주가 와서 좋았어",
    "묻지 마 화투에서 이겼어",
    "별 얘기 안 했는데",
    "잘 잤어요. 아침에 산책도 했어",
    "아들이랑 얘기를 많이 했어",
    "소용없어 그 약은",
    # 11/12 점검 M-1: 남 이야기 (가족 주어·전해 들은 말투)
    "손주가 말하지 않고 있어서 걱정이야",
    "아들이 아무 말도 안 하고 가버렸어",
    "영감이 말하기 싫다고 방에 들어갔어",
    "딸이 그 얘기는 하기 싫대",
    "며느리가 묻지 마.",
    "몰라 내가 혼구녕을 내줄라다가 우리 딸이 참으라고 그래서 아무 말도 안 했어요.",  # CareCall 오탐
]
# 몸이 불편해 말을 못 하는 말 - 응급일 수 있어 피하는 말로 보지 않는다 (평소 경로·위험 판정)
BODY = [
    "숨이 차서 말을 못하겠네",
    "가슴이 답답해서 말을 못 하겠어",
    "머리가 아파서 말하기 싫어",
    # 11/12 점검 H-1: 뇌졸중 말하기 장애·쓰러짐·기력 저하 (낱말 위험 규칙에 안 걸림)
    "혀가 굳어서 말을 못 하겠어",
    "입이 안 돌아가서 말을 못 하겠네",
    "쓰러질 것 같아서 말을 못 하겠어",
    "기운이 없어서 말을 못 하겠어",
    "힘이 없어서 얘기 못 하겠어",
    "정신이 없어서 얘기 못 하겠어",
    "몸이 너무 안 좋아서 말하기 싫어",
    "손이 저려서 말하기도 힘들어",
]


def test_go_away_is_not_body_word():
    # '저리 가'는 몸 증상(저리다)이 아니다 - CareCall "난 너랑 얘기하기 싫으니까 저리 가!" (DGX 대조 11/12)
    assert is_avoidance("난 너랑 얘기하기 싫으니까 저리 가!")


@pytest.fixture
def settings(monkeypatch):
    s = get_settings()

    def set_(**kw):
        for k, v in kw.items():
            monkeypatch.setattr(s, k, v)
    return set_


@pytest.mark.parametrize("text", AVOIDING)
def test_avoiding_answers_are_detected(text):
    assert is_avoidance(text), text


@pytest.mark.parametrize("text", NOT_AVOIDING + BODY + ["", "   "])
def test_other_answers_are_not_detected(text):
    assert not is_avoidance(text), text


@pytest.mark.parametrize("sentence", FALLBACK_AVOID)
def test_avoid_sentences_pass_reply_rules(sentence):
    assert check_reply(sentence, is_last_turn=False) == [], sentence


@pytest.mark.parametrize("sentence", FALLBACK_AVOID)
def test_avoid_sentences_do_not_ask_for_more(sentence):
    # 더 말해 달라는 말이 없어야 한다 (받아 주는 대체 문장·cc7 대답이 캐묻던 꼴)
    assert not any(w in sentence for w in ("더 들려", "더 해 주", "더 이야기", "걱정되는 일", "무슨 일")), sentence


def test_avoid_list_does_not_overlap_other_lists():
    others = set(FALLBACK_FOLLOWUP) | set(FALLBACK_EMPATHY) | set(FALLBACK_NO_RESPONSE) | set(FALLBACK_LAST)
    assert not set(FALLBACK_AVOID) & others


def test_last_turn_avoid_uses_closing_sentence():
    assert fallback_options(is_last_turn=True, avoid=True) == FALLBACK_LAST


# --- pipeline -----------------------------------------------------------------------------------

LLM_REPLY = ReplyResult(text="잠이 오지 않으셨군요?", isFollowUp=True, source="LLM", failures=[])


def _stub(monkeypatch, risk: bool = False, risk_prob: float | None = None):
    called = {"reply": 0}

    def gen(*a, **k):
        called["reply"] += 1
        return LLM_REPLY
    monkeypatch.setattr(pipeline, "generate_reply", gen)
    monkeypatch.setattr(pipeline, "extract_signals", lambda *a, **k: {"riskFlag": False})
    monkeypatch.setattr(pipeline, "summarize_answer", lambda *a, **k: "요약")
    monkeypatch.setattr(pipeline, "_risk_prob_capped", lambda *a, **k: risk_prob)
    monkeypatch.setattr(pipeline, "risk_probability", lambda *a, **k: risk_prob)
    if risk:
        monkeypatch.setattr(pipeline, "keyword_risk", lambda t: True)
    return called


def _turn(text: str, turn_no: int = 2, is_last_turn: bool = False, label: str = "우울"):
    return pipeline._analyzed_turn(
        text, {"label": None, "usable": False},
        lambda t: TextEmotionResult(label, 0.95, True, {}),
        llm=None, question="잘 주무셨어요?", prev_signals=None,
        is_last_turn=is_last_turn, turn_no=turn_no,
    )


@pytest.mark.parametrize("turn_no", [1, 2])
def test_avoiding_turn_gets_avoid_sentence_without_llm_reply(monkeypatch, turn_no):
    called = _stub(monkeypatch)
    r = _turn("말하기 싫어", turn_no=turn_no)
    assert r["reply"]["text"] in FALLBACK_AVOID
    assert r["reply"]["isFollowUp"] is True
    assert r["replySource"] == "AVOID"
    assert r["avoided"] is True
    assert called["reply"] == 0  # 모델 대답을 만들지 않는다
    assert r["turnResult"]["label"]  # 기분 판정은 그대로 돈다
    assert r["summary"] == "요약"


def test_avoid_sentences_rotate_by_turn(monkeypatch):
    _stub(monkeypatch)
    assert _turn("말하기 싫어", turn_no=1)["reply"]["text"] != _turn("말하기 싫어", turn_no=2)["reply"]["text"]


def test_avoiding_last_turn_gets_closing_sentence(monkeypatch):
    _stub(monkeypatch)
    r = _turn("할 말 없어", turn_no=8, is_last_turn=True)
    assert r["reply"] == {"text": FALLBACK_LAST[0], "isFollowUp": False}


def test_risk_turn_gets_safety_reply_even_when_avoiding(monkeypatch):
    _stub(monkeypatch)  # 낱말 규칙("죽고 싶")으로 위험
    r = _turn("죽고 싶어서 말도 하기 싫어")
    assert r["riskFlag"] is True
    assert r["reply"]["text"] == SAFETY_REPLY
    assert r["replySource"] == "SAFETY"


def test_model_risk_turn_gets_safety_reply_even_when_avoiding(monkeypatch):
    _stub(monkeypatch, risk_prob=0.99)  # 낱말 규칙에는 안 걸리고 v9 확률로 위험
    r = _turn("말하기 싫어")
    assert r["riskFlag"] is True
    assert r["reply"]["text"] == SAFETY_REPLY


def test_avoid_turn_without_risk_first_still_gets_safety_reply(monkeypatch, settings):
    # 11/12 점검 L-3: 위험 확률을 생성과 함께 계산하는 설정에서도 위험 턴은 안전 안내
    settings(companion_risk_first=False)
    _stub(monkeypatch, risk_prob=0.99)
    r = _turn("말하기 싫어")
    assert r["avoided"] is True and r["riskFlag"] is True
    assert r["reply"]["text"] == SAFETY_REPLY


def test_avoid_turn_with_risk_timeout_uses_signal_risk_flag(monkeypatch):
    # 11/12 점검 L-3: 위험 확률을 못 구하면(None) 신호 추출의 riskFlag 로 판정한다 - 말 피함 턴에도 신호 추출은 돈다
    _stub(monkeypatch, risk_prob=None)
    monkeypatch.setattr(pipeline, "extract_signals", lambda *a, **k: {"riskFlag": True})
    r = _turn("말하기 싫어")
    assert r["riskFlag"] is True
    assert r["reply"]["text"] == SAFETY_REPLY


def test_body_distress_goes_through_normal_path(monkeypatch):
    called = _stub(monkeypatch)
    r = _turn("숨이 차서 말을 못하겠네")
    assert r["avoided"] is False
    assert called["reply"] == 1
    assert r["replySource"] == "LLM"


@pytest.mark.parametrize("text", ["말도 마 손주가 와서 좋았어", "묻지 마 화투에서 이겼어"])
def test_not_avoiding_answers_get_model_reply(monkeypatch, text):
    called = _stub(monkeypatch)
    r = _turn(text, label="기쁨")
    assert r["avoided"] is False and called["reply"] == 1
    assert r["reply"]["text"] == LLM_REPLY.text


def test_switch_off_keeps_old_behavior(monkeypatch, settings):
    settings(companion_avoid_reply=False)
    called = _stub(monkeypatch)
    r = _turn("말하기 싫어")
    assert r["avoided"] is False and called["reply"] == 1
    assert r["replySource"] == "LLM"


# --- 라우터: 두 번 피하면 마무리 ---------------------------------------------------------------------

def _dialogue():
    return store.create("ward-avoid", "s1", "09:00", "잘 주무셨어요?", 8, "q1")


def _fake_run_turn(monkeypatch, avoided: bool, risk: bool = False):
    def run_turn(**kw):
        text = SAFETY_REPLY if risk else (
            fallback_options(kw["is_last_turn"], avoid=True)[kw["turn_no"] % 2] if avoided else "잘 주무셨군요?")
        return {"turnResult": {"label": "평온", "source": "TEXT"}, "textMood": None,
                "reply": {"text": text, "isFollowUp": not kw["is_last_turn"]}, "signals": None,
                "replySource": "SAFETY" if risk else ("AVOID" if avoided else "LLM"), "summary": None,
                "riskFlag": risk, "riskProbability": None, "avoided": avoided}
    monkeypatch.setattr(pipeline, "run_turn", run_turn)


def _turn_api(d, monkeypatch, avoided: bool, risk: bool = False) -> dict:
    _fake_run_turn(monkeypatch, avoided, risk)
    result = companion_router._run_turn(d, "말", {"label": None, "usable": False}, None, None)
    return companion_router._turn_response(store.get(d.dialogueId), result)


def test_second_avoid_ends_dialogue_with_closing_sentence(monkeypatch):
    d = _dialogue()
    first = _turn_api(d, monkeypatch, avoided=True)
    assert first["ended"] is False and first["avoided"] is True
    assert first["reply"]["text"] in FALLBACK_AVOID
    _turn_api(d, monkeypatch, avoided=False)  # 사이에 평범한 턴이 있어도 대화 안 횟수로 센다
    second = _turn_api(d, monkeypatch, avoided=True)
    assert second["ended"] is True
    assert second["endReason"] == "USER_STOP"
    assert second["reply"] == {"text": FALLBACK_LAST[0], "isFollowUp": False}


def test_risk_turn_is_not_counted_and_does_not_end(monkeypatch):
    d = _dialogue()
    _turn_api(d, monkeypatch, avoided=True)
    r = _turn_api(d, monkeypatch, avoided=True, risk=True)
    assert r["ended"] is False
    assert r["reply"]["text"] == SAFETY_REPLY
    assert store.get(d.dialogueId).avoidCount == 1


def test_end_count_zero_never_ends_early(monkeypatch, settings):
    settings(companion_avoid_end_count=0)
    d = _dialogue()
    for _ in range(3):
        r = _turn_api(d, monkeypatch, avoided=True)
        assert r["ended"] is False
        assert r["reply"]["text"] in FALLBACK_AVOID  # 반복되면 물러나는 문장 안에서 바꾸고, 다 쓰면 그대로 둔다


def test_repeated_avoid_sentence_is_replaced_within_avoid_list(monkeypatch, settings):
    settings(companion_avoid_end_count=0)
    d = _dialogue()
    d.replies.append(FALLBACK_AVOID[1])  # 첫 턴(turn_no=1)에 나갈 문장을 이미 썼다
    r = _turn_api(d, monkeypatch, avoided=True)
    assert r["reply"]["text"] == FALLBACK_AVOID[0]
