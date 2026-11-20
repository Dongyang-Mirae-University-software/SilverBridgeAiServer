"""말벗 대답에 대화 흐름(말벗 문장 + 어르신 말 한 줄 요약)을 넣는다 - A안 (2026-11-15, COMPANION_REPLY_HISTORY).

11/15 친구 dev 서버 실사용: 마지막 인사가 "어떤 음식을 생각하고 계신가요?" → "김치찌개."에
"김치찌개 잘 드셨네요… 보내셨다니"로 답했다. 마지막 인사 프롬프트(reply-v1-last)에 직전 질문이 없었고,
꼬리질문도 직전 말벗 문장 하나와 이번 대답만 보고 만들었다.
LLM·위험·신호·요약은 모두 스텁이다 (API 키를 쓰지 않는다).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.core.config import get_settings
from app.routers import companion_router
from app.services.companion import pipeline, reply_service
from app.services.companion.dialogue_store import store
from app.services.companion.pipeline import LOW_CONFIDENCE_REPLY
from app.services.companion.reply_service import (
    HISTORY_EMPTY,
    HISTORY_NO_ANSWER,
    HISTORY_NO_SUMMARY,
    HISTORY_UNCLEAR,
    build_reply_prompt,
    generate_reply,
    history_lines,
    reply_prompt_version,
)
from app.services.companion.text_emotion_service import TextEmotionResult
from app.services.companion.turn_record import build_dialogue_result

PROMPTS = Path(reply_service.__file__).parent / "prompts"
OPENING = "점심은 어디서, 누구와 드셨어요?"
R1 = "혼자 드셨군요. 국수는 직접 만드셨어요?"
R2 = "자주 해 드시지는 않으시군요. 어떤 음식을 생각하고 계신가요?"
R3 = "김치찌개 생각하고 계셨군요. 오늘도 이야기 나눠 주셔서 고맙습니다. 편안한 하루 보내세요."
A1, A2, A3 = "집에서 혼자 국수 먹었어", "자주는 안 해 먹어", "김치찌개."
SUMMARY = {A1: "집에서 혼자 국수를 먹음", A2: "자주 해 먹지는 않음", A3: "김치찌개를 생각함"}


@pytest.fixture
def settings(monkeypatch):
    s = get_settings()

    def set_(**kw):
        for k, v in kw.items():
            monkeypatch.setattr(s, k, v)
    set_(companion_reply_history="summary", companion_reply_history_turns=6,
         companion_confirm_mode="logprob", companion_confirm_logprob=-0.3, companion_short_answer_pass=True,
         companion_reply_similar_ratio=0.65, companion_reply_redo_temperature=0.7, companion_llm_retry=0,
         companion_risk_first=True, companion_reply_relax_calm=True, companion_avoid_reply=True,
         companion_keep_fixed_replies=True, companion_llm_temperature=0.1, companion_flat_topic_switch=0)
    return set_


class ScriptedLlm:
    """대답 생성만 하는 가짜 LLM. 부른 프롬프트를 남기고 정해 둔 문장을 차례로 낸다."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def generate(self, prompt, **kw):
        self.prompts.append(prompt)
        return self.replies.pop(0)


@pytest.fixture
def stubs(monkeypatch):
    monkeypatch.setattr(pipeline, "extract_signals", lambda *a, **k: {"riskFlag": False})
    monkeypatch.setattr(pipeline, "summarize_answer", lambda llm, t: SUMMARY.get(t, "요약"))
    monkeypatch.setattr(pipeline, "_risk_prob_capped", lambda *a, **k: 0.01)
    monkeypatch.setattr(pipeline, "risk_probability", lambda *a, **k: 0.01)


def _calm(_text):
    return TextEmotionResult("CALM", 0.9, True, {})


NO_FACE = {"label": None, "usable": False}


def _say(d, llm, text, logprob=-0.1):
    result = companion_router._run_turn(d, text, NO_FACE, _calm, llm, stt_logprob=logprob)
    return companion_router._turn_response(store.get(d.dialogueId), result, face=NO_FACE)


def _history_block(prompt: str) -> str:
    return re.search(r"지금까지의 대화 \(어르신 말은 한 줄 요약\):\n(.*?)\n\n질문:", prompt, re.S).group(1)


# --- 흐름 줄 ------------------------------------------------------------------------------------------

def test_history_lines_format_and_limit():
    pairs = ((OPENING, "국수를 먹음"), (R1, None), (R2, "김치찌개"))
    assert history_lines((), 6) == HISTORY_EMPTY
    assert history_lines(pairs, 0).splitlines() == [
        f"- 말벗: {OPENING}", "- 어르신: 국수를 먹음", f"- 말벗: {R1}", f"- 어르신: {HISTORY_NO_SUMMARY}",
        f"- 말벗: {R2}", "- 어르신: 김치찌개"]
    assert history_lines(pairs, 1).splitlines() == [f"- 말벗: {R2}", "- 어르신: 김치찌개"]  # 최근 1쌍만
    assert history_lines(((" 줄\n바꿈  있는 문장 ", "요약\n둘째 줄"),), 6) == "- 말벗: 줄 바꿈 있는 문장\n- 어르신: 요약 둘째 줄"


# --- 프롬프트 -----------------------------------------------------------------------------------------

def test_followup_prompt_has_history_question_and_answer(settings):
    p = build_reply_prompt(R1, A2, None, False, ((OPENING, SUMMARY[A1]),))
    assert "{{" not in p
    assert _history_block(p).splitlines() == [f"- 말벗: {OPENING}", f"- 어르신: {SUMMARY[A1]}"]
    assert f"질문: {R1}" in p and f"어르신의 대답: {A2}" in p
    assert "이미 나눈 이야기를 다시 묻지 않습니다" in p


def test_first_turn_prompt_says_no_history_yet(settings):
    p = build_reply_prompt(OPENING, A1, None, False, ())
    assert _history_block(p) == HISTORY_EMPTY and f"질문: {OPENING}" in p


def test_last_prompt_now_has_previous_question(settings):
    # 11/15 김치찌개: 마지막 인사도 '질문'을 보고 만든다 + 하지 않은 일을 넘겨짚지 않는다
    p = build_reply_prompt(R2, A3, None, True, ((OPENING, SUMMARY[A1]), (R1, SUMMARY[A2])))
    assert "{{" not in p
    assert f"질문: {R2}" in p and f"어르신의 마지막 대답: {A3}" in p
    assert "하지 않은 일을 했다고 말하지 않습니다" in p
    assert f"- 어르신: {SUMMARY[A2]}" in _history_block(p)


def test_off_mode_is_exactly_the_old_prompts(settings):
    settings(companion_reply_history="off")
    signals = {"emotion": "neutral"}
    old_follow = (PROMPTS / "reply-v2-followup.txt").read_text(encoding="utf-8")
    old_last = (PROMPTS / "reply-v1-last.txt").read_text(encoding="utf-8")
    line = "직전 턴 신호: emotion=neutral"
    assert build_reply_prompt(R1, A2, signals, False, ((OPENING, "x"),)) == (
        old_follow.replace("{{PREV_SIGNALS}}", line).replace("{{QUESTION}}", R1).replace("{{TRANSCRIPT}}", A2))
    assert build_reply_prompt(R2, A3, signals, True, ((OPENING, "x"),)) == (
        old_last.replace("{{PREV_SIGNALS}}", line).replace("{{TRANSCRIPT}}", A3))
    assert reply_prompt_version() == "reply-v2"


def test_prompt_version_follows_switch(settings):
    assert reply_prompt_version() == "reply-v3"
    d = store.create("ward-hist-ver", "s-ver", "12:00", OPENING, 8, None)
    assert build_dialogue_result(d)["promptVersion"]["reply"] == "reply-v3"
    settings(companion_reply_history="off")
    assert build_dialogue_result(d)["promptVersion"]["reply"] == "reply-v2"


# --- 라우터에서 턴을 이어 갈 때 ------------------------------------------------------------------------

def test_history_grows_turn_by_turn_with_summaries_only(settings, stubs):
    d = store.create("ward-hist", "s-hist", "12:00", OPENING, 3, None)  # 3번째 턴이 마지막 인사
    llm = ScriptedLlm([R1, R2, R3])
    outs = [_say(d, llm, a) for a in (A1, A2, A3)]
    assert [o["reply"]["text"] for o in outs] == [R1, R2, R3] and outs[-1]["ended"] is True
    p1, p2, p3 = llm.prompts
    assert _history_block(p1) == HISTORY_EMPTY
    assert _history_block(p2).splitlines() == [f"- 말벗: {OPENING}", f"- 어르신: {SUMMARY[A1]}"]
    assert _history_block(p3).splitlines() == [f"- 말벗: {OPENING}", f"- 어르신: {SUMMARY[A1]}",
                                               f"- 말벗: {R1}", f"- 어르신: {SUMMARY[A2]}"]
    assert f"질문: {R2}" in p3 and "어르신의 마지막 대답: 김치찌개." in p3  # 마지막 인사도 직전 질문을 본다
    # 지난 턴의 어르신 대답 원문은 프롬프트에 없다 (요약만)
    assert A1 not in p2 and A1 not in p3 and A2 not in p3


def test_history_marks_no_response_and_unclear_turns(settings, stubs):
    d = store.create("ward-hist2", "s-hist2", "12:00", OPENING, 8, None)
    llm = ScriptedLlm([R1])
    silent = _say(d, llm, "")  # 무응답 턴 (모델을 부르지 않는다)
    unclear = _say(d, llm, "어버버 그거 저기", logprob=-0.9)  # 확신도 낮음 → 되묻기 (모델을 부르지 않는다)
    assert unclear["reply"]["text"] == LOW_CONFIDENCE_REPLY and not llm.prompts
    _say(d, llm, A1)
    lines = _history_block(llm.prompts[0]).splitlines()
    assert lines == [f"- 말벗: {OPENING}", f"- 어르신: {HISTORY_NO_ANSWER}",
                     f"- 말벗: {silent['reply']['text']}", f"- 어르신: {HISTORY_UNCLEAR}"]


def test_history_turns_limit_in_router(settings, stubs):
    settings(companion_reply_history_turns=1)
    d = store.create("ward-hist3", "s-hist3", "12:00", OPENING, 8, None)
    llm = ScriptedLlm([R1, R2, "김치찌개는 직접 끓이세요?"])
    for a in (A1, A2, A3):
        _say(d, llm, a)
    assert _history_block(llm.prompts[2]).splitlines() == [f"- 말벗: {R1}", f"- 어르신: {SUMMARY[A2]}"]


def test_off_mode_router_prompt_has_no_history(settings, stubs):
    settings(companion_reply_history="off")
    d = store.create("ward-hist4", "s-hist4", "12:00", OPENING, 8, None)
    llm = ScriptedLlm([R1, R2])
    _say(d, llm, A1)
    _say(d, llm, A2)
    assert "지금까지의 대화" not in llm.prompts[1] and SUMMARY[A1] not in llm.prompts[1]


# --- 안전장치: 질문은 여전히 이번 대답을 이어 받아야 한다 ----------------------------------------------

def test_reply_that_only_follows_history_is_still_unrelated(settings):
    # 흐름에만 있는 '국수'를 다시 묻고 이번 대답(자주 안 해 먹어)을 이어 받지 않은 문장 → 전과 같이 '대답과 무관'
    llm = ScriptedLlm(["국수는 어떻게 만드셨어요?"])
    r = generate_reply(llm, R1, A2, None, False, 2, None, history=((OPENING, SUMMARY[A1]),))
    assert r.source == "FALLBACK" and "대답과 무관" in r.failures


def test_default_is_off():
    """11/20: 품질 세트 재생에서 summary 가 off 보다 대답 실패가 많아(33.3% vs 28.4%) 기본은 off. 설정으로만 켠다."""
    from app.core.config import Settings
    assert Settings.model_fields["companion_reply_history"].default == "off"
