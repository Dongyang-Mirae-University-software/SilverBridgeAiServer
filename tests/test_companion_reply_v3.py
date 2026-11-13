"""꼬리질문 프롬프트 v3 - 이 대화에서 앞서 한 질문을 넣는다 (2026-11-13, 대답 LoRA cc8 과 함께 쓴다).

11/13 실기기(BE sessionId=212)·DGX 재생: 짧은 대답이 이어지면 cc7 이 "좋아하시는 활동"을 30턴 중 15번 물었다.
생성 뒤 거르기로는 바뀌지 않아 대답 모델이 앞선 질문을 보도록 다시 학습했다(cc8, reply-v3 형식).
LLM·위험·신호·요약은 스텁이다 (API 키를 쓰지 않는다).
"""
from __future__ import annotations

import re

import pytest

from app.core.config import get_settings
from app.routers import companion_router
from app.services.companion import pipeline
from app.services.companion.dialogue_store import store
from app.services.companion.pipeline import LOW_CONFIDENCE_REPLY
from app.services.companion.reply_service import (
    FALLBACK_FOLLOWUP,
    SAFETY_REPLY,
    build_reply_prompt,
    past_questions_block,
    question_sentence,
    reply_prompt_version,
)
from app.services.companion.text_emotion_service import TextEmotionResult

OPENING = "좋은 아침이에요. 아침에 일어나셨을 때 몸은 좀 어떠셨어요?"
R1 = "아침부터 피곤하시다니 안타깝네요. 혹시 특별히 힘든 일이 있으셨나요?"
R2 = "그런 날도 있으시죠. 오늘 하루는 좀 더 편안하게 보내셨나요?"
R3 = "그렇군요. 어젯밤에는 잘 주무셨어요?"
R4 = "다행이네요. 요즘 자주 연락하시는 분은 계세요?"


@pytest.fixture
def settings(monkeypatch):
    s = get_settings()

    def set_(**kw):
        for k, v in kw.items():
            monkeypatch.setattr(s, k, v)
    set_(companion_reply_prompt="v3", companion_confirm_mode="logprob", companion_confirm_logprob=-0.3,
         companion_reply_similar_ratio=0.65, companion_llm_retry=0, companion_risk_first=True,
         companion_reply_relax_calm=True, companion_avoid_reply=True, companion_reply_lora=False)
    return set_


class Llm:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt, **kw):
        self.prompts.append(prompt)
        return self.replies.pop(0)


@pytest.fixture
def stubs(monkeypatch):
    monkeypatch.setattr(pipeline, "extract_signals", lambda *a, **k: {"riskFlag": False})
    monkeypatch.setattr(pipeline, "summarize_answer", lambda *a, **k: "요약")
    monkeypatch.setattr(pipeline, "_risk_prob_capped", lambda *a, **k: 0.01)
    monkeypatch.setattr(pipeline, "risk_probability", lambda *a, **k: 0.01)


def _calm(_t):
    return TextEmotionResult("CALM", 0.9, True, {})


NO_FACE = {"label": None, "usable": False}


def _past_block(prompt: str) -> list[str]:
    block = re.search(r"앞서 한 질문:\n(.*?)\n질문:", prompt, re.S).group(1)
    return [] if block == "없음" else [line[2:] for line in block.split("\n")]


def _say(d, llm, text, logprob=-0.1):
    result = companion_router._run_turn(d, text, NO_FACE, _calm, llm, stt_logprob=logprob)
    companion_router._turn_response(store.get(d.dialogueId), result, face=NO_FACE)
    return result


# --- 프롬프트 ------------------------------------------------------------------------------------

def test_v2_prompt_is_unchanged(settings):
    settings(companion_reply_prompt="v2")
    p = build_reply_prompt(R2, "몰라", None, False, [OPENING, R1])
    assert "앞서 한 질문" not in p and f"질문: {R2}" in p
    assert reply_prompt_version() == "reply-v2"


def test_v3_prompt_lists_question_sentences(settings):
    p = build_reply_prompt(R2, "몰라", None, False, [OPENING, R1])
    assert _past_block(p) == ["아침에 일어나셨을 때 몸은 좀 어떠셨어요?", "혹시 특별히 힘든 일이 있으셨나요?"]
    assert "앞서 한 질문과 같은 것을 다시 묻지 마세요." in p
    assert p.rstrip().endswith("어르신의 대답: 몰라") and f"질문: {R2}" in p
    assert reply_prompt_version() == "reply-v3"


def test_v3_prompt_without_past_says_none(settings):
    assert _past_block(build_reply_prompt(OPENING, "피곤했어", None, False, [])) == []
    assert "앞서 한 질문:\n없음\n" in build_reply_prompt(OPENING, "피곤했어", None, False)


def test_v3_keeps_three_most_recent(settings):
    assert past_questions_block([OPENING, R1, R2, R3]).split("\n") == [
        f"- {question_sentence(x)}" for x in (R1, R2, R3)]


def test_sentences_without_question_are_skipped(settings):
    assert past_questions_block(["말씀해 주셔서 고맙습니다.", ""]) == "없음"
    assert question_sentence("네, 잘 들었어요. 더 하고 싶으신 말씀 있으세요?") == "더 하고 싶으신 말씀 있으세요?"


def test_last_turn_uses_closing_prompt_even_on_v3(settings):
    p = build_reply_prompt(R2, "몰라", None, True, [OPENING, R1])
    assert "앞서 한 질문" not in p


# --- 대화 흐름: 앞서 한 질문 = 여는 질문 + 모델 대답 (직전 질문 제외) -------------------------------------

def test_dialogue_passes_earlier_questions_only(settings, stubs):
    d = store.create("ward-v3", "sv3", "09:00", OPENING, 8, None)
    llm = Llm([R1, R2, R3])
    for text in ("아침에 일어났을 때 너무 피곤했어", "알 거 없잖아", "그냥 그래"):
        _say(d, llm, text)
    assert _past_block(llm.prompts[0]) == []  # 첫 턴: 직전 질문 = 여는 질문, 그 앞은 없음
    assert _past_block(llm.prompts[1]) == ["아침에 일어나셨을 때 몸은 좀 어떠셨어요?"]
    assert _past_block(llm.prompts[2]) == ["아침에 일어나셨을 때 몸은 좀 어떠셨어요?", "혹시 특별히 힘든 일이 있으셨나요?"]
    assert store.get(d.dialogueId).askedQuestions == [OPENING, R1, R2, R3]


def test_confirm_and_fixed_replies_are_not_past_questions(settings, stubs, monkeypatch):
    d = store.create("ward-v3b", "sv3b", "09:00", OPENING, 8, None)
    llm = Llm([R1, "물음 없는 대답", R4])  # 두 번째 대답은 규칙 위반 → 대체 문장
    _say(d, llm, "아침에 일어났을 때 너무 피곤했어")
    r = _say(d, llm, "어버버 그거 저기", logprob=-0.9)  # 되묻기
    assert r["reply"]["text"] == LOW_CONFIDENCE_REPLY
    r = _say(d, llm, "힘든 일은 없었어")
    assert r["reply"]["text"] in FALLBACK_FOLLOWUP
    _say(d, llm, "그냥 그래")
    asked = store.get(d.dialogueId).askedQuestions
    assert asked == [OPENING, R1, R4]
    assert LOW_CONFIDENCE_REPLY not in asked and not set(asked) & set(FALLBACK_FOLLOWUP)


def test_safety_reply_is_not_past_question(settings, stubs, monkeypatch):
    monkeypatch.setattr(pipeline, "keyword_risk", lambda t: True)
    d = store.create("ward-v3c", "sv3c", "09:00", OPENING, 8, None)
    r = _say(d, Llm([R1]), "다 그만두고 싶어")
    assert r["reply"]["text"] == SAFETY_REPLY or r["replySource"] == "SAFETY"
    assert store.get(d.dialogueId).askedQuestions == [OPENING]


def test_review_version_follows_setting(settings, stubs):
    d = store.create("ward-v3d", "sv3d", "09:00", OPENING, 8, None)
    _say(d, Llm([R1]), "아침에 일어났을 때 너무 피곤했어")
    from app.services.companion.turn_record import build_dialogue_result
    assert build_dialogue_result(store.get(d.dialogueId))["promptVersion"]["reply"] == "reply-v3"
