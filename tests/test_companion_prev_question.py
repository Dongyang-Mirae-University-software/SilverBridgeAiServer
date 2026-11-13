"""말벗 대답은 직전 말벗 문장을 질문으로 받고, 같은 질문을 되풀이하지 않는다 + 표정 한국어 이름 (2026-11-13).

11/12 실기기(BE 세션 203): 매 턴 프롬프트의 "질문:"에 여는 질문(아침 몸 상태)이 들어가 5번 턴 "계획 없어"에
2번 턴 질문("…특별히 불편하신 곳…")을 거의 그대로 되풀이했다. 3번 턴 "아니 없어"는 받아쓰기 확신도 때문에 되묻기로 빠졌다.
LLM·위험·신호·요약은 모두 스텁이다 (API 키를 쓰지 않는다).
"""
from __future__ import annotations

import re

import pytest

from app.core.config import get_settings
from app.routers import companion_router
from app.services.companion import pipeline
from app.services.companion.dialogue_store import store
from app.services.companion.face_service import FACE_LABEL_KO, FaceResult, face_label_ko
from app.services.companion.pipeline import LOW_CONFIDENCE_REPLY, is_short_plain_answer
from app.services.companion.reply_service import FALLBACK_FOLLOWUP, generate_reply, reply_similarity
from app.services.companion.text_emotion_service import TextEmotionResult
from app.services.companion.turn_record import build_dialogue_result

OPENING = "좋은 아침이에요. 아침에 일어나셨을 때 몸은 좀 어떠셨어요?"
# 세션 203 실제 대답 (말벗 문장 1~4·6은 실제 문장, 5번 첫 생성은 실제로 나간 되풀이 문장)
R1 = "아침부터 피곤하시다니 안타깝네요. 혹시 특별히 힘든 일이 있으셨나요?"
R2 = "아침에 몸 상태가 좋지 않으셨군요. 혹시 특별히 불편하신 곳이 있으셨나요?"
R3 = "아니라고 하시니 다행이에요. 오늘은 무엇을 하실 생각이세요?"
R4 = "아침에 몸 상태가 괜찮으셨다니 다행이네요. 오늘 하루는 어떻게 보내실 계획이세요?"
R5_REPEAT = "아침에 몸 상태가 좀 불편하셨군요. 특별히 불편하신 곳은 없으셨나요?"
R5_NEW = "계획이 없으셔도 괜찮아요. 오늘은 편히 쉬시면서 좋아하시는 걸 하시면 어떠세요?"
R6 = "하루 계획이 없으시다니 여유로운 하루가 되실 것 같네요. 오늘은 무엇이 제일 하고 싶으세요?"


@pytest.fixture
def settings(monkeypatch):
    s = get_settings()

    def set_(**kw):
        for k, v in kw.items():
            monkeypatch.setattr(s, k, v)
    set_(companion_confirm_mode="logprob", companion_confirm_logprob=-0.3, companion_short_answer_pass=True,
         companion_reply_similar_ratio=0.65, companion_reply_redo_temperature=0.7, companion_llm_retry=0,
         companion_risk_first=True, companion_reply_relax_calm=True, companion_avoid_reply=True,
         companion_keep_fixed_replies=True, companion_llm_temperature=0.1)
    return set_


class ScriptedLlm:
    """대답 생성만 하는 가짜 LLM. 부른 프롬프트·온도를 남기고, 정해 둔 문장을 차례로 낸다."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[tuple[str, float]] = []

    def generate(self, prompt, **kw):
        self.calls.append((prompt, kw.get("temperature")))
        return self.replies.pop(0)


def _question_in(prompt: str) -> str:
    return re.search(r"^질문: (.*)$", prompt, re.M).group(1)


@pytest.fixture
def stubs(monkeypatch):
    monkeypatch.setattr(pipeline, "extract_signals", lambda *a, **k: {"riskFlag": False})
    monkeypatch.setattr(pipeline, "summarize_answer", lambda *a, **k: "요약")
    monkeypatch.setattr(pipeline, "_risk_prob_capped", lambda *a, **k: 0.01)
    monkeypatch.setattr(pipeline, "risk_probability", lambda *a, **k: 0.01)


def _calm(_text):
    return TextEmotionResult("CALM", 0.9, True, {})


NO_FACE = {"label": None, "usable": False}


def _say(d, llm, text, logprob=-0.1):
    result = companion_router._run_turn(d, text, NO_FACE, _calm, llm, stt_logprob=logprob)
    return companion_router._turn_response(store.get(d.dialogueId), result, face=NO_FACE)


# --- ① 질문 = 직전 말벗 문장 · ② 되풀이 막기 · ③ 짧은 부정 대답 (세션 203 재현) ---------------------------

def test_session_203_replay(settings, stubs):
    d = store.create("ward-203", "s203", "09:00", OPENING, 8, "q203")
    # 3번 턴 "아니 없어"는 실제로 확신도가 낮았다 (값은 로그에 없어 임계 아래 아무 값)
    turns = [("아침에 일어났을 때 너무 피곤했어", -0.1), ("알 거 없잖아", -0.1), ("아니 없어", -0.9),
             ("없다고", -0.1), ("계획 없어", -0.1), ("오늘은 하루 보낼 거 계획 없다니까", -0.1)]
    llm = ScriptedLlm([R1, R2, R3, R4, R5_REPEAT, R5_NEW, R6])
    expected_questions = [OPENING, R1, R2, R3, R4, R5_NEW]
    outs = []
    for i, (text, lp) in enumerate(turns):
        before = len(llm.calls)
        out = _say(d, llm, text, lp)
        outs.append(out)
        prompts = [p for p, _ in llm.calls[before:]]
        assert prompts, f"{i + 1}번 턴에 대답을 만들지 않았다"
        for p in prompts:
            assert _question_in(p) == expected_questions[i]  # 매 턴 "질문:" = 직전 말벗 문장 (첫 턴은 여는 질문)
    texts = [o["reply"]["text"] for o in outs]
    assert texts[2] != LOW_CONFIDENCE_REPLY  # ③ "아니 없어"는 되묻지 않는다
    assert texts[4] == R5_NEW  # ② 5번 첫 생성은 2번과 비슷해 한 번 다시 만들었다
    assert llm.calls[5][1] == 0.7 and llm.calls[4][1] == 0.1  # 다시 만들 때만 온도를 올린다
    assert reply_similarity(texts[4], texts[1]) < 0.65
    assert not llm.replies  # 정해 둔 문장을 다 썼다 (재생성은 5번 턴 한 번뿐)


def test_confirm_turn_keeps_previous_question(settings, stubs):
    d = store.create("ward-203b", "s203b", "09:00", OPENING, 8, None)
    llm = ScriptedLlm([R1, R2])
    _say(d, llm, "아침에 일어났을 때 너무 피곤했어")
    confirm = _say(d, llm, "어버버 그거 저기", logprob=-0.9)  # 확신도 낮은 짧지 않은 말 → 되묻기
    assert confirm["reply"]["text"] == LOW_CONFIDENCE_REPLY
    assert store.get(d.dialogueId).lastQuestion == R1  # 되묻는 문장으로 바꾸지 않는다
    _say(d, llm, "힘든 일은 없었어")
    assert _question_in(llm.calls[-1][0]) == R1  # 다시 말해 준 대답은 그 전 질문(R1)에 대한 것


def test_opening_question_is_used_on_first_turn(settings, stubs):
    d = store.create("ward-203c", "s203c", "09:00", OPENING, 8, None)
    llm = ScriptedLlm([R1])
    _say(d, llm, "아침에 일어났을 때 너무 피곤했어")
    assert _question_in(llm.calls[0][0]) == OPENING


def test_still_similar_after_redo_uses_fallback(settings, stubs):
    llm = ScriptedLlm([R5_REPEAT, R5_REPEAT])
    r = generate_reply(llm, R4, "계획 없어", None, False, 5,
                       {"usable": True, "turnLabel": "평온"}, past_replies=(OPENING, R1, R2, R4))
    assert r.source == "FALLBACK" and r.failures == ["반복"]
    assert r.text in FALLBACK_FOLLOWUP
    assert len(llm.calls) == 2  # 다시 만들기는 한 번뿐


def test_not_similar_reply_is_used_without_redo(settings, stubs):
    llm = ScriptedLlm([R5_NEW])
    r = generate_reply(llm, R4, "계획 없어", None, False, 5,
                       {"usable": True, "turnLabel": "평온"}, past_replies=(OPENING, R1, R2, R4))
    assert r.source == "LLM" and r.text == R5_NEW and len(llm.calls) == 1


def test_similar_ratio_zero_turns_off_redo(settings, stubs):
    settings(companion_reply_similar_ratio=0)
    llm = ScriptedLlm([R5_REPEAT])
    r = generate_reply(llm, R4, "계획 없어", None, False, 5,
                       {"usable": True, "turnLabel": "평온"}, past_replies=(R2,))
    assert r.source == "LLM" and r.text == R5_REPEAT


def test_session_203_similarity_numbers():
    assert reply_similarity(R2, R5_REPEAT) >= 0.65  # 되풀이 (0.76)
    for a, b in [(R1, R2), (R2, R4), (R4, R6), (R1, R5_REPEAT)]:
        assert reply_similarity(a, b) < 0.65  # 서로 다른 질문은 다시 만들지 않는다


@pytest.mark.parametrize("text", ["아니 없어", "없어", "없다고", "아니요.", "괜찮아요", "몰라", "별로 없어", "아뇨, 없어요"])
def test_short_negative_answers_pass(text):
    assert is_short_plain_answer(text)


@pytest.mark.parametrize("text", ["", "네", "응", "어", "계획 없어", "아니요 없어요 괜찮아요 몰라요", "어버버 그거"])
def test_other_short_utterances_are_not_short_answers(text):
    assert not is_short_plain_answer(text)


def test_short_answer_switch_off_keeps_reasking(settings, stubs):
    settings(companion_short_answer_pass=False)
    r = pipeline.run_turn("아니 없어", NO_FACE, _calm, ScriptedLlm([]), R2, None, False, 3, stt_logprob=-0.9)
    assert r["reply"]["text"] == LOW_CONFIDENCE_REPLY


def test_short_answer_still_gets_risk_check(settings, stubs, monkeypatch):
    monkeypatch.setattr(pipeline, "keyword_risk", lambda t: True)
    r = pipeline.run_turn("아니 없어", NO_FACE, _calm, ScriptedLlm([R3]), R2, None, False, 3, stt_logprob=-0.9)
    assert r["riskFlag"] is True and r["replySource"] == "SAFETY"


# --- ④ 표정 한국어 이름 ---------------------------------------------------------------------------------

@pytest.mark.parametrize("label,ko", [("Neutral", "무표정"), ("Happiness", "기쁨"), ("Sadness", "슬픔"),
                                      ("Anger", "화남"), ("Fear", "두려움"), ("Surprise", "놀람"),
                                      ("Disgust", "싫어함"), ("Contempt", "못마땅함")])
def test_face_label_ko_mapping(label, ko):
    assert face_label_ko(label) == ko
    f = companion_router._face_dict(FaceResult(label, 0.8, 5, True, 6, 5))
    assert f["labelKo"] == ko and f["label"] == label  # 영어 라벨은 그대로


def test_face_label_ko_covers_exactly_eight():
    assert len(FACE_LABEL_KO) == 8


def test_unknown_or_missing_face_is_null():
    assert face_label_ko("Confused") is None and face_label_ko(None) is None
    assert companion_router._face_dict(FaceResult("Confused", 0.8, 5, True, 6, 5))["labelKo"] is None
    assert companion_router._face_dict(FaceResult(None, 0.0, 0, False, 6, 0))["labelKo"] is None
    # 프레임이 모자라 쓸 수 없는 표정은 이름을 채우지 않는다
    assert companion_router._face_dict(FaceResult("Neutral", 0.8, 1, False, 6, 1))["labelKo"] is None
    no_face = companion_router._no_face()
    assert no_face["labelKo"] is None and no_face["label"] is None


def test_face_dict_keeps_existing_fields():
    f = companion_router._face_dict(FaceResult("Sadness", 0.7, 4, True, 6, 4))
    assert set(f) == {"label", "labelKo", "confidence", "frameCount", "usable", "totalFrames", "facesFound"}
    assert set(companion_router._no_face()) == set(f)


def test_dialogue_result_has_korean_dominant_face(settings, stubs):
    d = store.create("ward-face", "sface", "09:00", OPENING, 8, None)
    llm = ScriptedLlm([R1, R2])
    face = companion_router._face_dict(FaceResult("Sadness", 0.7, 4, True, 6, 4))
    for text in ("아침에 일어났을 때 너무 피곤했어", "힘든 일은 없었어"):
        result = companion_router._run_turn(d, text, face, _calm, llm, stt_logprob=-0.1)
        out = companion_router._turn_response(store.get(d.dialogueId), result, face=face)
        assert out["face"]["labelKo"] == "슬픔"
    res = build_dialogue_result(store.get(d.dialogueId))
    assert res["dominantFace"] == "Sadness" and res["dominantFaceKo"] == "슬픔"
    assert res["faceCounts"] == {"Sadness": 2}  # 키는 영어 그대로


def test_dialogue_result_without_face_has_null_korean_name():
    d = store.create("ward-noface", "snoface", "09:00", OPENING, 8, None)
    res = build_dialogue_result(d)
    assert res["dominantFace"] is None and res["dominantFaceKo"] is None
