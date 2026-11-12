"""말벗 평온·기쁨 턴의 대답 규칙 완화 (2026-11-12, COMPANION_REPLY_RELAX_CALM).

DGX 측정(11/11, CareCall 200): '대답과 무관'(어르신 대답 낱말을 이어 받지 않으면 탈락)만으로 대체된 62개 중
약 50개가 멀쩡한 대답이었다("아니 못잤어" → "못 주무셨다니 안타깝네요. 오늘 아침은 어떻게 시작하셨나요?").
글 기분이 평온·기쁨인 턴만 그 검사를 빼고, 대신 권유 문장을 거르는 '권유' 검사를 꼬리질문 턴 전체에 더한다.
힘든 이야기(걱정·우울) 턴은 풀면 질문 주제에 끌려간 오해가 살아나서 전 규칙 그대로다.
"""
from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.services.companion import pipeline
from app.services.companion.reply_service import (
    FALLBACK_EMPATHY,
    FALLBACK_FOLLOWUP,
    FALLBACK_LAST,
    FALLBACK_NO_RESPONSE,
    check_reply,
    generate_reply,
    is_relaxed_mood,
)
from app.services.companion.text_emotion_service import TextEmotionResult

CALM = {"turnLabel": "평온", "usable": True}
JOY = {"turnLabel": "기쁨", "usable": True}
DEPRESSED = {"turnLabel": "우울", "usable": True}
TRANSCRIPT = "아니 못잤어"
# 측정에서 멀쩡했는데 '대답과 무관'으로 대체되던 대답 (대답 낱말 '못잤어'의 앞 두 글자 '못잤'이 없다)
UNRELATED_OK = "못 주무셨다니 안타깝네요. 오늘 아침은 어떻게 시작하셨나요?"
ADVICE = (
    "잠이 안 오실 땐 가벼운 스트레칭을 해 보시는 건 어떠신가요?",
    "따뜻한 차를 드셔 보시는 건 어떨까요?",
    "자기 전에 따뜻한 물로 샤워를 해 보세요. 괜찮으세요?",
    "낮에 산책을 하시면 좋을 것 같아요. 괜찮으세요?",
    "따님께 연락해 보시는 건 어떠세요?",
    "가벼운 산책 어떠세요?",
    "음악을 들어 보시면 어떨까요?",
    "병원에 가 보시는 건 어떨까요?",
    "자주 물을 드시는 건 어떠세요?",
    "산책하시는 건 어떠세요?",
    "명상 한번 어떠세요?",
    # DGX 대조(11/12)에서 좁힌 첫 정규식이 놓친 실제 권유 대답
    "허리가 아프셔서 운동이 어려우시군요. 집에서 할 수 있는 가벼운 스트레칭은 어떠신가요?",
    "혹시 가까운 이웃이나 친구분께 연락해 보는 건 어떨까요?",
)
NOT_ADVICE = (
    "손주와 함께 하는 시간은 어떠신가요?",
    "못 주무셨다니 마음이 쓰이네요. 요즘 기분은 어떠세요?",
    # 11/12 점검 M-1: 권유 낱말이 있어도 어르신 이야기를 이어 가는 물음은 통과해야 한다
    "요즘 운동은 어떠세요?",
    "산책 다녀오시니 기분은 어떠세요?",
    "음악 들으실 때 기분은 어떠신가요?",
    "그 이야기 조금 더 말씀해 보세요?",
    "약은 잘 드셔 보시니 어떠세요?",
    "그렇게 하시면 좋을까요?",
    "오늘 산책은 어떠셨나요?",
    "따님께 연락해 보셨어요?",
    "명상 같은 걸 해 보신 적 있으세요?",
    "천천히 생각해 보시고 말씀해 주시겠어요?",
    "운동하시고 나면 몸은 좀 어떠세요?",
    "요즘 주무시는 건 어떠세요?",
)


@pytest.fixture
def settings(monkeypatch):
    s = get_settings()

    def set_(**kw):
        for k, v in kw.items():
            monkeypatch.setattr(s, k, v)
    return set_


def test_switch_is_on_by_default():
    assert get_settings().companion_reply_relax_calm is True


@pytest.mark.parametrize("mood", [CALM, JOY])
def test_calm_or_joy_turn_passes_without_echoing_answer_words(mood):
    assert check_reply(UNRELATED_OK, is_last_turn=False, transcript=TRANSCRIPT, text_mood=mood) == []


@pytest.mark.parametrize("mood", [
    None,
    {},
    DEPRESSED,
    {"turnLabel": "슬픔", "usable": True},
    {"turnLabel": "화남", "usable": True},
    {"turnLabel": "평온", "usable": False},  # 판단 불가(짧은 말·낮은 확신)는 기분으로 보지 않는다
])
def test_other_moods_keep_the_unrelated_rule(mood):
    assert check_reply(UNRELATED_OK, is_last_turn=False, transcript=TRANSCRIPT, text_mood=mood) == ["대답과 무관"]


@pytest.mark.parametrize("mood", [None, CALM, DEPRESSED])
@pytest.mark.parametrize("sentence", ADVICE)
def test_advice_sentences_fail_on_every_follow_up_turn(sentence, mood):
    assert "권유" in check_reply(sentence, is_last_turn=False, transcript="", text_mood=mood)


@pytest.mark.parametrize("sentence", NOT_ADVICE)
def test_question_without_advice_word_passes(sentence):
    assert check_reply(sentence, is_last_turn=False, transcript="", text_mood=CALM) == []


def test_last_turn_is_not_checked_for_advice():
    assert check_reply("푹 쉬시고 산책도 해 보세요.", is_last_turn=True) == []


def test_switch_off_restores_previous_rules(settings):
    settings(companion_reply_relax_calm=False)
    assert check_reply(UNRELATED_OK, is_last_turn=False, transcript=TRANSCRIPT, text_mood=CALM) == ["대답과 무관"]
    assert check_reply(ADVICE[0], is_last_turn=False, transcript="", text_mood=CALM) == []
    assert not is_relaxed_mood(None)


@pytest.mark.parametrize("sentence", FALLBACK_FOLLOWUP + FALLBACK_EMPATHY + FALLBACK_NO_RESPONSE)
@pytest.mark.parametrize("mood", [None, CALM, DEPRESSED])
def test_every_fallback_passes_the_new_rules(sentence, mood):
    assert check_reply(sentence, is_last_turn=False, text_mood=mood) == [], sentence


@pytest.mark.parametrize("sentence", FALLBACK_LAST)
def test_closing_fallback_passes_the_new_rules(sentence):
    assert check_reply(sentence, is_last_turn=True, text_mood=CALM) == []


class _FixedLlm:
    """늘 같은 문장을 돌려주는 LLM (어댑터 없음 → 기본 생성)."""

    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    def generate(self, prompt, **kwargs):
        self.calls += 1
        return self.text


def test_generate_reply_keeps_model_answer_on_calm_turn():
    r = generate_reply(_FixedLlm(UNRELATED_OK), "잘 주무셨어요?", TRANSCRIPT, None, False, 1, text_mood=CALM)
    assert (r.source, r.text) == ("LLM", UNRELATED_OK)


def test_generate_reply_without_mood_falls_back_as_before():
    r = generate_reply(_FixedLlm(UNRELATED_OK), "잘 주무셨어요?", TRANSCRIPT, None, False, 1)
    assert r.source == "FALLBACK" and r.failures == ["대답과 무관"]


def test_generate_reply_replaces_advice_on_calm_turn():
    r = generate_reply(_FixedLlm(ADVICE[0]), "잘 주무셨어요?", TRANSCRIPT, None, False, 1, text_mood=CALM)
    assert r.source == "FALLBACK" and "권유" in r.failures


# --- pipeline: 그 턴의 글 기분이 실제로 대답 규칙에 들어가는지 ---

def _stub_others(monkeypatch):
    monkeypatch.setattr(pipeline, "extract_signals", lambda *a, **k: {"riskFlag": False})
    monkeypatch.setattr(pipeline, "summarize_answer", lambda *a, **k: "요약")
    monkeypatch.setattr(pipeline, "_risk_prob_capped", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "risk_probability", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "keyword_risk", lambda t: False)


def _run(code: str, llm, transcript: str = "어젯밤엔 아니 못잤어"):
    return pipeline._analyzed_turn(
        transcript, {"label": None, "usable": False},
        lambda t: TextEmotionResult(code, 0.95, True, {}),
        llm=llm, question="잘 주무셨어요?", prev_signals=None, is_last_turn=False, turn_no=1,
    )


def test_pipeline_calm_turn_keeps_model_answer(monkeypatch):
    _stub_others(monkeypatch)
    out = _run("CALM", _FixedLlm(UNRELATED_OK))
    assert out["textMood"]["turnLabel"] == "평온"
    assert (out["replySource"], out["reply"]["text"]) == ("LLM", UNRELATED_OK)


def test_pipeline_joy_turn_keeps_model_answer(monkeypatch):
    _stub_others(monkeypatch)
    out = _run("JOY", _FixedLlm(UNRELATED_OK))
    assert out["textMood"]["turnLabel"] == "기쁨"
    assert (out["replySource"], out["reply"]["text"]) == ("LLM", UNRELATED_OK)


def test_pipeline_angry_turn_falls_back_to_empathy(monkeypatch):
    # 원래 6감정 '분노'는 턴 라벨 '화남' (부정) - 전 규칙으로 탈락하고 공감 문장
    _stub_others(monkeypatch)
    out = _run("분노", _FixedLlm(UNRELATED_OK))
    assert out["textMood"]["turnLabel"] == "화남"
    assert out["replySource"] == "FALLBACK" and out["reply"]["text"] in FALLBACK_EMPATHY


def test_fallback_log_has_rule_names_only(caplog):
    # 11/12 점검 L-2: 대체 이유는 규칙 이름만 남기고, 대답·어르신 말 본문은 남기지 않는다
    with caplog.at_level("INFO", logger="app.services.companion.reply_service"):
        generate_reply(_FixedLlm(ADVICE[0]), "잘 주무셨어요?", TRANSCRIPT, None, False, 1, text_mood=CALM)
    lines = [r.getMessage() for r in caplog.records if "[COMPANION-REPLY-FALLBACK]" in r.getMessage()]
    assert lines == ["[COMPANION-REPLY-FALLBACK] failures=권유 relaxed=True last=False"]
    assert TRANSCRIPT not in lines[0] and ADVICE[0] not in lines[0]


def test_model_answer_is_not_logged_as_fallback(caplog):
    with caplog.at_level("INFO", logger="app.services.companion.reply_service"):
        generate_reply(_FixedLlm(UNRELATED_OK), "잘 주무셨어요?", TRANSCRIPT, None, False, 1, text_mood=CALM)
    assert not any("[COMPANION-REPLY-FALLBACK]" in r.getMessage() for r in caplog.records)


def test_pipeline_depressed_turn_still_falls_back_to_empathy(monkeypatch):
    _stub_others(monkeypatch)
    out = _run("DEPRESSED", _FixedLlm(UNRELATED_OK))
    assert out["replySource"] == "FALLBACK"
    assert out["reply"]["text"] in FALLBACK_EMPATHY


def test_pipeline_risk_first_off_uses_previous_rules(monkeypatch, settings):
    # risk_first 가 꺼지면 대답 생성 때 기분이 아직 없다 → 전 규칙 ('대답과 무관'으로 대체)
    settings(companion_risk_first=False)
    _stub_others(monkeypatch)
    out = _run("CALM", _FixedLlm(UNRELATED_OK))
    assert out["replySource"] == "FALLBACK"
    assert out["reply"]["text"] in FALLBACK_FOLLOWUP
