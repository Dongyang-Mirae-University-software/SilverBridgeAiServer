from __future__ import annotations

import time

import pytest

from app.core.config import get_settings
from app.services.companion import pipeline, reply_service
from app.services.companion.llm_client import LlmUnavailable
from app.services.companion.transcript_check import check_transcript
from app.services.companion.text_emotion_service import TextEmotionResult
from app.services.companion.reply_service import (
    FALLBACK_FOLLOWUP,
    FALLBACK_LAST,
    check_reply,
    generate_reply,
)

# 어르신 대답 "입맛이 별로 없어요"의 낱말을 꼬리질문이 이어 받는다 (대답과 무관 규칙, 2026-11-08 추가)
ANSWER = "입맛이 별로 없어요"


def _retry_once(monkeypatch):
    # 기본 재시도 횟수는 0(60초 턴 상한 때문). 이 시험들은 "재시도 1회"를 직접 켠다.
    settings = get_settings().model_copy(update={"companion_llm_retry": 1})
    monkeypatch.setattr(reply_service, "get_settings", lambda: settings)


class ScriptedLlm:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def generate(self, prompt, **kwargs):
        self.calls += 1
        item = self.answers.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


GOOD = "그러셨군요. 요즘 입맛은 좀 어떠세요?"


@pytest.mark.parametrize("text,is_last,expected", [
    (GOOD, False, []),
    ("그러셨군요. 요즘 입맛은 좀 어떠세요", False, ["물음으로 끝나지 않음"]),
    ("지난번에 병원 다녀오셨군요. 괜찮으세요?", False, ["과거 가정"]),
    ("우울증이 있으시면 병원에 가 보세요.", True, ["질병명"]),
    ("고맙습니다. " * 25, True, ["길이 초과"]),
    ("", False, ["비어 있음"]),
    ("오늘도 고맙습니다. 편안한 하루 보내세요.", True, []),
    ("오늘도 고맙습니다. 또 봬요?", True, []),  # 마지막 턴은 물음이 아니어도 된다
])
def test_rule_check_cases(text, is_last, expected):
    assert check_reply(text, is_last) == expected


def test_fallback_sentences_all_pass_their_own_rules():
    for s in FALLBACK_FOLLOWUP:
        assert check_reply(s, is_last_turn=False) == [], s
    for s in FALLBACK_LAST:
        assert check_reply(s, is_last_turn=True) == [], s


def test_good_llm_reply_is_used():
    llm = ScriptedLlm([GOOD])
    r = generate_reply(llm, "잘 주무셨어요?", ANSWER, None, is_last_turn=False, turn_no=1)
    assert (r.text, r.source, r.isFollowUp) == (GOOD, "LLM", True)
    assert llm.calls == 1


def test_rule_failure_regenerates_once_then_succeeds(monkeypatch):
    _retry_once(monkeypatch)
    llm = ScriptedLlm(["지난번 여행 얘기네요?", GOOD])
    r = generate_reply(llm, "잘 주무셨어요?", ANSWER, None, is_last_turn=False, turn_no=1)
    assert r.source == "LLM" and r.text == GOOD
    assert llm.calls == 2


def test_two_rule_failures_use_a_fallback_sentence(monkeypatch):
    _retry_once(monkeypatch)
    llm = ScriptedLlm(["지난번 여행 얘기네요?", "우울증 같으세요?"])
    r = generate_reply(llm, "잘 주무셨어요?", ANSWER, None, is_last_turn=False, turn_no=0)
    assert r.source == "FALLBACK"
    assert r.text in FALLBACK_FOLLOWUP
    assert llm.calls == 2


def test_llm_outage_retries_then_falls_back_without_error():
    llm = ScriptedLlm([LlmUnavailable("끔"), LlmUnavailable("끔")])
    r = generate_reply(llm, "잘 주무셨어요?", "네", None, is_last_turn=True, turn_no=7)
    assert r.source == "FALLBACK"
    assert r.text in FALLBACK_LAST
    assert r.isFollowUp is False


def test_last_turn_never_asks_a_question():
    r = generate_reply(ScriptedLlm([LlmUnavailable("끔")] * 2), "q", "a", None, is_last_turn=True, turn_no=8)
    assert r.isFollowUp is False and "?" not in r.text


def test_run_turn_runs_signals_and_reply_in_parallel():
    class SlowLlm:
        def generate(self, prompt, **kwargs):
            time.sleep(0.3)
            if "정서 신호" in prompt or "JSON" in prompt or "분석기" in prompt:
                return '{"emotion": "neutral", "activityMotivation": "mid", "socialContact": "unknown", ' \
                       '"physicalCondition": "unknown", "futureOrientation": "unknown", ' \
                       '"responseEvasion": false, "riskFlag": false}'
            return GOOD

    # 형태소 분석기는 프로세스에서 처음 쓸 때 약 2초 로드된다. 그 비용은 측정에서 뺀다 (병렬성만 본다).
    check_transcript(ANSWER)
    started = time.monotonic()
    out = pipeline.run_turn(
        transcript=ANSWER,
        face={"label": None, "usable": False},
        classify_text=lambda t: TextEmotionResult("기쁨", 0.9, True, {}),
        llm=SlowLlm(),
        question="잘 주무셨어요?",
        prev_signals=None,
        is_last_turn=False,
        turn_no=1,
    )
    elapsed = time.monotonic() - started
    assert out["reply"]["text"] == GOOD
    assert out["signals"] is not None
    assert elapsed < 0.55, elapsed  # 두 호출을 합치면 0.6초 이상이 된다
