"""위험 판정: '있' 확률이 기준(0.5) 이상이면 위험. 계산이 안 되면 신호 추출의 riskFlag 를 쓴다."""

import json

import pytest

from app.core.config import get_settings
from app.services.companion import pipeline
from app.services.companion.llm_client import LlmUnavailable
from app.services.companion.risk import RISK_THRESHOLD, is_risk, risk_probability
from app.services.companion.text_emotion_service import TextEmotionResult


@pytest.fixture(autouse=True)
def _jaehee_risk_v4(monkeypatch):
    """이 파일은 재희 v4 위험 판정(기준 0.5)을 본다. 11/10 기본 v9 는 tests/test_companion_judgment_stage.py."""
    s = get_settings()
    monkeypatch.setattr(s, "companion_risk_prompt", "v4")
    monkeypatch.setattr(s, "companion_risk_threshold", None)


class ProbLlm:
    """답은 만들지 않고 확률만 준다. 신호 추출·응답 생성은 연결 실패로 둔다."""

    def __init__(self, p_yes: float):
        self.p_yes = p_yes

    def generate(self, *_args, **_kwargs):
        raise LlmUnavailable("off")

    def next_token_probs(self, prompt, prefix, candidates, timeout):
        assert prefix == '{"risk": "'
        assert candidates == ("있", "없")
        assert "{{TRANSCRIPT}}" not in prompt
        return {"있": self.p_yes, "없": 1 - self.p_yes}


class SignalOnlyLlm:
    """확률 기능이 없는 모델: 신호 추출 답에 riskFlag 가 들어 있다."""

    def __init__(self, risk: bool):
        self.risk = risk

    def generate(self, *_args, **_kwargs):
        signals = {"emotion": "negative", "activityMotivation": "low", "socialContact": "unknown",
                   "physicalCondition": "unknown", "futureOrientation": "unknown",
                   "responseEvasion": False, "riskFlag": self.risk}
        return json.dumps(signals, ensure_ascii=False)


def _run(llm, transcript="이렇게 오래 살아서 뭐 하냐"):
    classify = lambda _t: TextEmotionResult("우울", 0.9, True, {})  # noqa: E731
    return pipeline.run_turn(transcript=transcript, face={}, classify_text=classify, llm=llm,
                             question="오늘 하루는 어떠세요?", prev_signals=None, is_last_turn=False, turn_no=1)


def test_threshold_is_point_five_after_recalibration():
    assert RISK_THRESHOLD == 0.5
    assert is_risk(0.5) is True
    assert is_risk(0.95) is True
    assert is_risk(0.49) is False
    assert is_risk(None) is False


def test_high_probability_marks_risk():
    result = _run(ProbLlm(0.995))
    assert result["riskFlag"] is True


def test_low_probability_does_not_mark_risk():
    result = _run(ProbLlm(0.3))
    assert result["riskFlag"] is False


def test_without_probability_fallback_to_signal_flag():
    assert _run(SignalOnlyLlm(risk=False))["riskFlag"] is False
    assert _run(SignalOnlyLlm(risk=True))["riskFlag"] is True


def test_risk_probability_returns_none_when_unavailable():
    assert risk_probability(SignalOnlyLlm(risk=False), "말") is None
    assert risk_probability(ProbLlm(0.9), "   ") is None
    assert risk_probability(ProbLlm(0.9), "말") == 0.9
