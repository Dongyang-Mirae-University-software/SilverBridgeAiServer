"""백엔드 결정 E·F: 응급 낱말 규칙(모델 없이도 위험 알림), 한 턴 60초 상한(넘으면 미리 정한 문장)."""

import time

from app.core.config import get_settings
from app.services.companion import pipeline
from app.services.companion.llm_client import LlmUnavailable
from app.services.companion.risk import keyword_risk
from app.services.companion.text_emotion_service import TextEmotionResult


class NoModelLlm:
    """모델 연결 실패: 답도, 확률도 없다."""

    def generate(self, *_args, **_kwargs):
        raise LlmUnavailable("off")


class SlowLlm:
    def __init__(self, sleep_sec: float):
        self.sleep_sec = sleep_sec

    def generate(self, *_args, **_kwargs):
        time.sleep(self.sleep_sec)
        return "느린 대답입니다?"


def _classify(_t):
    return TextEmotionResult("기쁨", 0.9, True, {})


def _run(llm, transcript):
    return pipeline.run_turn(transcript=transcript, face={}, classify_text=_classify, llm=llm,
                             question="오늘 하루는 어떠세요?", prev_signals=None, is_last_turn=False, turn_no=1)


def test_keyword_rule_catches_emergency_words_with_spacing_ignored():
    assert keyword_risk("숨이 안 쉬어져요") is True
    assert keyword_risk("숨이안쉬어져요") is True
    assert keyword_risk("이렇게 죽고 싶어요") is True
    assert keyword_risk("오늘 날씨가 좋아요") is False


def test_keyword_rule_flags_risk_even_when_model_is_off():
    result = _run(NoModelLlm(), "숨이 안 쉬어져서 쓰러졌어요")
    assert result["riskFlag"] is True


def test_no_keyword_and_no_model_means_no_risk():
    result = _run(NoModelLlm(), "오늘 아침은 토스트를 먹었어요")
    assert result["riskFlag"] is False


def test_turn_cap_returns_fallback_reply_without_waiting(monkeypatch):
    settings = get_settings().model_copy(update={"companion_turn_cap_sec": 0.3})
    monkeypatch.setattr(pipeline, "get_settings", lambda: settings)
    started = time.monotonic()
    result = _run(SlowLlm(sleep_sec=2.0), "오늘 아침은 토스트를 먹었어요")
    elapsed = time.monotonic() - started
    assert elapsed < 1.5, elapsed  # 느린 작업을 기다리지 않는다
    assert result["replySource"] == "FALLBACK"
    assert result["reply"]["text"]  # 미리 정한 문장이 나간다
    assert result["signals"] is None  # 시간 안에 못 끝난 신호는 쓰지 않는다
