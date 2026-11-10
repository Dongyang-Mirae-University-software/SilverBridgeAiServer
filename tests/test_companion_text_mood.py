"""턴 응답의 글 기분 분류 결과 `textMood` (백엔드 테스트 계정 화면용, 2026-11-10).

analyze_turn_result 의 반환 모양은 그대로이고, text_out 으로만 따로 받는다. 저장하지 않는다.
"""
from __future__ import annotations

from app.services.companion import pipeline
from app.services.companion.text_emotion_service import TextEmotionResult, TextEmotionUnavailable

NO_FACE = {"label": None, "usable": False}


def _clf(label: str, conf: float = 0.9):
    return lambda text: TextEmotionResult(label, conf, conf >= 0.6, {})


def test_text_out_filled_and_return_shape_unchanged():
    out: dict = {}
    r = pipeline.analyze_turn_result("오늘은 손주가 와서 정말 기분이 좋아요", NO_FACE, _clf("기쁨"), text_out=out)
    assert r == {"label": "기쁨", "source": "TEXT"}  # 반환값 모양은 그대로
    assert out == {"label": "기쁨", "turnLabel": "기쁨", "confidence": 0.9, "usable": True, "reason": None}


def test_text_out_low_confidence_is_not_usable():
    out: dict = {}
    pipeline.analyze_turn_result("오늘은 손주가 와서 정말 기분이 좋아요", NO_FACE, _clf("기쁨", 0.3), text_out=out)
    assert out["usable"] is False
    assert out["label"] == "기쁨"
    assert out["reason"] is None


def test_text_out_short_text(monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "companion_text_mood_engine", "ggara02")  # 10자 미만은 분류하지 않는다
    out: dict = {}
    pipeline.analyze_turn_result("네 좋아요", NO_FACE, _clf("기쁨"), text_out=out)
    assert out == {"label": None, "turnLabel": None, "confidence": None, "usable": False, "reason": "SHORT"}


def test_text_out_model_unavailable():
    def broken(_text):
        raise TextEmotionUnavailable("없음")

    out: dict = {}
    pipeline.analyze_turn_result("오늘은 손주가 와서 정말 기분이 좋아요", NO_FACE, broken, text_out=out)
    assert out == {"label": None, "turnLabel": None, "confidence": None, "usable": False, "reason": "UNAVAILABLE"}


def test_no_response_turn_has_no_text_mood():
    r = pipeline.run_no_response_turn(1, False)
    assert r.get("textMood") is None
