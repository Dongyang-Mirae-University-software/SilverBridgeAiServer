"""회피 → 기분 보완(mood_adjust) 시험 (2026-11-19, 정서분석 고도화).

회피를 평온으로 읽던 것을 괴로움(WORRY)으로 바로잡는다. 보고 기분(label)만 바꾸고 대답 선택(turnLabel)은 그대로 둔다.
"""
from __future__ import annotations

from app.core.config import get_settings
from app.services.companion import mood_adjust


def _mood(label, usable=True, turn="편안"):
    return {"label": label, "turnLabel": turn, "confidence": 0.8, "usable": usable, "reason": None}


def test_avoid_calm_becomes_worry():
    # 평가 세트 142턴에서 실제로 되살린 회피 턴들 (V1-2 "말하기 싫어", V3-2 "됐어, 그만 물어봐")
    for text in ("말하기 싫어", "됐어, 그만 물어봐", "묻지 마"):
        m = _mood("CALM")
        mood_adjust.adjust_avoid_mood(text, m)
        assert m["label"] == "WORRY", text
        assert m["adjusted"] == "avoid"


def test_turn_label_kept_so_reply_flow_unchanged():
    m = _mood("CALM", turn="편안")
    mood_adjust.adjust_avoid_mood("말하기 싫어", m)
    assert m["label"] == "WORRY" and m["turnLabel"] == "편안"  # 대답 선택용 turnLabel 은 안 바꾼다


def test_non_avoid_calm_unchanged():
    # avoidance 패턴에 안 걸리는 말(물음 되받기·평범한 일상)은 그대로 둔다 - 오탐 0 유지
    for text in ("점심은 맛있게 먹었어", "손주가 와서 좋았어", "그냥 집에 있었어", "그런 건 왜 물어"):
        m = _mood("CALM")
        mood_adjust.adjust_avoid_mood(text, m)
        assert m["label"] == "CALM", text
        assert "adjusted" not in m


def test_non_calm_label_unchanged():
    # 이미 부정/기쁨이면 건드리지 않는다 (회피여도)
    for label in ("WORRY", "DEPRESSED", "JOY", "UNKNOWN"):
        m = _mood(label)
        mood_adjust.adjust_avoid_mood("그만해", m)
        assert m["label"] == label


def test_not_usable_unchanged():
    m = _mood("CALM", usable=False)
    mood_adjust.adjust_avoid_mood("그만해", m)
    assert m["label"] == "CALM"


def test_none_text_mood_no_crash():
    mood_adjust.adjust_avoid_mood("그만해", None)  # 예외 없이 넘어간다


def test_flag_default_on():
    assert get_settings().companion_avoid_mood is True
