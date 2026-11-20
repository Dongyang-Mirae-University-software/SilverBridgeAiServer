"""회피 → 기분 보완(mood_adjust) 시험 (2026-11-19 정서분석 고도화, 2026-11-20 점검 L-4).

회피를 평온으로 읽던 것을 괴로움(WORRY)으로 바로잡는다. 11/20 부터는 표정과 합치기 전에 보정해 턴 판정(turnResult,
백엔드가 저장하는 턴 기분)까지 간다. scores 는 모델 원래 확률 그대로. 스위치를 끄면 전과 같다.
"""
from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.services.companion import mood_adjust
from app.services.companion.pipeline import analyze_turn_result
from app.services.companion.text_emotion_service import TextEmotionResult

CALM_PROBS = {"JOY": 0.05, "CALM": 0.7, "WORRY": 0.2, "DEPRESSED": 0.05}
NO_FACE = {"label": None, "usable": False}


def _clf(label="CALM", probs=CALM_PROBS, usable=True):
    return lambda t: TextEmotionResult(label, probs.get(label, 1.0) if probs else 1.0, usable, dict(probs or {}))


@pytest.fixture
def on(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "companion_avoid_mood", True)
    monkeypatch.setattr(s, "companion_fusion_rule", "prob_add")
    return s


# --- 판정 함수 ---------------------------------------------------------------------------------------

def test_applies_only_to_usable_calm_avoidance():
    for text in ("말하기 싫어", "됐어, 그만 물어봐", "묻지 마"):  # 평가 세트에서 되살린 회피 턴 (V1-2·V3-2)
        assert mood_adjust.applies(text, "CALM", True), text
    for text in ("점심은 맛있게 먹었어", "손주가 와서 좋았어", "그냥 집에 있었어", "그런 건 왜 물어"):  # 오탐 0 유지
        assert not mood_adjust.applies(text, "CALM", True), text
    for label in ("WORRY", "DEPRESSED", "JOY", "UNKNOWN", None):
        assert not mood_adjust.applies("그만해", label, True)
    assert not mood_adjust.applies("그만해", "CALM", False)


def test_swap_keeps_total_and_moves_confidence():
    out = mood_adjust.swap_calm_worry(CALM_PROBS)
    assert out["WORRY"] == 0.7 and out["CALM"] == 0.2 and out["JOY"] == 0.05
    assert sum(out.values()) == pytest.approx(sum(CALM_PROBS.values()))
    assert CALM_PROBS["CALM"] == 0.7  # 원본은 그대로


# --- 턴 판정까지 (L-4) -------------------------------------------------------------------------------

def test_avoid_calm_reaches_turn_result(on):
    out: dict = {}
    r = analyze_turn_result("말하기 싫어", NO_FACE, _clf(), text_out=out)
    assert r == {"label": "슬픔", "source": "TEXT"}  # 백엔드가 저장하는 턴 기분 (WORRY)
    assert out["label"] == "WORRY" and out["turnLabel"] == "슬픔" and out["adjusted"] == "avoid"
    assert out["scores"]["CALM"] == 0.7 and out["scores"]["WORRY"] == 0.2  # 모델 원래 확률 그대로


def test_with_face_fuses_as_if_model_said_worry(on):
    face = {"label": "Happiness", "usable": True}
    adjusted = analyze_turn_result("말하기 싫어", face, _clf())
    on.companion_avoid_mood = False
    as_worry = analyze_turn_result("말하기 싫어", face, _clf("WORRY", mood_adjust.swap_calm_worry(CALM_PROBS)))
    assert adjusted == as_worry


def test_plan41_rule_uses_worry_label(on):
    on.companion_fusion_rule = "plan41"
    r = analyze_turn_result("말하기 싫어", NO_FACE, _clf())
    assert r["label"] == "슬픔"


def test_unchanged_when_not_avoid_or_not_calm_or_not_usable(on):
    out: dict = {}
    assert analyze_turn_result("점심은 맛있게 먹었어", NO_FACE, _clf(), text_out=out)["label"] == "평온"
    assert out["label"] == "CALM" and "adjusted" not in out
    joy = {"JOY": 0.8, "CALM": 0.1, "WORRY": 0.05, "DEPRESSED": 0.05}
    assert analyze_turn_result("그만해", NO_FACE, _clf("JOY", joy))["label"] == "기쁨"
    out2: dict = {}
    analyze_turn_result("말하기 싫어", NO_FACE, _clf(usable=False), text_out=out2)
    assert out2["label"] == "CALM" and "adjusted" not in out2


def test_switch_off_is_the_old_behavior(on):
    on.companion_avoid_mood = False
    out: dict = {}
    r = analyze_turn_result("말하기 싫어", NO_FACE, _clf(), text_out=out)
    assert r["label"] == "평온" and out["label"] == "CALM" and "adjusted" not in out


def test_flag_default_on():
    from pathlib import Path
    from app.core import config
    assert '_as_bool("COMPANION_AVOID_MOOD", True)' in Path(config.__file__).read_text(encoding="utf-8")
