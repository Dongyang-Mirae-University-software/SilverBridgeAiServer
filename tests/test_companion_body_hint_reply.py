"""첫 위험 턴의 몸 응급 안내 넓히기 (2026-11-15, I-21 사용자 결정 '고치기').

몸 응급 낱말(kw4 BODY_EMERGENCY_KEYWORDS)이 좁아 "가슴이 꽉 조이고 숨이 차"가 모델로 위험이 돼도 마음 쪽 안내
("마음이 많이 무거우신 것 같아요")가 나갔다. 위험 낱말이 없고 몸 증상 표시만 있으면 몸 응급 안내로 바꾼다.
판정(위험 여부)은 바꾸지 않는다 - 안내 문장 고르기만.
"""
from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.services.companion.reply_service import BODY_SAFETY_REPLY, SAFETY_REPLY, safety_reply
from app.services.companion.risk import keyword_risk


@pytest.fixture
def settings(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "companion_body_safety_reply", True)
    monkeypatch.setattr(s, "companion_risk_keyword_rule", "kw4")
    return lambda **kw: [monkeypatch.setattr(s, k, v) for k, v in kw.items()]


@pytest.mark.parametrize("text", [
    "가슴이 꽉 조이고 숨이 차",  # 평가 세트 K2
    "아까부터 말이 잘 안 나오고 오른팔에 힘이 없어",  # 평가 세트 K3
    "갑자기 어지럽고 식은땀이 나",
    "다리에 힘이 하나도 없어",
])
def test_model_only_body_symptoms_get_body_reply(settings, text):
    assert keyword_risk(text) is False  # 낱말로는 안 걸리고 모델로만 위험이 되는 말
    assert safety_reply(text) == BODY_SAFETY_REPLY


@pytest.mark.parametrize("text", [
    "요즘 다 내려놓고 싶어",  # 기존: 무슨 위험인지 모르면 마음 쪽
    "요즘은 그냥 자다가 안 깼으면 좋겠어",  # 평가 세트 K1
    "아무도. 나 같은 건 없어져도 아무도 모를 거야",  # 평가 세트 K4
    "숨이 차서 죽을 것 같아",  # 몸 표시 + 마음 쪽 표시('죽') → 마음 쪽 (119 포함)
])
def test_mind_or_unclear_keeps_mind_reply(settings, text):
    assert safety_reply(text) == SAFETY_REPLY


def test_risk_words_present_keeps_existing_rule(settings):
    # 낱말 규칙에 걸리는 말은 기존 규칙(body_emergency_only) 그대로 - 몸 낱말만이면 몸, 다른 위험 낱말이 섞이면 마음
    assert safety_reply("숨이 안 쉬어져") == BODY_SAFETY_REPLY
    mixed = "숨이 안 쉬어지고 죽고 싶어"
    assert keyword_risk(mixed) and safety_reply(mixed) == SAFETY_REPLY
    # 몸 증상 표시(가슴)가 있어도 마음 쪽 위험 낱말('사라지고 싶')이 있으면 마음 쪽 - 마음 쪽 표시에 안 걸리는 낱말로 확인
    other = "가슴이 답답하고 그냥 사라지고 싶어"
    assert keyword_risk(other) and safety_reply(other) == SAFETY_REPLY


def test_switch_off_and_jaehee_rule_unchanged(settings):
    settings(companion_body_safety_reply=False)
    assert safety_reply("가슴이 꽉 조이고 숨이 차") == SAFETY_REPLY
    settings(companion_body_safety_reply=True, companion_risk_keyword_rule="jaehee")
    assert safety_reply("숨이 안 쉬어져") == SAFETY_REPLY  # 재희 규칙에서는 기존처럼 마음 쪽
