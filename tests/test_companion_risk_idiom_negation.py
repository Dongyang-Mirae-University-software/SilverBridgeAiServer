"""말벗 낱말 규칙 kw4 - 띄어 쓴 감정 관용구 '죽겠'과 "죽고 싶지 않아" 부정 제외 (2026-11-16).

DGX ~/llmbench/cmp/kw_check.py: 감성대화 노년 2500문장 걸림 23→17, KETI 4497문장 11→3, 위험 문장 469개 중 새로 놓친 것 0.
"""
from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.services.companion.risk import body_emergency_only, keyword_risk


@pytest.fixture(autouse=True)
def kw4(monkeypatch):
    monkeypatch.setattr(get_settings(), "companion_risk_keyword_rule", "kw4")


@pytest.mark.parametrize("text", [
    # 띄어 쓴·조사 붙은 감정 관용구
    "짜증 나 죽겠어", "짜증이 나서 죽겠어", "짜증나 죽겠어", "화가 나 죽겠어", "화나 죽겠네", "한심해 죽겠어", "성가셔 죽겠어",
    "열받아 죽겠어", "창피해 죽겠어", "부끄러워 죽겠네", "억울해 죽겠어", "분해 죽겠어", "지루해 죽겠다",
    # 죽음을 바라지 않는다는 부정
    "죽고 싶지 않아", "죽고 싶진 않아", "죽고 싶지는 않아", "죽고싶지않아", "아직은 죽고 싶지 않아요",
])
def test_idiom_and_negation_are_not_keyword_risk(text):
    assert keyword_risk(text) is False


@pytest.mark.parametrize("text", [
    "죽고 싶어", "죽고 싶지만 무서워", "살기 싫어", "자살하고 싶어", "그냥 죽어 버리고 싶다", "아이고 죽겠네",
    # 관용구·부정 뒤에 다른 위험 말이 오면 따로 걸린다
    "화가 나서 죽고 싶어", "짜증 나 죽겠어 그냥 죽고 싶다", "죽고 싶진 않은데 살기 싫어", "죽고 싶지 않아 근데 사라지고 싶어",
    # 이중 부정은 위험으로 남긴다
    "죽고 싶지 않은 건 아니야", "죽고 싶지 않은 게 아니야", "죽고 싶지 않다면 거짓말이지", "죽고 싶지 않다고 하면 거짓말이야",
])
def test_real_risk_still_caught(text):
    assert keyword_risk(text) is True


def test_negation_without_other_keyword_is_left_to_adapter():
    # 낱말 없는 하소연은 낱말 규칙이 잡지 않는다 - 위험 어댑터(OR)가 맡는다 (사용자 결정 11/16)
    assert keyword_risk("죽고 싶진 않은데 살기가 너무 힘들어") is False


def test_body_emergency_only_unchanged_by_negation():
    assert body_emergency_only("죽고 싶지 않아 근데 숨이 안 쉬어져") is True
    assert body_emergency_only("죽고 싶어 숨이 안 쉬어져") is False


def test_jaehee_rule_unchanged(monkeypatch):
    monkeypatch.setattr(get_settings(), "companion_risk_keyword_rule", "jaehee")
    assert keyword_risk("짜증 나 죽겠어") is True
    assert keyword_risk("죽고 싶지 않아") is True
