"""대답 한 턴의 판단 규칙 (fusion.decide_turn, 사용자 규칙 2026-10-25) 시험."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app" / "companion"))
from fusion import decide_turn  # noqa: E402


@pytest.mark.parametrize("text_pol, face_pol, label", [
    # 말이 좋으면 표정을 보지 않고 넘어간다
    ("pos", None, "기쁨"),
    ("pos", "neg", "기쁨"),
    ("neu", "neg", "평온"),
    # 말이 나쁘면 표정도 본다
    ("neg", "pos", "판단 보류"),
    ("neg", "neg", "확인 필요"),
    ("neg", "neu", "확인 필요"),
    ("neg", None, "확인 필요"),
    # 말이 애매하면
    (None, "neg", "확인 필요"),
    (None, "pos", "판단 보류"),
    (None, "neu", "판단 보류"),
    (None, None, "확인 필요"),
])
def test_turn_rules(text_pol, face_pol, label):
    assert decide_turn(text_pol, face_pol)["label"] == label


def test_every_result_has_a_reason():
    for tp in ("pos", "neu", "neg", None):
        for fp in ("pos", "neu", "neg", None):
            assert decide_turn(tp, fp)["reason"]
