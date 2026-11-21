"""턴 응답 turnResult.moodCode (2026-11-21 최종 점검 H-1).

라벨 '슬픔'·'화남'은 백엔드 라벨 표(기쁨·평온·걱정·우울)에 없어 턴 기분이 null 로 저장되던 문제.
대화 단위와 같은 표(SCREEN_MOOD -> MOOD_CODE)로 코드를 함께 보낸다.
"""
from __future__ import annotations

import pytest

from app.routers import companion_router
from app.services.companion.dialogue_store import store
from app.services.companion.turn_record import turn_mood_code


@pytest.mark.parametrize("label,code", [
    ("기쁨", "JOY"), ("평온", "CALM"), ("슬픔", "WORRY"), ("화남", "WORRY"), ("우울", "DEPRESSED"),
    ("판단 보류", "UNKNOWN"), (None, "UNKNOWN"), ("모르는 값", "UNKNOWN"),
])
def test_turn_mood_code(label, code):
    assert turn_mood_code(label) == code


@pytest.mark.parametrize("label,code", [("슬픔", "WORRY"), ("화남", "WORRY"), ("기쁨", "JOY"), ("판단 보류", "UNKNOWN")])
def test_turn_response_carries_mood_code(label, code):
    d = store.create("ward-mood", "s-mood", "09:00", "좋은 아침이에요.", 8, "q-mood")
    original = {"label": label, "source": "TEXT"}
    result = {"turnResult": original, "reply": "네."}
    out = companion_router._turn_response(store.get(d.dialogueId), result, face={"label": None, "usable": False})
    assert out["turnResult"] == {"label": label, "source": "TEXT", "moodCode": code}
    assert "moodCode" not in original  # 원래 dict 는 그대로


def test_turn_response_no_response_is_unknown():
    d = store.create("ward-mood2", "s-mood2", "09:00", "좋은 아침이에요.", 8, "q-mood2")
    result = {"turnResult": {"label": None, "source": "NO_RESPONSE"}, "reply": "네."}
    out = companion_router._turn_response(store.get(d.dialogueId), result)
    assert out["turnResult"]["moodCode"] == "UNKNOWN"
