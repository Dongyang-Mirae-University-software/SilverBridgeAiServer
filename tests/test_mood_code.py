"""기분 코드값 (백엔드 결정 1번): 기쁨→JOY, 평온→CALM, 걱정→WORRY, 우울→DEPRESSED, 판단 못 함→UNKNOWN."""

import pytest

from app.services.companion.dialogue_store import DialogueStore
from app.services.companion.turn_record import build_dialogue_result


@pytest.mark.parametrize("label, code", [("기쁨", "JOY"), ("평온", "CALM"), ("우울", "DEPRESSED"), ("슬픔", "WORRY"), ("화남", "WORRY")])
def test_mood_maps_to_code(label, code):
    store = DialogueStore()
    d = store.create("w", "s", "09:00", "질문", 8, "q")
    store.add_turn(d.dialogueId, label, "TEXT", 3)
    store.finish(d.dialogueId, "USER_STOP")
    result = build_dialogue_result(d)
    assert result["moodCode"] == code
    assert result["mood"] in ("기쁨", "평온", "걱정", "우울")


def test_undecided_is_unknown_not_zero():
    store = DialogueStore()
    d = store.create("w", "s", "09:00", "질문", 8, "q")
    store.add_turn(d.dialogueId, "판단 보류", "TEXT", 3)
    store.finish(d.dialogueId, "USER_STOP")
    result = build_dialogue_result(d)
    assert result["mood"] is None
    assert result["moodCode"] == "UNKNOWN"
