"""기분 코드값 (백엔드 결정 D): mood 는 JOY·CALM·WORRY·DEPRESSED·UNKNOWN 중 하나. 한글은 moodKo 로 따로 준다."""

import pytest

from app.services.companion.dialogue_store import DialogueStore
from app.services.companion.turn_record import build_dialogue_result


@pytest.mark.parametrize("label, code, ko", [("기쁨", "JOY", "기쁨"), ("평온", "CALM", "평온"),
                                             ("우울", "DEPRESSED", "우울"), ("슬픔", "WORRY", "걱정"),
                                             ("화남", "WORRY", "걱정")])
def test_mood_maps_to_code(label, code, ko):
    store = DialogueStore()
    d = store.create("w", "s", "09:00", "질문", 8, "q")
    store.add_turn(d.dialogueId, label, "TEXT", 3)
    store.finish(d.dialogueId, "USER_STOP")
    result = build_dialogue_result(d)
    assert result["mood"] == code
    assert result["moodCode"] == code
    assert result["moodKo"] == ko


def test_undecided_is_unknown_not_null():
    store = DialogueStore()
    d = store.create("w", "s", "09:00", "질문", 8, "q")
    store.add_turn(d.dialogueId, "판단 보류", "TEXT", 3)
    store.finish(d.dialogueId, "USER_STOP")
    result = build_dialogue_result(d)
    assert result["mood"] == "UNKNOWN"
    assert result["moodCode"] == "UNKNOWN"
    assert result["moodKo"] is None
