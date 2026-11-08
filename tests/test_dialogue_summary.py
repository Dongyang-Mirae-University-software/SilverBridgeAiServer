"""대화 결과의 최종 분석: 감정·표정별 횟수와 가장 많이 나온 것, 같은 문장 반복 막기."""

from app.routers.companion_router import _avoid_repeat
from app.services.companion import reply_service
from app.services.companion.dialogue_store import DialogueStore
from app.services.companion.turn_record import build_dialogue_result


def _dialogue(store: DialogueStore, max_turns: int = 8):
    return store.create("w", "s", "09:00", "오늘 아침은 어떻게 보내고 계세요?", max_turns, "q-1")


def test_dominant_emotion_and_face_are_the_most_frequent_ones():
    store = DialogueStore()
    d = _dialogue(store)
    store.add_turn(d.dialogueId, "기쁨", "TEXT", 3, face="Happiness")
    store.add_turn(d.dialogueId, "기쁨", "FACE", 3, face="Happiness")
    store.add_turn(d.dialogueId, "평온", "TEXT", 3, face="Neutral")
    store.finish(d.dialogueId, "USER_STOP")
    result = build_dialogue_result(d)
    assert result["dominantEmotion"] == "기쁨"
    assert result["emotionCounts"] == {"기쁨": 2, "평온": 1}
    assert result["dominantFace"] == "Happiness"
    assert result["faceCounts"] == {"Happiness": 2, "Neutral": 1}


def test_undecided_is_counted_but_never_dominant():
    store = DialogueStore()
    d = _dialogue(store)
    store.add_turn(d.dialogueId, "판단 보류", "TEXT", 3)
    store.add_turn(d.dialogueId, "판단 보류", "TEXT", 3)
    store.add_turn(d.dialogueId, "우울", "TEXT", 3)
    store.finish(d.dialogueId, "USER_STOP")
    result = build_dialogue_result(d)
    assert result["emotionCounts"]["판단 보류"] == 2
    assert result["dominantEmotion"] == "우울"


def test_no_face_and_no_response_give_none_not_zero():
    store = DialogueStore()
    d = _dialogue(store)
    store.add_turn(d.dialogueId, None, "NO_RESPONSE", 3)
    store.finish(d.dialogueId, "USER_STOP")
    result = build_dialogue_result(d)
    assert result["dominantEmotion"] is None
    assert result["dominantFace"] is None
    assert result["faceCounts"] == {}


def test_repeated_reply_is_replaced_with_an_unused_sentence():
    store = DialogueStore()
    d = _dialogue(store)
    first = reply_service.FALLBACK_FOLLOWUP[0]
    d.replies.append(first)  # 이미 한 말
    result = {"reply": {"text": first, "isFollowUp": True}}
    out = _avoid_repeat(d, result, is_last_turn=False)
    assert out["reply"]["text"] != first
    assert out["reply"]["text"] in reply_service.FALLBACK_FOLLOWUP
    assert d.replies[-1] == out["reply"]["text"]


def test_question_itself_is_not_repeated_as_a_reply():
    store = DialogueStore()
    d = _dialogue(store)
    result = {"reply": {"text": "오늘 아침은 어떻게 보내고 계세요?", "isFollowUp": True}}
    out = _avoid_repeat(d, result, is_last_turn=False)
    assert out["reply"]["text"] != "오늘 아침은 어떻게 보내고 계세요?"


def test_new_reply_is_kept_as_is():
    store = DialogueStore()
    d = _dialogue(store)
    result = {"reply": {"text": "오늘 점심은 무엇을 드셨어요?", "isFollowUp": True}}
    out = _avoid_repeat(d, result, is_last_turn=False)
    assert out["reply"]["text"] == "오늘 점심은 무엇을 드셨어요?"
