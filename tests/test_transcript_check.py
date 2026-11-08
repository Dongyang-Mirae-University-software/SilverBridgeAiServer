"""음성 인식 결과 검증: 없는 단어(삼기탕)는 확인 질문으로 돌리고, 맞는 말은 그대로 통과시킨다."""

from app.services.companion import pipeline
from app.services.companion.dialogue_store import DialogueStore
from app.services.companion.llm_client import LlmUnavailable
from app.services.companion.text_emotion_service import TextEmotionResult
from app.services.companion.transcript_check import check_transcript, confirm_reply
from app.services.companion.turn_record import build_dialogue_result


class NeverCalledLlm:
    def generate(self, *_args, **_kwargs):
        raise AssertionError("확인 턴에서는 LLM을 부르면 안 된다")


def test_misrecognized_food_is_flagged_with_the_close_word():
    c = check_transcript("삼기탕을 먹었어요")
    assert "삼기탕" in c.suspicious
    assert c.candidates["삼기탕"] == "삼계탕"


def test_correct_everyday_sentences_are_not_flagged():
    for text in ["삼계탕을 먹었어요", "산책을 다녀왔어요", "한 시간쯤 걸었어요", "아침에 토스트를 먹었어요",
                 "손주가 와서 같이 점심을 먹었어요", "고구마를 쪄서 먹었어요", "오늘은 날씨가 좋아요"]:
        assert check_transcript(text).suspicious == [], text


def test_confirm_reply_asks_instead_of_guessing():
    text = confirm_reply(check_transcript("삼기탕을 먹었어요"))
    assert "삼계탕 이야기이신가요?" in text
    assert "먹었어요" not in text  # 틀린 말을 되풀이하지 않는다


def test_suspicious_turn_skips_llm_and_marks_needs_check():
    store = DialogueStore()
    d = store.create("w", "s", "12:00", "점심은 드셨어요?", 8, "q")
    classify = lambda _t: TextEmotionResult("기쁨", 0.9, True, {})  # noqa: E731
    result = pipeline.run_turn(
        transcript="삼기탕을 먹었어요", face={}, classify_text=classify, llm=NeverCalledLlm(),
        question=d.question, prev_signals=None, is_last_turn=False, turn_no=1,
    )
    assert result["replySource"] == "CONFIRM"
    assert result["needsCheck"] is True
    assert result["signals"] is None
    assert "삼계탕" in result["reply"]["text"]


def test_needs_check_count_is_in_the_result():
    store = DialogueStore()
    d = store.create("w", "s", "12:00", "점심은 드셨어요?", 8, "q")
    store.add_turn(d.dialogueId, "판단 보류", "TEXT", 3, needs_check=True)
    store.add_turn(d.dialogueId, "기쁨", "TEXT", 3)
    store.finish(d.dialogueId, "USER_STOP")
    assert build_dialogue_result(d)["needsCheckCount"] == 1
