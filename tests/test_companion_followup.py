"""꼬리질문 관련성 검사, 턴 요약, 대화 결과의 요약 합치기."""

from app.services.companion.answer_summary import MAX_CHARS, summarize_answer
from app.services.companion.dialogue_store import DialogueStore
from app.services.companion.llm_client import LlmUnavailable
from app.services.companion.reply_service import check_reply
from app.services.companion.turn_record import build_dialogue_result


class FakeLlm:
    def __init__(self, text=None, error=False):
        self.text = text
        self.error = error

    def generate(self, prompt, *, temperature, num_predict, timeout):
        if self.error:
            raise LlmUnavailable("down")
        return self.text


def test_followup_that_reuses_a_word_from_the_answer_passes():
    assert check_reply("토스트를 드셨군요. 같이 곁들인 음료는 무엇이었어요?", False, "아침은 토스트를 먹었어요") == []


def test_followup_about_an_unrelated_topic_is_rejected():
    failures = check_reply("날씨가 참 좋네요. 바깥에 나가 보셨어요?", False, "아침은 토스트를 먹었어요")
    assert "대답과 무관" in failures


def test_last_turn_greeting_is_not_checked_for_relevance():
    assert "대답과 무관" not in check_reply("편안한 하루 보내세요. 고맙습니다.", True, "아침은 토스트를 먹었어요")


def test_common_words_alone_do_not_trigger_relevance_check():
    assert "대답과 무관" not in check_reply("그러셨군요. 어떻게 지내셨어요?", False, "그냥 오늘은 좋아요")


def test_summary_is_short_and_has_no_quotes():
    out = summarize_answer(FakeLlm(text='"아침으로 토스트를 드셨다."'), "아침은 토스트를 먹었어요")
    assert out == "아침으로 토스트를 드셨다."


def test_summary_is_truncated_to_the_limit():
    out = summarize_answer(FakeLlm(text="가" * 200), "x")
    assert len(out) == MAX_CHARS


def test_summary_failure_returns_none_without_breaking_the_turn():
    assert summarize_answer(FakeLlm(error=True), "x") is None
    assert summarize_answer(FakeLlm(text="   "), "x") is None


def test_dialogue_result_joins_turn_summaries_and_keeps_no_raw_text():
    store = DialogueStore()
    d = store.create("w", "s", "09:00", "질문", 8, "q")
    store.add_turn(d.dialogueId, "기쁨", "TEXT", 3, summary="아침으로 토스트를 드셨다.")
    store.add_turn(d.dialogueId, None, "NO_RESPONSE", 3)
    store.add_turn(d.dialogueId, "평온", "TEXT", 3, summary="산책을 다녀오셨다.")
    store.finish(d.dialogueId, "USER_STOP")
    result = build_dialogue_result(d)
    assert result["turnSummaries"] == ["아침으로 토스트를 드셨다.", "산책을 다녀오셨다."]
    assert result["summary"] == "아침으로 토스트를 드셨다. · 산책을 다녀오셨다."
