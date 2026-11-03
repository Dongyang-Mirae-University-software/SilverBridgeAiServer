"""되묻기 다음 턴 종료 판단(2026-11-04): 종료 낱말이 있어도 문장 전체가 인사·종료일 때만 앞 질문과 이어 붙이지 않는다.
"며칠 됐어요"(기간)·"충혈됐어요"·"열은 없고 괜찮아요"(정도)는 되묻기에 대한 답이라 이어 붙인다."""
from __future__ import annotations

import pytest

from app.services.chat_service import _is_closing_message
from tests.test_chat_followup_question import INTRO, _send, env  # noqa: F401 - 같은 대화 흐름 하네스

ANSWERS = ["며칠 됐어요", "사흘 됐어요, 가래도 나와요", "일주일쯤 됐어요", "오래됐어요", "3일 됐어요", "아침부터 충혈됐어요",
           "열은 없고 괜찮아요", "참을 만해서 괜찮아요", "그만큼 심하진 않아요", "그만그만해요", "고마워요, 근데 열도 나요"]
CLOSINGS = ["고마워요", "감사합니다", "감사 합니다", "고맙습니다", "이제 됐어요", "됐어요", "괜찮아요", "네 괜찮아요",
            "그만할게요", "그만 할게요", "그만", "안녕히 계세요", "알겠어요 고마워요!", "네네 고마워요 ㅎㅎ", "정말 감사합니다~"]


@pytest.mark.parametrize("message", ANSWERS)
def test_answers_with_close_word_are_not_closing(message: str) -> None:
    assert not _is_closing_message(message)


@pytest.mark.parametrize("message", CLOSINGS)
def test_greeting_only_is_closing(message: str) -> None:
    assert _is_closing_message(message)


@pytest.mark.parametrize("message", ["", "며칠 전부터요", "열도 나요"])
def test_no_close_word_is_not_closing(message: str) -> None:
    assert not _is_closing_message(message)


@pytest.mark.parametrize("answer", ["며칠 됐어요", "아침부터 충혈됐어요", "열은 없고 괜찮아요", "고마워요, 근데 열도 나요"])
def test_answer_with_close_word_continues_previous_question(env, answer: str) -> None:  # noqa: F811
    _send(env, "머리가 아파요")
    env.router.route_value = "general"  # 짧은 답을 라우터가 일반 대화로 보내도
    out = _send(env, answer)
    call = env.llm.calls[-1]
    assert call["message"] == f"머리가 아파요 / {answer}"
    assert call["intent"] == "medical_advice"
    assert INTRO not in out["reply"]


@pytest.mark.parametrize("closing", ["고마워요", "이제 됐어요", "그만할게요", "안녕히 계세요"])
def test_closing_after_followup_is_not_joined(env, closing: str) -> None:  # noqa: F811
    _send(env, "머리가 아파요")
    env.router.route_value = "general"
    _send(env, closing)
    call = env.llm.calls[-1]
    assert call["message"] == closing and call["intent"] == "general_chat"
