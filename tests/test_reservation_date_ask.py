from __future__ import annotations

from app.core.reservation_prompts import build_user_prompt
from tests.reservation_flow_helpers import FakeReservation, make_orchestrator, make_request


def test_prompt_does_not_default_date_to_today() -> None:
    prompt = build_user_prompt(message="내과 예약해줘", history_text="", state_json="{}", location="")
    assert "오늘 날짜로 짐작해 채우지 마라" in prompt
    assert "reservation_date는 오늘 날짜로 둔다" not in prompt


def test_request_without_date_asks_for_date_instead_of_assuming_today() -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake)  # LLM 도 날짜를 비워 둔다
    out = orch.handle(make_request("강남메디내과 내과 예약해줘"), {})
    assert out["ui"].kind == "date" and out["ui"].field == "date"
    assert "create" not in fake.names()


def test_new_request_does_not_inherit_date_from_earlier_assistant_text() -> None:
    from app.schemas.chat_schema import ChatHistoryItem

    history = [ChatHistoryItem(role="assistant", content="오늘은 예약 가능한 시간이 없어 10월 13일(화) 예약 가능 시간을 보여드립니다.")]
    orch = make_orchestrator(FakeReservation())
    out = orch.handle(make_request("강남메디내과 내과 예약해줘", history=history), {})
    assert out["ui"].kind == "date"
