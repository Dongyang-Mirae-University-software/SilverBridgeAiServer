"""예약 서비스가 PAST_TIME(오늘 이미 지난 시간, 예약 서비스 #4)으로 거절하면 막다른 오류가 아니라 시간을 다시 고르게 한다."""
from __future__ import annotations

from tests.reservation_flow_helpers import FakeReservation, make_orchestrator
from tests.test_reservation_confirm_cancel import FULL, _confirm_card, _turn, _value

PAST = {"httpStatus": 400, "data": {"success": False, "code": "PAST_TIME", "message": "이미 지난 시간이에요. 다른 시간을 골라 주세요."}}


def test_past_time_rejection_leads_to_time_reselect() -> None:
    fake = FakeReservation()
    fake.create_result = PAST
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    out = _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")))
    assert fake.names() == ["create"]
    assert out["ui"].field == "time", out["reply"]
    assert out["reply"].startswith("그 시간은 이미 지났어요. 다른 시간을 골라 주세요.")
    assert out.get("type") != "tool_error"


def test_past_time_without_other_slots_shows_service_message() -> None:
    fake = FakeReservation()
    fake.create_result = PAST
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    fake.slots = []  # 다시 고를 시간이 없으면 예약 서비스 문구로 끝낸다
    out = _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")))
    assert out["type"] == "tool_error" and out["error"]["code"] == "PAST_TIME"
    assert "이미 지난 시간" in out["reply"]


def test_duplicate_reselect_keeps_its_own_wording() -> None:
    fake = FakeReservation()
    fake.create_result = {"httpStatus": 400, "data": {"code": "DUPLICATE_RESERVATION", "message": "동일 시간대에 이미 예약이 있습니다."}}
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    out = _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")))
    assert out["ui"].field == "time"
    assert out["reply"].startswith("같은 시간대에 이미 예약이 있어요. 다른 시간을 골라 주세요.")
