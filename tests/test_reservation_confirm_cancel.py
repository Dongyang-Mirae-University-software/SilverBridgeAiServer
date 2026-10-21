"""예약 최종 확인(R-1)·대리 표시(R-4)·지역 변경(R-5)·취소(R-3)·변경 흐름. 예약 서비스는 가짜로 대체한다."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from tests.reservation_flow_helpers import FUTURE, FUTURE2, FakeReservation, make_orchestrator, make_request

FULL = {"reservation_date": FUTURE, "reservation_time": "14:00"}


def _stored(out: dict[str, Any] | None) -> dict[str, Any]:
    """chat_service 가 다음 턴에 넘기는 값과 같다: 직전 턴 meta 의 pendingAction·flowState."""
    meta = (out or {}).get("meta") or {}
    return {k: meta[k] for k in ("pendingAction", "flowState") if isinstance(meta.get(k), dict)}


def _turn(orch, prev: dict[str, Any] | None, message: str = "", ui: tuple[str, str] | None = None, **ctx: Any) -> dict[str, Any]:
    out = orch.handle(make_request(message, ui=ui, context=ctx or None), {"storedFlow": _stored(prev)})
    assert out is not None
    return out


def _confirm_card(orch, fake: FakeReservation, **ctx: Any) -> dict[str, Any]:
    out = _turn(orch, None, "강남메디내과 내과 14:00 예약해줘", **ctx)
    assert out["ui"].field == "reservation_confirm", out["reply"]
    return out


def _value(out: dict[str, Any], prefix: str) -> str:
    return next(o.value for o in out["ui"].options if o.value.startswith(prefix))


# ---- R-1 최종 확인 -------------------------------------------------------------


def test_complete_draft_shows_confirm_card_and_does_not_book() -> None:
    fake = FakeReservation()
    out = _confirm_card(make_orchestrator(fake, FULL), fake)
    assert "create" not in fake.names()
    for text in ("홍길동님", "강남메디내과", "내과", "14:00", "010-****-5678"):
        assert text in out["reply"]
    assert "1234" not in out["reply"].replace("010-****-5678", "")  # 전화번호 가운데는 보이지 않는다
    assert [o.label for o in out["ui"].options] == ["예약하기", "고치기"]


def test_confirm_click_books_exactly_once() -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    done = _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")))
    assert fake.names() == ["create"]
    assert done["tool"] == "make_appointment" and "완료" in done["reply"]
    # 같은 버튼을 다시 눌러도(이미 직전 턴이 아님) 두 번 만들지 않는다
    again = _turn(orch, done, ui=("reservation_confirm", _value(card, "confirm:")))
    assert fake.names() == ["create"] and "시간이 지났" in again["reply"]


def test_confirm_without_matching_pending_never_books() -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    forged = _turn(orch, card, ui=("reservation_confirm", "confirm:forged-token"))
    nothing = _turn(orch, None, ui=("reservation_confirm", _value(card, "confirm:")))  # 직전 턴이 확인 카드가 아님
    assert fake.names() == []
    assert "시간이 지났" in forged["reply"] and "시간이 지났" in nothing["reply"]


def test_typing_yes_instead_of_clicking_shows_card_again_without_booking() -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    out = _turn(orch, card, "네 강남메디내과 내과 14:00 예약해줘")
    assert out["ui"].field == "reservation_confirm" and fake.names() == []


def test_edit_time_reselects_and_books_new_time_not_the_old_one() -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    edit = _turn(orch, card, ui=("reservation_confirm", _value(card, "edit:")))
    assert edit["ui"].field == "reservation_edit" and fake.names() == []
    pick = _turn(orch, edit, ui=("reservation_edit", _value(edit, "time:")))
    assert pick["ui"].field == "time"
    again = _turn(orch, pick, ui=("time", f"{FUTURE} 10:00"))
    assert again["ui"].field == "reservation_confirm" and "10:00" in again["reply"] and "14:00" not in again["reply"]
    _turn(orch, again, ui=("reservation_confirm", _value(again, "confirm:")))
    assert [c[0] for c in fake.calls] == ["create"] and fake.calls[0][1]["reservation_time"] == "10:00"


def test_edit_date_asks_date_again() -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    edit = _turn(orch, card, ui=("reservation_confirm", _value(card, "edit:")))
    asked = _turn(orch, edit, ui=("reservation_edit", _value(edit, "date:")))
    assert asked["ui"].kind == "date"
    slots = _turn(orch, asked, ui=("date", FUTURE2))
    assert slots["ui"].field == "time" and slots["ui"].options[0].value.startswith(FUTURE2)


def test_failed_booking_is_reported_not_hidden() -> None:
    fake = FakeReservation()
    fake.create_result = {"httpStatus": 400, "data": {"code": "HOSPITAL_CLOSED", "message": "휴진일입니다."}}
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    out = _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")))
    assert out["type"] == "tool_error" and out["error"]["code"] == "HOSPITAL_CLOSED" and "휴진일" in out["reply"]


# ---- R-4 보호자 대리 -------------------------------------------------------------

WARD = {"wardId": "ward-1", "wardName": "김영희", "name": "김영희", "phone": "010-9999-0000"}


def test_guardian_proxy_card_says_whose_reservation_it_is() -> None:
    fake = FakeReservation()
    out = _confirm_card(make_orchestrator(fake, FULL), fake, **WARD)
    assert "김영희님의 예약을 보호자님이 대신" in out["reply"] and "환자: 김영희님" in out["reply"]


def test_confirm_from_a_different_ward_context_is_rejected() -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake, **WARD)
    other = _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")), **{**WARD, "wardId": "ward-2"})
    assert fake.names() == [] and "시간이 지났" in other["reply"]


# ---- R-5 지역 ---------------------------------------------------------------------


def test_profile_region_search_offers_other_region_button() -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake, {"department": "내과"})
    out = orch.handle(make_request("근처 내과 예약해줘"), {})
    assert out["tool"] == "search_hospital"
    assert out["ui"].field == "location_confirm" and out["ui"].options[0].value == "__other__"
    asked = orch.handle(make_request(ui=("location_confirm", "__other__")), {})
    assert asked["ui"].field == "location" and "지역" in asked["reply"]


def test_region_the_user_typed_does_not_get_other_region_button() -> None:
    orch = make_orchestrator(FakeReservation(), {"department": "내과", "location": "서울 송파구"})
    out = orch.handle(make_request("송파구 내과 예약해줘"), {})
    assert out["tool"] == "search_hospital" and out["ui"] is None


# ---- R-3 취소 ---------------------------------------------------------------------


def _with_appointments() -> FakeReservation:
    fake = FakeReservation()
    fake.appointments = {
        5: {"reservationId": 5, "hospitalName": "강남메디내과", "department": "내과", "reservationDate": FUTURE, "reservationTime": "09:00", "status": "BOOKED"},
        6: {"reservationId": 6, "hospitalName": "서초내과", "department": "내과", "reservationDate": FUTURE, "reservationTime": "10:00", "status": "CANCELED"},
    }
    return fake


def test_cancel_lists_only_cancellable_and_confirms_before_calling() -> None:
    fake = _with_appointments()
    orch = make_orchestrator(fake)
    pick = _turn(orch, None, "예약 취소할래")
    assert pick["ui"].field == "reservation_cancel_pick"
    assert [o.value for o in pick["ui"].options] == ["5", "__stop__"]
    confirm = _turn(orch, pick, ui=("reservation_cancel_pick", "5"))
    assert confirm["ui"].field == "reservation_cancel_confirm" and "강남메디내과" in confirm["reply"] and "취소할까요?" in confirm["reply"]
    assert fake.names() == []  # 확인 전에는 취소하지 않는다
    done = _turn(orch, confirm, ui=("reservation_cancel_confirm", _value(confirm, "yes:")))
    assert fake.names() == ["cancel"] and fake.calls[0][1] == 5 and "취소했어요" in done["reply"]


def test_cancel_keep_and_forged_token_do_not_cancel() -> None:
    fake = _with_appointments()
    orch = make_orchestrator(fake)
    confirm = _turn(orch, _turn(orch, None, "예약 취소할래"), ui=("reservation_cancel_pick", "5"))
    kept = _turn(orch, confirm, ui=("reservation_cancel_confirm", _value(confirm, "no:")))
    forged = _turn(orch, confirm, ui=("reservation_cancel_confirm", "yes:forged"))
    skipped = _turn(orch, None, ui=("reservation_cancel_confirm", _value(confirm, "yes:")))
    assert fake.names() == []
    assert "그대로 둘게요" in kept["reply"] and "시간이 지났" in forged["reply"] and "시간이 지났" in skipped["reply"]


def test_someone_elses_reservation_id_relays_the_service_404() -> None:
    fake = _with_appointments()
    orch = make_orchestrator(fake)
    out = _turn(orch, _turn(orch, None, "예약 취소할래"), ui=("reservation_cancel_pick", "999"))
    assert out["type"] == "tool_error" and "예약을 찾을 수 없습니다" in out["reply"] and fake.names() == []


def test_cancelled_reservation_cannot_be_cancelled_again_and_empty_list_is_told() -> None:
    fake = _with_appointments()
    orch = make_orchestrator(fake)
    again = _turn(orch, _turn(orch, None, "예약 취소할래"), ui=("reservation_cancel_pick", "6"))
    assert "이미 취소" in again["reply"] and fake.names() == []
    fake.appointments = {}
    assert "취소할 수 있는 예약이 없어요" in _turn(orch, None, "예약 취소해줘")["reply"]


def test_cancel_failure_from_service_is_reported() -> None:
    fake = _with_appointments()
    fake.cancel_result = {"httpStatus": 400, "data": {"code": "INVALID_STATUS", "message": "취소할 수 없는 예약 상태입니다."}}
    orch = make_orchestrator(fake)
    confirm = _turn(orch, _turn(orch, None, "예약 취소할래"), ui=("reservation_cancel_pick", "5"))
    out = _turn(orch, confirm, ui=("reservation_cancel_confirm", _value(confirm, "yes:")))
    assert out["type"] == "tool_error" and out["error"]["code"] == "INVALID_STATUS"


def test_lookup_sentence_about_cancelled_reservations_is_not_a_cancel_request() -> None:
    from app.services.reservation_orchestrator import ReservationOrchestrator as R

    assert R._is_cancel_request("예약 취소해줘") and R._is_cancel_request("내일 예약 취소할게요")
    assert not R._is_cancel_request("취소된 예약 내역 보여줘")
    assert not R._is_cancel_request("내과 예약해줘")
    assert R._is_change_request("예약 시간 바꿔줘") and not R._is_change_request("내일로 바꿔줘")


# ---- 변경: 새 예약 확정 뒤에만 기존 예약 취소 ------------------------------------------


def _change_to_card(orch, fake) -> dict[str, Any]:
    pick = _turn(orch, None, "예약 바꿔줘")
    assert pick["ui"].field == "reservation_change_pick"
    asked = _turn(orch, pick, ui=("reservation_change_pick", "5"))
    assert asked["ui"].kind == "date" and "바꿀게요" in asked["reply"]
    slots = _turn(orch, asked, ui=("date", FUTURE2))
    card = _turn(orch, slots, ui=("time", f"{FUTURE2} 10:00"))
    assert card["ui"].field == "reservation_confirm" and "기존 예약" in card["reply"]
    return card


def test_change_creates_new_first_then_cancels_old() -> None:
    fake = _with_appointments()
    orch = make_orchestrator(fake)
    card = _change_to_card(orch, fake)
    assert fake.names() == []
    done = _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")))
    assert fake.names() == ["create", "cancel"] and fake.calls[1][1] == 5
    assert fake.calls[0][1]["reservation_date"] == FUTURE2 and "기존 예약" in done["reply"] and "취소했어요" in done["reply"]


def test_change_keeps_old_reservation_when_new_one_fails() -> None:
    fake = _with_appointments()
    fake.create_result = {"httpStatus": 400, "data": {"code": "HOSPITAL_CLOSED", "message": "휴진일입니다."}}
    orch = make_orchestrator(fake)
    card = _change_to_card(orch, fake)
    out = _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")))
    assert fake.names() == ["create"] and out["type"] == "tool_error" and fake.appointments[5]["status"] == "BOOKED"


def test_change_reports_and_offers_retry_when_old_cancel_fails() -> None:
    fake = _with_appointments()
    fake.cancel_result = {"httpStatus": 500, "data": None}
    orch = make_orchestrator(fake)
    card = _change_to_card(orch, fake)
    out = _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")))
    assert "새 예약은 완료" in out["reply"] and "취소에 실패" in out["reply"]
    assert out["ui"].field == "reservation_cancel_confirm" and out["tool"] == "make_appointment"
    fake.cancel_result = None
    retry = _turn(orch, out, ui=("reservation_cancel_confirm", _value(out, "yes:")))
    assert fake.names()[-1] == "cancel" and "취소했어요" in retry["reply"]


def test_change_with_no_reservation_says_so() -> None:
    assert "바꿀 수 있는 예약이 없어요" in _turn(make_orchestrator(FakeReservation()), None, "예약 바꿔줘")["reply"]


# ---- chat_service 연결 -------------------------------------------------------------


def test_new_ui_fields_are_routed_as_reservation_flow() -> None:
    from app.services.reservation_confirm_flow import CONFIRM_FIELDS
    from app.services.reservation_orchestrator import ReservationOrchestrator as R

    assert CONFIRM_FIELDS == {"reservation_confirm", "reservation_edit", "reservation_cancel_pick", "reservation_cancel_confirm", "reservation_change_pick"}
    assert all(R._reservation_related_ui(f) for f in CONFIRM_FIELDS)


def test_last_flow_state_reads_only_the_immediately_previous_turn_of_the_same_session() -> None:
    from app.services.chat_service import ChatService

    def db_with(session_id: str, meta: dict[str, Any]):
        log = SimpleNamespace(context_json=json.dumps({"sessionId": session_id, "upstreamMeta": meta}))
        chain = SimpleNamespace()
        chain.filter = lambda *_a, **_k: chain
        chain.order_by = lambda *_a, **_k: chain
        chain.first = lambda: log
        return SimpleNamespace(query=lambda *_a, **_k: chain)

    state = {"pendingAction": {"kind": "create", "token": "t"}, "router": {"x": 1}, "flowState": {"draft": {}}}
    got = ChatService._last_flow_state(db_with("s1", state), "u", "s1")
    assert set(got) == {"pendingAction", "flowState"}  # 라우터 등 다른 값은 넘기지 않는다
    assert ChatService._last_flow_state(db_with("other", state), "u", "s1") == {}
    assert ChatService._last_flow_state(db_with("s1", {"pendingAction": "문자열"}), "u", "s1") == {}


def test_cancel_and_change_sentences_override_a_non_reservation_route() -> None:
    from app.services.chat_service import ChatService

    # 응급 신호가 있으면 예약 쪽으로 끌려가지 않는다(키워드 층이 먼저 강제한다)
    forced = ChatService._force_emergency({"route": "reservation", "emergency": False}, "숨을 못 쉬겠어요 예약 취소할래")
    assert forced["route"] == "symptom" and forced["emergency"] is True
