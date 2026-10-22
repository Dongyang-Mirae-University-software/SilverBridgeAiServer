"""예약 확인 흐름 점검(2026-10-22) 이슈 반영 검증: 토큰 한 번만 쓰기(M-1), 시나리오 빈틈(L-1), 문장 오인(L-2), 목록 제한(L-3), 로그(L-4)."""
from __future__ import annotations

import logging
import threading
import time
from datetime import date, timedelta
from typing import Any

import pytest

from app.services.reservation_orchestrator import ReservationOrchestrator as R
from tests.reservation_flow_helpers import FUTURE, FakeReservation, make_orchestrator, make_request
from tests.test_reservation_confirm_cancel import FULL, _confirm_card, _stored, _turn, _value, _with_appointments
from tests.test_reservation_confirm_chat_service import _send, _token, env  # noqa: F401


# ---- M-1 같은 확인 버튼 동시 클릭 ------------------------------------------------------


def _slow_create(fake: FakeReservation, delay: float = 0.2) -> None:
    original = fake.create_reservation

    def slow(**kwargs):
        time.sleep(delay)  # 예약 서비스 응답 지연 - 선점이 없으면 둘 다 검증을 통과해 두 번 만들어진다
        return original(**kwargs)

    fake.create_reservation = slow  # type: ignore[assignment]


def _click_concurrently(orch, stored: dict[str, Any], ui: tuple[str, str], n: int = 3) -> list[dict[str, Any]]:
    outs: list[dict[str, Any]] = []
    barrier = threading.Barrier(n)

    def click() -> None:
        barrier.wait()
        outs.append(orch.handle(make_request(ui=ui), {"storedFlow": stored}))

    threads = [threading.Thread(target=click) for _ in range(n)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    return outs


def test_simultaneous_confirm_clicks_create_exactly_one_reservation() -> None:
    fake = FakeReservation()
    _slow_create(fake)
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    outs = _click_concurrently(orch, _stored(card), ("reservation_confirm", _value(card, "confirm:")))
    assert fake.names().count("create") == 1
    assert sorted(o["tool"] or "" for o in outs) == ["", "", "make_appointment"]
    assert sum("이미 처리 중" in o["reply"] for o in outs) == 2


def test_simultaneous_cancel_clicks_cancel_exactly_once() -> None:
    fake = _with_appointments()
    original = fake.cancel_reservation

    def slow(reservation_id, api_key_override=None):
        time.sleep(0.2)
        return original(reservation_id, api_key_override)

    fake.cancel_reservation = slow  # type: ignore[assignment]
    orch = make_orchestrator(fake)
    confirm = _turn(orch, _turn(orch, None, "예약 취소할래"), ui=("reservation_cancel_pick", "5"))
    _click_concurrently(orch, _stored(confirm), ("reservation_cancel_confirm", _value(confirm, "yes:")))
    assert fake.names().count("cancel") == 1


def test_claim_is_not_spent_by_forged_or_edit_tokens() -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    _turn(orch, card, ui=("reservation_confirm", "confirm:forged0000"))  # 위조는 선점하지 않는다
    done = _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")))
    assert fake.names() == ["create"] and done["tool"] == "make_appointment"


def test_existing_booking_at_same_time_is_caught_before_create() -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    fake.appointments = {9: {"reservationId": 9, "hospitalName": "서초내과", "department": "내과", "reservationDate": FUTURE, "reservationTime": "14:00", "status": "BOOKED"}}
    out = _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")))
    assert fake.names() == [] and "같은 시간대에 이미 예약" in out["reply"] and out["ui"].field == "time"


def test_precheck_lookup_failure_does_not_block_booking() -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    fake.list_my_appointments = lambda api_key_override=None: {"httpStatus": 500, "data": None}  # type: ignore[assignment]
    out = _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")))
    assert fake.names() == ["create"] and out["tool"] == "make_appointment"


# ---- L-1 점검 시나리오 편입 --------------------------------------------------------------


@pytest.mark.parametrize("val", ["0", "-1", "9" * 30, "abc", "5.0", "5;6", ""])
def test_odd_pick_values_never_reach_confirm_or_cancel(val: str) -> None:
    fake = _with_appointments()
    orch = make_orchestrator(fake)
    out = _turn(orch, _turn(orch, None, "예약 취소할래"), ui=("reservation_cancel_pick", val))
    assert "cancel" not in fake.names() and (out.get("ui") is None or out["ui"].field != "reservation_cancel_confirm")


@pytest.mark.parametrize("val", ["+5", " 5 ", "５"])
def test_numeric_looking_pick_values_still_need_confirmation(val: str) -> None:
    fake = _with_appointments()
    orch = make_orchestrator(fake)
    out = _turn(orch, _turn(orch, None, "예약 취소할래"), ui=("reservation_cancel_pick", val))
    assert out["ui"].field == "reservation_cancel_confirm" and fake.names() == []


def test_cross_kind_and_empty_tokens_are_rejected() -> None:
    fake = _with_appointments()
    orch = make_orchestrator(fake, FULL)
    confirm = _turn(orch, _turn(orch, None, "예약 취소할래"), ui=("reservation_cancel_pick", "5"))
    tok = _value(confirm, "yes:").split(":")[1]
    for ui in (("reservation_confirm", f"confirm:{tok}"), ("reservation_cancel_confirm", "yes:"), ("reservation_edit", f"time:{tok}")):
        assert "시간이 지났" in _turn(orch, confirm, ui=ui)["reply"]
    assert fake.names() == []


def test_edit_token_cannot_be_replayed_after_progress() -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    edit = _turn(orch, card, ui=("reservation_confirm", _value(card, "edit:")))
    pick = _turn(orch, edit, ui=("reservation_edit", _value(edit, "time:")))
    replay = _turn(orch, pick, ui=("reservation_edit", _value(edit, "date:")))  # 이미 지난 edit 토큰
    assert "시간이 지났" in replay["reply"]


def test_new_request_text_leaves_the_edit_flow_without_booking() -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    edit = _turn(orch, card, ui=("reservation_confirm", _value(card, "edit:")))
    pick = _turn(orch, edit, ui=("reservation_edit", _value(edit, "time:")))
    _turn(orch, pick, "근처 안과 예약해줘")
    assert fake.names() == []


def test_change_is_refused_when_old_reservation_was_cancelled_meanwhile() -> None:
    fake = _with_appointments()
    orch = make_orchestrator(fake)
    pick = _turn(orch, None, "예약 바꿔줘")
    fake.appointments[5]["status"] = "CANCELED"
    out = _turn(orch, pick, ui=("reservation_change_pick", "5"))
    assert "이미 취소" in out["reply"] and fake.names() == []


@pytest.mark.parametrize("stage", ["cancel_pick", "cancel_confirm", "confirm_card"])
def test_emergency_text_in_every_stage_goes_to_counselling(env, stage: str) -> None:  # noqa: F811
    env.fake.appointments = {5: {"reservationId": 5, "hospitalName": "강남메디내과", "department": "내과", "reservationDate": FUTURE, "reservationTime": "09:00", "status": "BOOKED"}}
    if stage == "cancel_pick":
        _send(env, "예약 취소할래")
    elif stage == "cancel_confirm":
        _send(env, "예약 취소할래")
        _send(env, ui=("reservation_cancel_pick", "5"))
    else:
        _send(env, "강남메디내과 내과 14:00 예약해줘")
    out = _send(env, "숨을 못 쉬겠어요 119 불러야 해요")
    assert out["intent"] == "emergency_guidance" and env.llm.calls == 1 and env.fake.names() == []


def test_two_sessions_of_the_same_user_do_not_cross(env) -> None:  # noqa: F811
    from app.schemas.chat_schema import ChatContext, ChatRequest, UiSelection

    card = _send(env, "강남메디내과 내과 14:00 예약해줘")
    other = ChatRequest(
        userId="user-1", message="", sessionId="s2", context=ChatContext(name="홍길동", phone="010-1234-5678", location="서울 강남구"),
        uiSelection=UiSelection(field="reservation_confirm", value=_token(card, "confirm:")),
    )
    out = env.svc.process_message(env.db, other)
    assert env.fake.names() == [] and "시간이 지났" in out["reply"]


# ---- L-2 문장 오인 ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,cancel,change,how",
    [
        ("예약 취소할래", True, False, False),
        ("내일 예약 취소할게요", True, False, False),
        ("예약 바꿔줘", False, True, False),
        ("예약 시간 바꿔줘", False, True, False),
        ("예약 취소 말고 변경할래", False, True, False),
        ("예약 변경 말고 취소할래", True, False, False),
        ("예약 잘 됐어요 취소 안 할게요", False, False, False),
        ("예약을 취소하지 않을래요", False, False, False),
        ("취소된 예약 내역 보여줘", False, False, False),
        ("병원 예약 변경 방법 알려줘", False, False, True),
        ("예약 취소 어떻게 해요?", False, False, True),
        ("내과 예약해줘", False, False, False),
        ("내일로 바꿔줘", False, False, False),
    ],
)
def test_cancel_and_change_sentence_classification(text: str, cancel: bool, change: bool, how: bool) -> None:
    assert (R._is_cancel_request(text), R._is_change_request(text), R._is_cancel_change_question(text)) == (cancel, change, how)


def test_how_to_question_gets_guidance_not_a_booking_flow() -> None:
    fake = FakeReservation()
    out = make_orchestrator(fake).handle(make_request("병원 예약 변경 방법 알려줘"), {})
    assert out is not None and out["ui"] is None and "예약 취소할래" in out["reply"] and fake.names() == []


# ---- L-3 목록 제한 ---------------------------------------------------------------------


def _row(i: int, day: str, status: str = "BOOKED") -> dict[str, Any]:
    return {"reservationId": i, "hospitalName": f"병원{i}", "department": "내과", "reservationDate": day, "reservationTime": "10:00", "status": status}


def test_pick_list_hides_past_dates_sorts_by_date_and_caps_at_ten() -> None:
    fake = FakeReservation()
    today = date.today()
    past = (today - timedelta(days=3)).isoformat()
    fake.appointments = {
        1: _row(1, past),
        **{i: _row(i, (today + timedelta(days=30 - i)).isoformat()) for i in range(2, 15)},  # 13건 미래, id 가 클수록 빠른 날짜
    }
    out = make_orchestrator(fake).handle(make_request("예약 취소할래"), {})
    values = [o.value for o in out["ui"].options]
    assert "1" not in values  # 지난 예약
    assert values[-1] == "__stop__" and len(values) == 11  # 10건 + 그만두기
    assert values[:3] == ["14", "13", "12"]  # 가까운 날짜순
    assert "가까운 10건만" in out["reply"]


def test_pick_list_with_only_past_reservations_says_nothing_to_cancel() -> None:
    fake = FakeReservation()
    fake.appointments = {1: _row(1, (date.today() - timedelta(days=1)).isoformat())}
    assert "취소할 수 있는 예약이 없어요" in make_orchestrator(fake).handle(make_request("예약 취소할래"), {})["reply"]


# ---- L-4 로그에 환자 이름이 남지 않는다 ------------------------------------------------------


def test_patient_name_is_not_written_to_trace_or_logs(caplog) -> None:
    caplog.set_level(logging.INFO)
    fake = FakeReservation()
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    done = _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")))
    traces = str(card["meta"].get("decisionTrace")) + str(done["meta"].get("decisionTrace"))
    assert "홍길동" not in traces and "홍길동" not in caplog.text
    assert "patient_name_len" in traces
