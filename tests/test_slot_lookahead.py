from __future__ import annotations

from datetime import date, timedelta

from app.services.reservation_intake import CHANGE_DATE_OPTION, ReservationDraft, build_reservation_draft
from app.services.reservation_orchestrator import ReservationOrchestrator


class _FakeReservation:
    def __init__(self, slots_by_date: dict[str, list[str] | None]) -> None:
        self.slots_by_date = slots_by_date
        self.calls: list[str] = []
        self.departments: list[str | None] = []

    def get_available_slots(self, hospital_id: int, date: str, *, api_key_override=None, department=None):  # noqa: ARG002
        self.calls.append(date)
        self.departments.append(department)
        slots = self.slots_by_date.get(date)
        if slots is None:
            return {"httpStatus": 500, "data": None}
        return {"httpStatus": 200, "data": {"availableSlots": slots}}


def _orchestrator(fake: _FakeReservation) -> ReservationOrchestrator:
    orch = ReservationOrchestrator.__new__(ReservationOrchestrator)
    orch._reservation = fake
    return orch


def test_find_slots_moves_to_next_available_day() -> None:
    base = date.today() + timedelta(days=10)
    d0, d1, d2 = [(base + timedelta(days=i)).isoformat() for i in range(3)]
    fake = _FakeReservation({d0: [], d1: [], d2: ["10:00", "10:30"]})
    assert _orchestrator(fake)._find_slots(1, d0, None, {}) == (d2, ["10:00", "10:30"])


def test_find_slots_keeps_requested_day_when_available() -> None:
    d0 = (date.today() + timedelta(days=10)).isoformat()
    fake = _FakeReservation({d0: ["09:00"]})
    assert _orchestrator(fake)._find_slots(1, d0, None, {}) == (d0, ["09:00"])
    assert fake.calls == [d0]


def test_find_slots_gives_up_after_lookahead_window() -> None:
    base = date.today() + timedelta(days=10)
    fake = _FakeReservation({(base + timedelta(days=i)).isoformat(): [] for i in range(10)})
    orch = _orchestrator(fake)
    assert orch._find_slots(1, base.isoformat(), None, {}) == (base.isoformat(), [])
    assert len(fake.calls) == 1 + orch.SLOT_LOOKAHEAD_DAYS


def test_find_slots_returns_none_date_when_lookup_fails() -> None:
    d0 = (date.today() + timedelta(days=10)).isoformat()
    assert _orchestrator(_FakeReservation({d0: None}))._find_slots(1, d0, None, {}) == (None, [])


def test_time_selection_carries_date_and_change_date_clears_both() -> None:
    draft = ReservationDraft(message="", reservation_date="2026-10-12")
    draft.apply_time_selection("2026-10-13 10:00")
    assert (draft.reservation_date, draft.reservation_time) == ("2026-10-13", "10:00")
    draft.apply_time_selection("11:00")
    assert (draft.reservation_date, draft.reservation_time) == ("2026-10-13", "11:00")
    draft.apply_time_selection(CHANGE_DATE_OPTION)
    assert (draft.reservation_date, draft.reservation_time) == (None, None)

    built = build_reservation_draft("", ui_field="time", ui_value="2026-10-13 10:00")
    assert (built.reservation_date, built.reservation_time) == ("2026-10-13", "10:00")


def test_new_request_detection() -> None:
    is_new = ReservationOrchestrator._is_new_request
    assert is_new("근처 내과 예약해줘")
    assert is_new("강남메디내과 예약 잡아줘")
    assert is_new("머리가 너무 아파서 병원 예약하고 싶어요")
    assert is_new("병원 좀 잡아줘 이비인후과로")
    assert not is_new("강남하늘안과로 해줘")  # 병원 고르기(예약 키워드 없음)
    assert not is_new("내일로 바꿔줘")
    assert not is_new("")


def test_gap_prompt_names_the_missing_field() -> None:
    from app.services.reservation_orchestrator import _gap_label, _ui_for_field

    assert "성함" in _gap_label("patient_name")
    assert _ui_for_field("phone").placeholder == "예: 010-1234-5678"


def test_find_slots_drops_times_the_user_already_booked() -> None:
    d0 = (date.today() + timedelta(days=10)).isoformat()
    fake = _FakeReservation({d0: ["09:00", "09:30", "10:00"]})
    fake.list_my_appointments = lambda api_key_override=None: {  # type: ignore[attr-defined]
        "httpStatus": 200,
        "data": [
            {"reservationDate": d0, "reservationTime": "09:30", "status": "BOOKED"},
            {"reservationDate": d0, "reservationTime": "10:00", "status": "CANCELED"},
        ],
    }
    assert _orchestrator(fake)._find_slots(1, d0, None, {}) == (d0, ["09:00", "10:00"])


def test_symptom_department_rule() -> None:
    from app.services.reservation_orchestrator import _symptom_department

    assert _symptom_department("머리가 너무 아파서 병원 예약하고 싶어요") == "신경과"
    assert _symptom_department("눈이 충혈됐어요") == "안과"
    assert _symptom_department("강남메디내과 예약 잡아줘") is None


def test_where_to_go_answers_department_before_asking_region() -> None:
    from app.services.reservation_orchestrator import _with_copula

    orch = ReservationOrchestrator.__new__(ReservationOrchestrator)
    orch._loader = type("L", (), {"state": type("S", (), {"model_name": "x"})()})()
    reply = orch._ask_location({}, "내과")["reply"]
    assert reply.startswith("내과로 가시면 돼요.") and "지역" in reply
    assert orch._ask_location({})["reply"] == "어느 지역에서 병원을 찾을까요?"
    assert _with_copula("내과") == "내과예요" and _with_copula("병원") == "병원이에요"


def test_bare_department_word_is_not_a_hospital_name() -> None:
    from app.services.reservation_intake import _find_hospital_name

    assert _find_hospital_name("알았어 정형외과 강남에 있는곳으로 예약해줘") is None
    assert _find_hospital_name("강남정형외과로 예약해줘") == "강남정형외과"


def test_reservation_row_gets_card_keys() -> None:
    row = ReservationOrchestrator._shape_reservation_row({"reservationId": 7, "hospitalName": "강남메디내과", "reservationDate": "2026-10-14", "reservationTime": "09:30"})
    assert (row["hospital_name"], row["date"], row["time"], row["reservation_id"]) == ("강남메디내과", "2026-10-14", "09:30", 7)


def test_department_question_answers_from_last_reply() -> None:
    from app.services.chat_service import ChatService

    hist = [{"role": "assistant", "content": "· 증상: 다리 부상\n정형외과 진료를 권해드려요."}]
    out = ChatService._department_question_reply("병원 무슨과로 가야되는지 알려줘", hist)
    assert out and out["reply"].startswith("정형외과로 가시면 돼요.")
    assert ChatService._department_question_reply("병원 예약해줘", hist) is None


def test_reservation_account_matches_frontend_bff_rule() -> None:
    import hashlib
    import hmac

    from app.services.reservation_credential_service import ReservationCredentialService

    email, password = ReservationCredentialService.account_for("test01", "silverbridge-reservation-dev-secret")
    assert email == "sb-test01@silverbridge.local"
    assert password == hmac.new(b"silverbridge-reservation-dev-secret", b"test01", hashlib.sha256).hexdigest()
