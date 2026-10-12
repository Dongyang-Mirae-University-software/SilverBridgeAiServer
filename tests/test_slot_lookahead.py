from __future__ import annotations

from datetime import date, timedelta

from app.services.reservation_intake import CHANGE_DATE_OPTION, ReservationDraft, build_reservation_draft
from app.services.reservation_orchestrator import ReservationOrchestrator


class _FakeReservation:
    def __init__(self, slots_by_date: dict[str, list[str] | None]) -> None:
        self.slots_by_date = slots_by_date
        self.calls: list[str] = []

    def get_available_slots(self, hospital_id: int, date: str, *, api_key_override=None):  # noqa: ARG002
        self.calls.append(date)
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
