"""예약 날짜 단계 = 선택지(2026-11-04). 피보호자 화면은 키보드·달력 없이 고른다.

- 날짜는 kind=select 로 오늘부터 7일 + [달력에서 다른 날 고르기](__calendar__). 값은 YYYY-MM-DD, 글자는 "오늘 11월 4일(수)".
- __calendar__ 를 누르면 달력(kind=date)으로 다시 묻는다. 형식이 틀리거나 지난 날짜는 날짜를 비워 다시 묻는다.
- 사람에게 보이는 문장·예약 목록 글자의 날짜는 "11월 5일(목)"(내부 값은 ISO 그대로).
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from app.services.chat_service import _HISTORY_SKIP_VALUES
from app.services.reservation_common import _date_choice_label
from app.services.reservation_intake import PICK_CALENDAR_OPTION
from app.services.reservation_orchestrator import DATE_CHOICE_DAYS, ReservationOrchestrator
from tests.test_reservation_choice_only import SELF_NO_ADDRESS, _fake, _Session

ISO = re.compile(r"\d{4}-\d{2}-\d{2}")


def _today() -> date:
    return datetime.now(ZoneInfo("Asia/Seoul")).date()


def _to_date_step() -> tuple[_Session, dict]:
    fake = _fake()
    s = _Session(fake, SELF_NO_ADDRESS)
    s.say("신경과 예약은 어떻게 하면 되나요?")
    s.pick("location_sido", "서울")
    s.pick("location", "서울 서초구")
    date_ask = s.pick("hospital_confirm", "서초내과")
    return s, date_ask


def test_date_step_offers_seven_days_from_today_and_calendar() -> None:
    _, date_ask = _to_date_step()
    ui = date_ask["ui"]
    assert ui.kind == "select" and ui.field == "date"
    values = [o.value for o in ui.options]
    labels = [o.label for o in ui.options]
    today = _today()
    assert values[:-1] == [(today + timedelta(days=i)).isoformat() for i in range(DATE_CHOICE_DAYS)]
    assert values[-1] == PICK_CALENDAR_OPTION and labels[-1] == "달력에서 다른 날 고르기"
    assert labels[0].startswith("오늘 ") and labels[1].startswith("내일 ")
    assert not any(ISO.search(label) for label in labels)


def test_calendar_option_asks_with_calendar() -> None:
    s, _ = _to_date_step()
    calendar = s._turn("", ("date", PICK_CALENDAR_OPTION))
    assert calendar["ui"].kind == "date" and calendar["ui"].field == "date"


def test_past_or_malformed_date_is_asked_again() -> None:
    s, _ = _to_date_step()
    for bad in ("2020-01-01", "내일", "2026-13-40"):
        again = s._turn("", ("date", bad))
        assert again["ui"].kind == "select" and again["ui"].field == "date", bad
    assert "create" not in s.orch._reservation.names()  # type: ignore[attr-defined]


def test_time_question_shows_readable_date_not_iso() -> None:
    s, _ = _to_date_step()
    tomorrow = (_today() + timedelta(days=1)).isoformat()
    time_ask = s.pick("date", tomorrow)
    assert time_ask["ui"].field == "time"
    assert not ISO.search(time_ask["reply"]), time_ask["reply"]
    assert not ISO.search(time_ask["ui"].label or ""), time_ask["ui"].label


def test_reservation_choice_label_uses_readable_date() -> None:
    label = ReservationOrchestrator._reservation_choice_label(
        {"hospitalName": "서초내과", "reservationDate": "2026-12-01", "reservationTime": "10:00", "status": "예약"}
    )
    assert label == "서초내과 · 12월 1일(화) · 10:00 · 예약"


def test_date_choice_label() -> None:
    today = date(2026, 11, 4)
    assert _date_choice_label("2026-11-04", today) == "오늘 11월 4일(수)"
    assert _date_choice_label("2026-11-05", today) == "내일 11월 5일(목)"
    assert _date_choice_label("2026-11-06", today) == "11월 6일(금)"


def test_calendar_option_is_not_history_text() -> None:
    assert PICK_CALENDAR_OPTION in _HISTORY_SKIP_VALUES
