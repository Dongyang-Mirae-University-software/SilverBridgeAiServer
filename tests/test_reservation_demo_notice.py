"""R-6·R-7: 예약 답에 시연용 안내 한 줄을 붙이지 않는다(2026-11-11 - 사전에 따로 고지) / 빈 시간 조회는 진료과별."""
from __future__ import annotations

import pathlib
from typing import Any

from app.services import reservation_common
from app.services.reservation_api_client import ReservationApiClient, _slot_department
from tests.reservation_flow_helpers import FUTURE, FakeReservation, make_orchestrator
from tests.test_reservation_confirm_cancel import FULL, _confirm_card, _turn, _value, _with_appointments

_DEMO_WORDS = ("시연용", "가상 병원", "실제 병원에는")


def _no_demo(text: str) -> bool:
    return not any(w in text for w in _DEMO_WORDS)


def test_notice_constant_and_helper_are_gone() -> None:
    assert not hasattr(reservation_common, "DEMO_NOTICE") and not hasattr(reservation_common, "with_demo_notice")
    root = pathlib.Path(__file__).resolve().parents[1] / "app"
    for path in root.rglob("*.py"):
        body = path.read_text(encoding="utf-8")
        # 주석 밖 문자열로 남은 시연용 문구가 없다(reservation_common 의 설명 주석만 허용).
        lines = [ln for ln in body.splitlines() if not ln.lstrip().startswith("#")]
        assert all(_no_demo(ln) for ln in lines), path


def test_confirm_card_and_completion_have_no_notice() -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    assert _no_demo(card["reply"]) and card["reply"].strip()
    done = _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")))
    assert done["tool"] == "make_appointment"
    assert done["reply"].startswith("예약이 완료되었습니다.") and _no_demo(done["reply"])


def test_lookup_list_has_no_notice() -> None:
    orch = make_orchestrator(_with_appointments())
    out = _turn(orch, None, "내 예약 조회해줘")
    assert out["tool"] == "list_my_appointments"
    assert out["reply"].startswith("예약") and _no_demo(out["reply"])


def test_hospital_list_has_no_notice() -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake, {"department": "내과", "location": "서울 강남구"})
    out = _turn(orch, None, "강남 내과 병원 예약하고 싶어요")
    assert out["tool"] == "search_hospital", out["reply"]
    assert _no_demo(out["reply"]) and out["ui"] is not None


def test_time_selection_asks_slots_of_the_chosen_department() -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake, {"reservation_date": FUTURE})
    out = _turn(orch, None, "강남메디내과 내과 예약해줘")
    assert out["ui"].field == "time", out["reply"]
    assert fake.slot_departments and set(fake.slot_departments) == {"내과"}


def test_slot_department_picks_real_department_name() -> None:
    assert _slot_department("내과", ["소화기내과", "내과"]) == "내과"  # 같은 이름 우선
    assert _slot_department("정형", ["재활의학과", "정형외과"]) == "정형외과"
    assert _slot_department("", ["내과"]) is None
    assert _slot_department("치과", ["내과"]) is None


def test_search_hospitals_sends_department_to_slot_lookup() -> None:
    client = ReservationApiClient.__new__(ReservationApiClient)
    calls: list[tuple[str, dict[str, Any] | None]] = []

    def fake_request(method: str, path: str, *, params=None, auth="none", token_override=None, **_: Any):  # noqa: ARG001
        calls.append((path, params))
        if path == "/hospitals":
            return 200, [{"id": 7, "name": "가상병원", "region": "서울", "address": "", "departments": ["안과", "내과"]}]
        return 200, {"availableSlots": ["10:00"]}

    client._request = fake_request  # type: ignore[method-assign]
    client._resolve_api_key = lambda override: None  # type: ignore[method-assign]
    out = client.search_hospitals("서울", "내과", region="서울", slot_date=FUTURE)
    slot_params = [p for path, p in calls if path.endswith("/available-slots")]
    assert slot_params == [{"date": FUTURE, "department": "내과"}]
    assert out["rows"][0]["availableSlots"] == [f"{FUTURE} 10:00"]
