"""예약 흐름 기능 점검(2026-10-28) 반영: M-1 진료과를 바꾸면 이전 병원도 비운다 / M-2 마감·형식 오류도 시간 재선택 /
L-1 병원에 없는 진료과(400 INVALID_DEPARTMENT)는 조회 실패가 아니라 진료과 다시 고르기 / L-4 가짜 예약 서비스 진료과 검증."""
from __future__ import annotations

import time
from typing import Any

from app.services.reservation_confirm_flow import draft_to_dict
from app.services.reservation_intake import ReservationDraft
from app.services.reservation_orchestrator import _clear_stale_links, _linked_fields
from tests.reservation_flow_helpers import FUTURE, FakeReservation, make_orchestrator, make_request
from tests.test_reservation_confirm_cancel import FULL, _confirm_card, _turn, _value


def _stored(**kw: Any) -> dict[str, Any]:
    base = {"message": "", "department": "내과", "location": "서울 강남구", "hospital_name": "강남메디내과", "hospital_id": 1,
            "reservation_date": FUTURE, "patient_name": "홍길동", "phone": "010-1234-5678"}
    d = ReservationDraft(**{**base, **kw})
    return {"flowState": {"draft": draft_to_dict(d), "replaces": None, "wardId": "", "createdAt": int(time.time())}}


def _handle(orch, stored, normalized: str = "", raw: str = "", ui: tuple[str, str] | None = None) -> dict[str, Any]:
    meta: dict[str, Any] = {"storedFlow": stored}
    if raw:
        meta["rawMessage"] = raw
    out = orch.handle(make_request(normalized, ui=ui), meta)
    assert out is not None
    return out


def _draft(out: dict[str, Any]) -> dict[str, Any]:
    m = out.get("meta") or {}
    return (m.get("flowState") or m.get("pendingAction") or {}).get("draft") or {}


def test_m1_changing_department_mid_flow_drops_the_old_hospital() -> None:
    fake = FakeReservation()
    fake.departments_by_hospital = {1: ["내과"]}
    orch = make_orchestrator(fake, {"department": "안과"})
    out = _handle(orch, _stored(), "안과 예약", "안과로 할게요")  # 라우터 규격화 문장 + 동사 없는 원문
    d = _draft(out)
    assert d.get("department") == "안과"
    assert d.get("hospital_name") in (None, "") and d.get("hospital_id") is None, "내과 병원에 안과를 묻지 않는다"
    assert "안과" not in fake.slot_departments


def test_m1_department_first_set_after_hospital_is_not_a_change() -> None:
    meta: dict[str, Any] = {}
    d = ReservationDraft(message="", hospital_name="강남메디내과", hospital_id=1)
    before = _linked_fields(d)
    d.department = "내과"  # 병원을 고른 뒤 진료과를 처음 고름
    _clear_stale_links(before, d, meta)
    assert d.hospital_name == "강남메디내과" and d.hospital_id == 1


def test_m1_department_and_hospital_chosen_together_are_kept() -> None:
    meta: dict[str, Any] = {}
    d = ReservationDraft(message="", hospital_name="강남메디내과", hospital_id=1, department="내과")
    before = _linked_fields(d)
    d.department, d.hospital_name = "안과", "서초안과"
    _clear_stale_links(before, d, meta)
    assert d.hospital_name == "서초안과" and d.hospital_id is None  # 병원이 바뀌어 옛 id 만 비운다


def _rejected_on_create(code: str, message: str) -> dict[str, Any]:
    fake = FakeReservation()
    fake.create_result = {"httpStatus": 400, "data": {"success": False, "code": code, "message": message}}
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    return _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")))


def test_m2_slot_taken_on_create_leads_to_time_reselect() -> None:
    out = _rejected_on_create("SLOT_TAKEN", "이미 마감된 시간대입니다.")
    assert out["ui"].field == "time" and out.get("type") != "tool_error"
    assert out["reply"].startswith("그 시간은 이미 마감됐어요.")


def test_m2_invalid_time_on_create_leads_to_time_reselect() -> None:
    out = _rejected_on_create("INVALID_TIME", "예약 시간이 병원 운영시간 또는 슬롯에 맞지 않습니다.")
    assert out["ui"].field == "time" and out.get("type") != "tool_error"
    assert out["reply"].startswith("그 시간에는 예약할 수 없어요.")


def test_l1_department_not_offered_asks_the_hospital_departments() -> None:
    fake = FakeReservation()
    fake.departments_by_hospital = {1: ["가정의학과", "소화기내과"]}  # 저장 초안의 '내과'가 그 사이 사라짐(이름 변경)
    orch = make_orchestrator(fake, {})
    out = _handle(orch, _stored(), ui=("date", FUTURE))
    assert out["ui"].field == "department", out["reply"]
    assert {o.value for o in out["ui"].options} == {"가정의학과", "소화기내과"}
    assert "_departmentNotOffered" not in (out.get("meta") or {}), "같은 턴 표시는 저장되지 않는다"


def test_l4_fake_reservation_validates_department() -> None:
    fake = FakeReservation()
    fake.departments_by_hospital = {1: ["내과"]}
    assert fake.get_available_slots(1, FUTURE, department="안과")["httpStatus"] == 400
    assert fake.get_available_slots(1, FUTURE, department="내과")["httpStatus"] == 200
