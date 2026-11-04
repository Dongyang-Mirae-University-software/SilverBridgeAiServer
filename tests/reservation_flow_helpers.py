"""예약 오케스트레이터 흐름 테스트용 가짜 로더·가짜 예약 서비스. 실제 LLM·예약 서비스를 부르지 않는다."""
from __future__ import annotations

import json
from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any

from app.schemas.chat_schema import ChatContext, ChatRequest, UiSelection
from app.services.reservation_orchestrator import ReservationOrchestrator

FUTURE = (date.today() + timedelta(days=10)).isoformat()
FUTURE2 = (date.today() + timedelta(days=11)).isoformat()


class FakeLoader:
    def __init__(self, extracted: dict[str, Any] | None = None) -> None:
        self.state = SimpleNamespace(model_name="fake")
        self.extracted = extracted or {}

    def ensure_loaded(self) -> None:
        return None

    def generate_structured_text(self, system: str, prompt: str, schema: Any = None, **kwargs) -> str:  # noqa: ARG002
        base = {
            "intent": "hospital_reservation", "hospital_name": "", "department": "", "location": "", "reservation_date": "",
            "reservation_time": "", "patient_name": "", "phone": "", "birth_date": "", "symptom_summary": "", "memo": "",
            "missing_fields": [], "confidence": "high", "needs_search": False,
        }
        return json.dumps({**base, **self.extracted}, ensure_ascii=False)


class FakeReservation:
    """호출 기록을 남기는 예약 서비스. appointments 는 id -> 행."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.appointments: dict[int, dict[str, Any]] = {}
        self.create_result: dict[str, Any] | None = None
        self.cancel_result: dict[str, Any] | None = None
        self.slots: list[str] = ["09:00", "10:00", "14:00"]
        self.slot_departments: list[str | None] = []  # 빈 시간 조회에 넘긴 진료과(시간 칸은 진료과별)
        # 병원 id -> 진료과. 정해 두면 진짜 예약 서비스처럼 병원에 없는 진료과의 빈 시간 조회를 400 INVALID_DEPARTMENT 로 답한다
        self.departments_by_hospital: dict[int, list[str]] | None = None
        # 병원이 있는 지역(지역 선택지 출처, 2026-11-04). None 이면 조회 실패와 같다 - 지역은 예전처럼 글 입력으로 묻는다.
        self.regions: list[str] | None = None
        self.region_departments: list[str] = []  # 지역 조회에 넘긴 진료과
        self.searched_regions: list[str | None] = []  # 병원 검색에 넘긴 지역(calls 에는 넣지 않는다 - 기존 단언 유지)
        self.search_rows: list[dict[str, Any]] | None = None  # 정하면 병원 검색이 이 결과를 돌려준다
        self.slots_fail = False  # True 면 빈 시간 조회가 500(조회 실패)

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]

    def resolve_hospital(self, name: str, *, api_key_override=None):  # noqa: ARG002
        return 1, "강남메디내과"

    def search_hospitals(self, *a, **k):  # noqa: ARG002
        self.searched_regions.append(k.get("region"))
        if self.search_rows is not None:
            return {"success": True, "rows": list(self.search_rows)}
        return {"success": True, "rows": [{"hospital": "강남메디내과"}, {"hospital": "서초내과"}]}

    def list_regions(self, department: str = "", *, api_key_override=None):  # noqa: ARG002
        self.region_departments.append(department)
        return None if self.regions is None else list(self.regions)

    def list_departments(self, hospital_id: int, *, api_key_override=None):  # noqa: ARG002
        names = (self.departments_by_hospital or {}).get(hospital_id, ["내과"])
        return {"httpStatus": 200, "data": {"departments": [{"name": n} for n in names]}}

    def get_available_slots(self, hospital_id: int, day: str, *, api_key_override=None, department=None):  # noqa: ARG002
        self.slot_departments.append(department)
        if self.slots_fail:
            return {"httpStatus": 500, "data": None}
        offered = (self.departments_by_hospital or {}).get(hospital_id)
        if offered is not None and department and department not in offered:
            return {"httpStatus": 400, "data": {"success": False, "code": "INVALID_DEPARTMENT", "message": "선택한 진료과가 병원에 없습니다."}}
        return {"httpStatus": 200, "data": {"availableSlots": list(self.slots)}}

    def list_my_appointments(self, api_key_override=None):  # noqa: ARG002
        return {"httpStatus": 200, "data": list(self.appointments.values())}

    def get_reservation(self, reservation_id: int, api_key_override=None):  # noqa: ARG002
        row = self.appointments.get(reservation_id)
        if row is None:
            return {"httpStatus": 404, "data": {"message": "예약을 찾을 수 없습니다."}}
        return {"httpStatus": 200, "data": row}

    def create_reservation(self, **kwargs):
        self.calls.append(("create", kwargs))
        if self.create_result is not None:
            return self.create_result
        rid = 100 + len(self.appointments)
        row = {
            "reservationId": rid, "hospitalName": "강남메디내과", "department": kwargs["department"],
            "reservationDate": kwargs["reservation_date"], "reservationTime": kwargs["reservation_time"], "status": "BOOKED",
        }
        self.appointments[rid] = row
        return {"httpStatus": 201, "data": row}

    def cancel_reservation(self, reservation_id: int, api_key_override=None):  # noqa: ARG002
        self.calls.append(("cancel", reservation_id))
        if self.cancel_result is not None:
            return self.cancel_result
        row = self.appointments.get(reservation_id)
        if row is None:
            return {"httpStatus": 404, "data": {"message": "예약을 찾을 수 없습니다."}}
        row["status"] = "CANCELED"
        return {"httpStatus": 200, "data": {"reservationId": reservation_id, "status": "CANCELED", "message": "예약이 취소되었습니다."}}


def make_orchestrator(reservation: FakeReservation, extracted: dict[str, Any] | None = None) -> ReservationOrchestrator:
    return ReservationOrchestrator(SimpleNamespace(chat_history_keep_turns=6), FakeLoader(extracted), reservation)  # type: ignore[arg-type]


def make_request(message: str = "", *, ui: tuple[str, str] | None = None, history=None, context: dict[str, Any] | None = None) -> ChatRequest:
    ctx = {"name": "홍길동", "phone": "010-1234-5678", "location": "서울 강남구", **(context or {})}
    return ChatRequest(
        userId="user-1", message=message, sessionId="s1", history=history or [], context=ChatContext(**ctx),
        uiSelection=UiSelection(field=ui[0], value=ui[1]) if ui else None,
    )
