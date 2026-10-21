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

    def generate_structured_text(self, system: str, prompt: str, schema: Any = None) -> str:  # noqa: ARG002
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

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]

    def resolve_hospital(self, name: str, *, api_key_override=None):  # noqa: ARG002
        return 1, "강남메디내과"

    def search_hospitals(self, *a, **k):  # noqa: ARG002
        return {"success": True, "rows": [{"hospital": "강남메디내과"}, {"hospital": "서초내과"}]}

    def list_departments(self, hospital_id: int, *, api_key_override=None):  # noqa: ARG002
        return {"httpStatus": 200, "data": {"departments": [{"name": "내과"}]}}

    def get_available_slots(self, hospital_id: int, day: str, *, api_key_override=None):  # noqa: ARG002
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
