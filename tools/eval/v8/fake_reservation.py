"""시험지 v8 가짜 예약 서비스(2026-11-20).

실제 예약 서비스·네트워크를 부르지 않는다. `ReservationApiClient._request`(urllib 층)만 바꿔 끼우므로
지역·진료과 필터와 정렬(`_location_match_score`)·빈 시간 걸러내기 같은 클라이언트 로직은 그대로 시험 대상에 남는다.
병원 46곳은 모두 가상 이름·가상 주소다(tools/eval/v8/fake_hospitals.json).
"""
from __future__ import annotations

import copy
import json
import pathlib
import re
import urllib.parse
from typing import Any

FIXTURE = pathlib.Path(__file__).resolve().parent / "fake_hospitals.json"
FAKE_MEMBER_KEY = "replay-fake-member-key"  # 가짜 회원 예약 키(실제 키 아님)
DEFAULT_SLOTS = ("09:00", "10:00", "11:00", "14:00", "15:00", "16:00")


def load_hospitals(path: str | pathlib.Path | None = None) -> list[dict[str, Any]]:
    data = json.loads(pathlib.Path(path or FIXTURE).read_text(encoding="utf-8"))
    return list(data["hospitals"])


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).casefold()


class FakeReservationBackend:
    """예약 서비스 HTTP 를 흉내 낸다. 호출 기록(calls)과 예약 기록(appointments)을 남긴다."""

    def __init__(self, hospitals: list[dict[str, Any]] | None = None, slots: tuple[str, ...] = DEFAULT_SLOTS) -> None:
        self.hospitals = copy.deepcopy(hospitals if hospitals is not None else load_hospitals())
        self.slots = tuple(slots)
        self.appointments: dict[int, dict[str, Any]] = {}
        self.calls: list[tuple[str, str]] = []
        self._next_id = 1000

    # ---- 조회 헬퍼 ----------------------------------------------------------------------------
    def creates(self) -> int:
        """실제로 만들어진 예약 수(거절된 시도는 세지 않는다 - 취소된 예약은 센다)."""
        return len(self.appointments)

    def create_attempts(self) -> int:
        return sum(1 for method, path in self.calls if method == "POST" and path == "/reservations")

    def hospital(self, hid: int) -> dict[str, Any] | None:
        return next((h for h in self.hospitals if h.get("id") == hid), None)

    # ---- HTTP 흉내 ----------------------------------------------------------------------------
    def request(self, method: str, path: str, *, params: dict[str, str] | None = None, body: dict[str, Any] | None = None,
                auth: str = "none", token_override: str | None = None) -> tuple[int, Any]:
        method = method.upper()
        path = "/" + path.lstrip("/")
        path = urllib.parse.urlsplit(path).path
        self.calls.append((method, re.sub(r"/\d+", "/{id}", path)))
        params = dict(params or {})
        needs_key = path.startswith("/reservations")
        if needs_key and auth == "api_key" and (token_override or "").strip() != FAKE_MEMBER_KEY:
            return 401, {"code": "INVALID_API_KEY", "message": "가짜 서비스: 회원 키가 다릅니다."}
        if needs_key and auth not in ("api_key",):
            return 401, {"code": "SERVER_CONFIG", "message": "가짜 서비스: 회원 키가 필요합니다."}
        if method == "GET" and path == "/hospitals":
            return 200, self._list_hospitals(params)
        m = re.fullmatch(r"/hospitals/(\d+)/departments", path)
        if method == "GET" and m:
            h = self.hospital(int(m.group(1)))
            if h is None:
                return 404, {"code": "HOSPITAL_NOT_FOUND"}
            return 200, {"departments": [{"name": d} for d in h["departments"]]}
        m = re.fullmatch(r"/hospitals/(\d+)/available-slots", path)
        if method == "GET" and m:
            return self._slots(int(m.group(1)), params.get("date", ""), params.get("department"))
        if method == "POST" and path == "/reservations":
            return self._create(body or {})
        if method == "GET" and path == "/reservations/my":
            return 200, [dict(r) for r in self.appointments.values()]
        m = re.fullmatch(r"/reservations/(\d+)", path)
        if method == "GET" and m:
            row = self.appointments.get(int(m.group(1)))
            return (200, dict(row)) if row else (404, {"code": "RESERVATION_NOT_FOUND"})
        m = re.fullmatch(r"/reservations/(\d+)/cancel", path)
        if method == "PATCH" and m:
            row = self.appointments.get(int(m.group(1)))
            if row is None:
                return 404, {"code": "RESERVATION_NOT_FOUND"}
            row["status"] = "CANCELED"
            return 200, {"reservationId": row["reservationId"], "status": "CANCELED"}
        return 404, {"code": "FAKE_NOT_IMPLEMENTED"}

    def _list_hospitals(self, params: dict[str, str]) -> list[dict[str, Any]]:
        rows = self.hospitals
        if params.get("name"):
            needle = _norm(params["name"])
            rows = [h for h in rows if needle in _norm(h["name"])]
        if params.get("region"):
            reg = _norm(params["region"])
            rows = [h for h in rows if reg in _norm(h["region"]) or _norm(h["region"]) in reg or reg in _norm(h["address"])]
        if params.get("department"):
            dept = params["department"].strip()
            rows = [h for h in rows if any(dept in d for d in h["departments"])]
        return [copy.deepcopy(h) for h in rows]

    def _booked(self, hid: int, day: str, department: str | None) -> set[str]:
        return {
            r["reservationTime"] for r in self.appointments.values()
            if r["hospitalId"] == hid and r["reservationDate"] == day and r["status"] != "CANCELED"
            and (not department or r["department"] == department)
        }

    def _slots(self, hid: int, day: str, department: str | None) -> tuple[int, Any]:
        h = self.hospital(hid)
        if h is None:
            return 404, {"code": "HOSPITAL_NOT_FOUND"}
        if department and department not in h["departments"]:
            return 400, {"success": False, "code": "INVALID_DEPARTMENT", "message": "선택한 진료과가 병원에 없습니다."}
        booked = self._booked(hid, day, department)
        return 200, {"availableSlots": [s for s in self.slots if s not in booked]}

    def _create(self, body: dict[str, Any]) -> tuple[int, Any]:
        h = self.hospital(int(body.get("hospitalId") or 0))
        if h is None:
            return 404, {"code": "HOSPITAL_NOT_FOUND"}
        dept, day, time = str(body.get("department") or ""), str(body.get("reservationDate") or ""), str(body.get("reservationTime") or "")
        if dept not in h["departments"]:
            return 400, {"code": "INVALID_DEPARTMENT"}
        if time not in self.slots or time in self._booked(h["id"], day, dept):
            return 409, {"code": "SLOT_UNAVAILABLE"}
        self._next_id += 1
        row = {
            "reservationId": self._next_id, "hospitalId": h["id"], "hospitalName": h["name"], "department": dept,
            "reservationDate": day, "reservationTime": time, "status": "BOOKED",
        }
        self.appointments[self._next_id] = row
        return 201, dict(row)


class FakeCredentialService:
    """회원 예약 키 발급을 흉내 낸다 - 항상 가짜 키를 돌려준다(예약 서비스 가입·로그인을 하지 않는다)."""

    def ensure_credential(self, db, *, user_id, client, name="", phone=""):  # noqa: ARG002
        return FAKE_MEMBER_KEY

    def __getattr__(self, name: str):  # 그 밖의 메서드는 아무 일도 하지 않는다
        return lambda *a, **k: None


def install(get_backend) -> None:
    """app 모듈에 가짜를 끼운다(되돌리지 않는다 - 재생 프로세스 전용). 네트워크 호출은 막는다.
    get_backend: 지금 문항의 FakeReservationBackend 를 돌려주는 함수(문항마다 새 기록으로 바꾼다)."""
    import urllib.request

    import app.services.chat_service as cs
    from app.services import reservation_api_client as rac

    def fake_request(self, method, path, *, params=None, body=None, auth="none", token_override=None):  # noqa: ARG001
        return get_backend().request(method, path, params=params, body=body, auth=auth, token_override=token_override)

    rac.ReservationApiClient._request = fake_request
    cs.get_reservation_credential_service = lambda: FakeCredentialService()

    def blocked(*a, **k):  # noqa: ARG001
        raise RuntimeError("replay_network_blocked")

    urllib.request.urlopen = blocked
