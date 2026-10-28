from __future__ import annotations

import json
import logging
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any
from functools import lru_cache

from app.core.config import Settings
from app.services.chat_deadline import call_timeout
from app.services.reservation_common import log_len

_LOG = logging.getLogger(__name__)
_SEOUL_TZ = ZoneInfo("Asia/Seoul")


def _normalize_location_text(value: str) -> str:
    text = re.sub(r"\s+", "", (value or "").strip())
    if not text:
        return ""
    replacements = (
        ("특별자치시", ""),
        ("특별자치도", ""),
        ("특별시", ""),
        ("광역시", ""),
        ("자치시", ""),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _location_match_score(query: str, target: str) -> int:
    normalized_query = _normalize_location_text(query)
    normalized_target = _normalize_location_text(target)
    if not normalized_query or not normalized_target:
        return 0
    if normalized_query == normalized_target:
        return 3
    if normalized_query in normalized_target or normalized_target in normalized_query:
        return 2
    return 1 if normalized_query.split("구")[0] and normalized_query.split("구")[0] in normalized_target else 0


def _department_matches(query: str, departments: list[Any] | tuple[Any, ...] | None, hospital_name: str = "") -> bool:
    dept = (query or "").strip()
    if not dept:
        return True
    candidates: list[str] = []
    if isinstance(departments, list):
        candidates.extend(str(item).strip() for item in departments if str(item).strip())
    if hospital_name:
        candidates.append(str(hospital_name).strip())
    return any(dept in candidate for candidate in candidates)


def _slot_department(query: str, departments: list[Any] | tuple[Any, ...] | None) -> str | None:
    """병원 진료과 중 사용자가 찾은 진료과에 해당하는 실제 이름(같은 이름 우선, 없으면 그 말을 포함한 첫 진료과).
    시간 칸은 진료과마다 따로라 빈 시간 조회에 이 이름을 넘긴다. 못 고르면 None(병원 전체 기준)."""
    q = (query or "").strip()
    if not q or not isinstance(departments, (list, tuple)):
        return None
    names = [str(item).strip() for item in departments if str(item).strip()]
    if q in names:
        return q
    return next((name for name in names if q in name), None)


def _filter_future_slot_times(slot_date: str | None, slots: list[str]) -> list[str]:
    date_text = (slot_date or "").strip()
    if not date_text:
        return slots

    today = datetime.now(_SEOUL_TZ).date()
    try:
        target_date = datetime.strptime(date_text, "%Y-%m-%d").date()
    except ValueError:
        return slots

    if target_date != today:
        return slots

    now_hm = datetime.now(_SEOUL_TZ).time().replace(second=0, microsecond=0)
    filtered: list[str] = []
    for slot in slots:
        slot_text = str(slot).strip()
        if not slot_text:
            continue
        time_text = slot_text.split()[-1]
        try:
            slot_time = datetime.strptime(time_text, "%H:%M").time()
        except ValueError:
            continue
        if slot_time > now_hm:
            filtered.append(slot_text)
    return filtered


# 예약 API 호출이 남겨 둘 시간(응답 정리·챗 기록 저장 몫)과 이보다 짧으면 시작하지 않는 최소 시간(초).
_API_RESERVE_SEC = 2.0
_API_MIN_SEC = 3.0


def _stage_path(path: str) -> str:
    """로그용 경로 - 숫자 ID 는 {id}로 바꾼다."""
    return re.sub(r"/\d+", "/{id}", path)


class ReservationApiClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._base = settings.reservation_api_base_url.rstrip("/")
        self._timeout = settings.reservation_api_timeout_sec
        self._service_user_jwt: str | None = settings.reservation_user_jwt.strip() or None
        self._service_user_login_failed = False
        self._service_user_lock = threading.Lock()

    def _resolve_api_key(self, api_key_override: str | None) -> str | None:
        key = (api_key_override or "").strip() or self._settings.reservation_api_key.strip()
        return key or None

    def _resolve_user_jwt(self, token_override: str | None) -> str | None:
        tok = (token_override or "").strip() or self._settings.reservation_user_jwt.strip()
        if tok:
            return tok
        if self._service_user_jwt:
            return self._service_user_jwt
        with self._service_user_lock:
            if self._service_user_jwt:
                return self._service_user_jwt
            if self._service_user_login_failed:
                return None
            token = self._login_service_user()
            if token:
                self._service_user_jwt = token
                return token
            self._service_user_login_failed = True
            return None

    def _login_service_user(self) -> str | None:
        if not self._settings.has_reservation_service_credentials:
            return None
        st, data = self._request(
            "POST",
            "/auth/login",
            body={
                "email": self._settings.reservation_service_email.strip(),
                "password": self._settings.reservation_service_password,
            },
            auth="none",
        )
        if st == 200 and isinstance(data, dict):
            token = str(data.get("accessToken") or data.get("access_token") or "").strip()
            if token:
                return token
        return None

    def login_account(self, email: str, password: str) -> dict[str, Any]:
        st, data = self._request(
            "POST",
            "/auth/login",
            body={"email": email, "password": password},
            auth="none",
        )
        return {"httpStatus": st, "data": data}

    def signup_account(self, email: str, password: str, name: str, phone: str) -> dict[str, Any]:
        st, data = self._request(
            "POST",
            "/auth/signup",
            body={"email": email, "password": password, "name": name, "phone": phone},
            auth="none",
        )
        return {"httpStatus": st, "data": data}

    def issue_api_key(self, access_token: str, name: str) -> dict[str, Any]:
        st, data = self._request(
            "POST",
            "/auth/api-keys",
            body={"name": name},
            auth="user",
            token_override=access_token,
        )
        return {"httpStatus": st, "data": data}

    def has_valid_reservation_key(self, api_key: str | None) -> bool:
        key = (api_key or "").strip()
        if not key:
            return False
        st, data = self._request("GET", "/reservations/my", auth="api_key", token_override=key)
        return 200 <= st < 300 and isinstance(data, list)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
        auth: str = "none",
        token_override: str | None = None,
    ) -> tuple[int, Any]:
        url = f"{self._base}{path if path.startswith('/') else '/' + path}"
        if params:
            query = urllib.parse.urlencode(params)
            url = f"{url}?{query}"
        headers: dict[str, str] = {"Accept": "application/json"}
        if body is not None or method.upper() in {"POST", "PUT", "PATCH"}:
            headers["Content-Type"] = "application/json"
        if auth == "none":
            api_key = self._resolve_api_key(None)
            if api_key:
                headers["X-API-Key"] = api_key
        elif auth == "api_key":
            api_key = self._resolve_api_key(token_override)
            if not api_key:
                return 401, {"code": "SERVER_CONFIG", "message": "RESERVATION_API_KEY 또는 저장된 reservation key가 필요합니다."}
            headers["X-API-Key"] = api_key
        elif auth == "user":
            token = self._resolve_user_jwt(token_override)
            if not token:
                return 401, {"code": "SERVER_CONFIG", "message": "RESERVATION_USER_JWT 또는 access_token이 필요합니다."}
            headers["Authorization"] = f"Bearer {token}"
        else:
            token = (token_override or "").strip() or self._settings.reservation_admin_jwt.strip()
            if not token:
                return 401, {"code": "SERVER_CONFIG", "message": "RESERVATION_ADMIN_JWT 또는 access_token(관리자)이 필요합니다."}
            headers["Authorization"] = f"Bearer {token}"

        # 챗 전체 마감 안에서는 남은 시간만큼만 기다린다(O-3). 모자라면 호출하지 않고 ChatDeadlineExceeded -
        # 경로·숫자 ID 만 단계 이름에 남긴다(쿼리의 지역·병원명은 넣지 않는다).
        timeout = call_timeout(
            f"reservation_api:{method.upper()}:{_stage_path(path)}", float(self._timeout),
            reserve=_API_RESERVE_SEC, minimum=_API_MIN_SEC,
        )
        req = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None,
            headers=headers,
            method=method.upper(),
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                text = response.read().decode("utf-8")
                data = json.loads(text) if text.strip() else None
                return response.status, data
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(detail) if detail.strip() else None
            except json.JSONDecodeError:
                data = {"_raw": detail[:500]}
            return exc.code, data
        except urllib.error.URLError as exc:
            return 0, {"message": str(exc.reason), "code": "UPSTREAM_UNREACHABLE"}
        except TimeoutError:
            # 응답 본문을 읽다 시간이 다 된 경우(urlopen 밖에서 나는 소켓 시간 초과). 마감을 줄인 뒤 더 자주 날 수 있어 실패로 돌려준다.
            return 0, {"message": "timeout", "code": "UPSTREAM_TIMEOUT"}
        except OSError:
            # 응답을 읽다 연결이 끊긴 경우 등(2026-10-29 점검 L-5 - 시간 초과와 코드를 나눈다, 호출자는 상태 0만 본다).
            return 0, {"message": "connection", "code": "UPSTREAM_UNREACHABLE"}

    def search_hospitals(
        self,
        location: str = "",
        department: str = "",
        *,
        name: str | None = None,
        region: str | None = None,
        slot_date: str | None = None,
        operation_status: str | None = None,
        booking_available: bool | None = None,
        api_key_override: str | None = None,
    ) -> dict[str, Any]:
        reg = (region or location or "").strip()
        dept = (department or "").strip()
        _LOG.info(
            "[예약검색] region=%s department=%s name=%s slot_date=%s",  # 지역·병원명·날짜는 길이만(I-1)
            log_len(reg),
            dept or "(empty)",
            log_len((name or "").strip()),
            log_len(slot_date),
        )
        params: dict[str, str] = {}
        if name:
            params["name"] = name.strip()
        if reg:
            params["region"] = reg
        if dept:
            params["department"] = dept
        if operation_status:
            params["operationStatus"] = operation_status.strip()
        if booking_available is not None:
            params["bookingAvailable"] = str(booking_available).lower()
        auth_mode = "api_key" if self._resolve_api_key(api_key_override) else "none"
        st, raw = self._request("GET", "/hospitals", params=params, auth=auth_mode, token_override=api_key_override)
        rows: list[dict[str, Any]] = []
        raw_hospitals: list[dict[str, Any]] = []
        if st == 200 and isinstance(raw, list):
            raw_hospitals = [item for item in raw if isinstance(item, dict)]
        else:
            _LOG.warning("[예약검색실패] http_status=%s raw_type=%s", st, type(raw).__name__)

        if not raw_hospitals:
            st_all, raw_all = self._request("GET", "/hospitals", auth=auth_mode, token_override=api_key_override)
            if st_all == 200 and isinstance(raw_all, list):
                raw_hospitals = [item for item in raw_all if isinstance(item, dict)]

        for h in raw_hospitals:
            hid = h.get("id")
            hname = str(h.get("name", ""))
            reg_out = str(h.get("region", ""))
            address = str(h.get("address", ""))
            depts = h.get("departments") or []
            if not isinstance(depts, list) or not depts:
                depts = [""]

            if reg and _location_match_score(reg, reg_out) == 0 and _location_match_score(reg, address) == 0:
                continue
            if dept and not _department_matches(dept, depts, hname):
                continue

            slot_list: list[str] = []
            if slot_date and hid is not None:
                slot_params = {"date": slot_date}
                slot_dept = _slot_department(dept, depts)
                if slot_dept:
                    slot_params["department"] = slot_dept
                st2, slot_data = self._request(
                    "GET",
                    f"/hospitals/{hid}/available-slots",
                    params=slot_params,
                    auth=auth_mode,
                    token_override=api_key_override,
                )
                if st2 == 200 and isinstance(slot_data, dict):
                    for s in slot_data.get("availableSlots") or []:
                        slot_list.append(f"{slot_date} {s}".strip())
                    slot_list = _filter_future_slot_times(slot_date, slot_list)

            rows.append(
                {
                    "id": hid,
                    "name": hname,  # FE 카드가 읽는 키. 내부 로직은 아래 "hospital"을 쓴다.
                    "hospital": hname,
                    "department": " / ".join(str(dep).strip() for dep in depts if str(dep).strip()) or "—",
                    "location": reg_out,
                    "address": address,
                    "availableSlots": list(slot_list),
                }
            )

        rows.sort(
            key=lambda row: (
                -_location_match_score(reg, str(row.get("location", ""))) - _location_match_score(reg, str(row.get("address", ""))),
                str(row.get("hospital", "")),
            )
        )
        _LOG.info("[예약검색결과] region=%s department=%s rows=%s", log_len(reg), dept or "(empty)", len(rows))
        return {"success": True, "rows": rows}

    def resolve_hospital_id(self, name_query: str, *, api_key_override: str | None = None) -> int | None:
        return self.resolve_hospital(name_query, api_key_override=api_key_override)[0]

    def resolve_hospital(self, name_query: str, *, api_key_override: str | None = None) -> tuple[int | None, str | None]:
        """이름으로 병원 하나를 찾는다(정확 일치 > 부분 일치). 못 찾으면 (None, None) - 아무 병원이나 고르지 않는다."""
        query = (name_query or "").strip()
        if not query:
            return None, None
        auth_mode = "api_key" if self._resolve_api_key(api_key_override) else "none"
        st, raw = self._request("GET", "/hospitals", params={"name": query}, auth=auth_mode, token_override=api_key_override)
        rows: list[dict[str, Any]] = []
        if st == 200 and isinstance(raw, list):
            rows = [r for r in raw if isinstance(r, dict)]
        if not rows:
            st_all, raw_all = self._request("GET", "/hospitals", auth=auth_mode, token_override=api_key_override)
            if st_all == 200 and isinstance(raw_all, list):
                rows = [r for r in raw_all if isinstance(r, dict)]
            if not rows:
                return None, None
        needle = re.sub(r"\s+", "", query).casefold()
        exact_match: dict[str, Any] | None = None
        partial_match: dict[str, Any] | None = None
        for row in rows:
            name = re.sub(r"\s+", "", str(row.get("name", ""))).casefold()
            if not name:
                continue
            if name == needle:
                exact_match = row
                break
            if partial_match is None and needle in name:
                partial_match = row
        chosen = exact_match or partial_match
        if chosen is None:
            return None, None
        hid = chosen.get("id")
        return (int(hid) if isinstance(hid, int) else None), (str(chosen.get("name") or "").strip() or None)

    def list_departments(self, hospital_id: int, *, api_key_override: str | None = None) -> dict[str, Any]:
        auth_mode = "api_key" if self._resolve_api_key(api_key_override) else "none"
        st, data = self._request(
            "GET",
            f"/hospitals/{hospital_id}/departments",
            auth=auth_mode,
            token_override=api_key_override,
        )
        return {"httpStatus": st, "data": data}

    def get_available_slots(
        self, hospital_id: int, date: str, *, api_key_override: str | None = None, department: str | None = None
    ) -> dict[str, Any]:
        """빈 시간. 시간 칸은 진료과마다 따로라 진료과를 넘기면 그 진료과 기준이다(생략하면 병원 전체 기준)."""
        auth_mode = "api_key" if self._resolve_api_key(api_key_override) else "none"
        params = {"date": date}
        if department and department.strip():
            params["department"] = department.strip()
        st, data = self._request(
            "GET",
            f"/hospitals/{hospital_id}/available-slots",
            params=params,
            auth=auth_mode,
            token_override=api_key_override,
        )
        return {"httpStatus": st, "data": data}

    def create_reservation(
        self,
        hospital_id: int,
        department: str,
        reservation_date: str,
        reservation_time: str,
        patient_name: str,
        phone: str,
        *,
        api_key_override: str | None = None,
        birth_date: str | None = None,
        symptom_summary: str | None = None,
        memo: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "hospitalId": hospital_id,
            "department": department,
            "reservationDate": reservation_date,
            "reservationTime": reservation_time,
            "patientName": patient_name,
            "phone": phone,
        }
        if birth_date:
            body["birthDate"] = birth_date
        if symptom_summary:
            body["symptomSummary"] = symptom_summary
        if memo:
            body["memo"] = memo
        _LOG.info(
            "[예약등록] hospital_id=%s department=%s date=%s time=%s phone_last4=%s",  # 날짜·시간은 길이만(I-1)
            hospital_id,
            department or "(empty)",
            log_len(reservation_date),
            log_len(reservation_time),
            phone[-4:] if phone else "",
        )
        if self._resolve_api_key(api_key_override):
            st, data = self._request("POST", "/reservations", body=body, auth="api_key", token_override=api_key_override)
        else:
            st, data = self._request("POST", "/reservations", body=body, auth="user", token_override=None)
        _LOG.info("[예약등록결과] http_status=%s data_type=%s", st, type(data).__name__)
        return {"httpStatus": st, "data": data}

    def list_my_appointments(self, api_key_override: str | None = None) -> dict[str, Any]:
        if self._resolve_api_key(api_key_override):
            st, data = self._request("GET", "/reservations/my", auth="api_key", token_override=api_key_override)
        else:
            st, data = self._request("GET", "/reservations/my", auth="user", token_override=None)
        return {"httpStatus": st, "data": data}

    def get_reservation(self, reservation_id: int, api_key_override: str | None = None) -> dict[str, Any]:
        if self._resolve_api_key(api_key_override):
            st, data = self._request("GET", f"/reservations/{reservation_id}", auth="api_key", token_override=api_key_override)
        else:
            st, data = self._request("GET", f"/reservations/{reservation_id}", auth="user", token_override=None)
        return {"httpStatus": st, "data": data}

    def cancel_reservation(self, reservation_id: int, api_key_override: str | None = None) -> dict[str, Any]:
        """예약 취소. 예약 서비스가 키 소유자의 예약만 취소한다(남의 예약·없는 예약은 404, 취소 불가 상태는 400)."""
        path = f"/reservations/{int(reservation_id)}/cancel"
        if self._resolve_api_key(api_key_override):
            st, data = self._request("PATCH", path, auth="api_key", token_override=api_key_override)
        else:
            st, data = self._request("PATCH", path, auth="user", token_override=None)
        _LOG.info("[예약취소결과] http_status=%s", st)
        return {"httpStatus": st, "data": data}


@lru_cache
def get_reservation_api_client() -> ReservationApiClient:
    return ReservationApiClient(Settings())

