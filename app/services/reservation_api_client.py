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

        req = urllib.request.Request(
            url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None,
            headers=headers,
            method=method.upper(),
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as response:
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
            "[예약검색] region=%s department=%s name=%s slot_date=%s",
            reg or "(empty)",
            dept or "(empty)",
            (name or "").strip() or "(empty)",
            slot_date or "(empty)",
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
                st2, slot_data = self._request(
                    "GET",
                    f"/hospitals/{hid}/available-slots",
                    params={"date": slot_date},
                    auth=auth_mode,
                    token_override=api_key_override,
                )
                if st2 == 200 and isinstance(slot_data, dict):
                    for s in slot_data.get("availableSlots") or []:
                        slot_list.append(f"{slot_date} {s}".strip())
                    slot_list = _filter_future_slot_times(slot_date, slot_list)

            rows.append(
                {
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
        _LOG.info("[예약검색결과] region=%s department=%s rows=%s", reg or "(empty)", dept or "(empty)", len(rows))
        return {"success": True, "rows": rows}

    def resolve_hospital_id(self, name_query: str, *, api_key_override: str | None = None) -> int | None:
        query = (name_query or "").strip()
        if not query:
            return None
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
                return None
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
        chosen = exact_match or partial_match or rows[0]
        hid = chosen.get("id")
        return int(hid) if isinstance(hid, int) else None

    def list_departments(self, hospital_id: int, *, api_key_override: str | None = None) -> dict[str, Any]:
        auth_mode = "api_key" if self._resolve_api_key(api_key_override) else "none"
        st, data = self._request(
            "GET",
            f"/hospitals/{hospital_id}/departments",
            auth=auth_mode,
            token_override=api_key_override,
        )
        return {"httpStatus": st, "data": data}

    def get_available_slots(self, hospital_id: int, date: str, *, api_key_override: str | None = None) -> dict[str, Any]:
        auth_mode = "api_key" if self._resolve_api_key(api_key_override) else "none"
        st, data = self._request(
            "GET",
            f"/hospitals/{hospital_id}/available-slots",
            params={"date": date},
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
            "[예약등록] hospital_id=%s department=%s date=%s time=%s phone_last4=%s",
            hospital_id,
            department or "(empty)",
            reservation_date,
            reservation_time,
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


@lru_cache
def get_reservation_api_client() -> ReservationApiClient:
    return ReservationApiClient(Settings())

