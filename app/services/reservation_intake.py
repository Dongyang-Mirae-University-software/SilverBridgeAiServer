from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

_PHONE_RE = re.compile(r"(?<!\d)(01\d[-\s]?\d{3,4}[-\s]?\d{4})(?!\d)")
_DEPARTMENT_ALIASES = {
    "소아과": "소아청소년과",
    "소아청소년과": "소아청소년과",
    "이비인후과": "이비인후과",
    "정신과": "정신건강의학과",
    "정신건강의학과": "정신건강의학과",
    "가정의학과": "가정의학과",
    "응급실": "응급의학과",
    "응급의학과": "응급의학과",
}


@dataclass
class ReservationDraft:
    message: str
    location: str | None = None
    hospital_name: str | None = None
    hospital_id: int | None = None
    department: str | None = None
    reservation_date: str | None = None
    reservation_time: str | None = None
    patient_name: str | None = None
    phone: str | None = None
    birth_date: str | None = None
    symptom_summary: str | None = None
    memo: str | None = None
    search_hint: str | None = None
    source_text: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    def required_gap(self) -> str | None:
        if not (self.department or "").strip():
            return "department"
        if self.hospital_id is None and not (self.hospital_name or "").strip():
            return "hospital"
        if not (self.reservation_date or "").strip():
            return "date"
        if not (self.reservation_time or "").strip():
            return "time"
        if not (self.patient_name or "").strip():
            return "patient_name"
        if not (self.phone or "").strip():
            return "phone"
        return None

    def to_make_appointment_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {
            "hospital": (self.hospital_name or "").strip(),
            "department": (self.department or "").strip(),
            "date": (self.reservation_date or "").strip(),
            "time": (self.reservation_time or "").strip(),
            "patient_name": (self.patient_name or "").strip(),
            "phone": (self.phone or "").strip(),
        }
        if self.hospital_id is not None:
            args["hospital_id"] = self.hospital_id
        if self.birth_date:
            args["birth_date"] = self.birth_date
        if self.symptom_summary:
            args["symptom_summary"] = self.symptom_summary
        if self.memo:
            args["memo"] = self.memo
        return args


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text or "")


def _context_text(context: dict[str, Any] | None, key: str) -> str | None:
    if not context:
        return None
    value = context.get(key)
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return None


def _context_phone(context: dict[str, Any] | None) -> str | None:
    value = _context_text(context, "phone")
    if not value:
        return None
    digits = _digits(value)
    if len(digits) == 10:
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    if len(digits) == 11:
        return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
    return value


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _trim_name_candidate(candidate: str) -> str | None:
    value = (candidate or "").strip()
    if not value:
        return None
    suffixes = ("입니다", "이에요", "예요", "이요", "이고요", "이고", "님", "씨", "분", "으로", "로")
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if value.endswith(suffix) and len(value) > len(suffix) + 1:
                value = value[: -len(suffix)].strip()
                changed = True
                break
    if 2 <= len(value) <= 4 and re.fullmatch(r"[가-힣]+", value):
        return value
    return None


def _find_department(text: str) -> str | None:
    raw = _normalized_text(text)
    for alias, normalized in _DEPARTMENT_ALIASES.items():
        if alias in raw:
            return normalized
    return None


def _find_hospital_name(text: str) -> str | None:
    raw = _normalized_text(text)
    patterns = (
        r"([가-힣A-Za-z0-9]+(?:병원|의원|클리닉|안과|내과|정형외과|소아과|소아청소년과|이비인후과|가정의학과|피부과|산부인과|신경과|정신건강의학과|외과))",
        r"([가-힣A-Za-z0-9]+(?:메디|의료원|센터))",
    )
    for pattern in patterns:
        m = re.search(pattern, raw)
        if m:
            return m.group(1).strip()
    return None


def _find_phone(text: str) -> str | None:
    m = _PHONE_RE.search(text or "")
    if not m:
        return None
    digits = _digits(m.group(1))
    if len(digits) == 10:
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"
    if len(digits) == 11:
        return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"
    return None


def _find_name(text: str) -> str | None:
    raw = _normalized_text(text)
    for marker in ("이름은", "환자명", "환자 이름", "예약자", "성함은", "성함", "본인 이름"):
        idx = raw.find(marker)
        if idx < 0:
            continue
        tail = raw[idx + len(marker) :].strip()
        tail = re.sub(r"^[=:는은이를로]*\s*", "", tail)
        m = re.match(r"^([가-힣]{2,8})", tail)
        if m:
            cleaned = _trim_name_candidate(m.group(1))
            if cleaned:
                return cleaned
    return None


def _find_date(text: str, base: date | None = None) -> str | None:
    raw = _normalized_text(text)
    base = base or date.today()
    m = re.search(r"\b(20\d{2})[./-](\d{1,2})[./-](\d{1,2})\b", raw)
    if m:
        try:
            parsed = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            return parsed.isoformat()
        except ValueError:
            return None
    relative_days = {"오늘": 0, "내일": 1, "모레": 2}
    for key, offset in relative_days.items():
        if key in raw:
            return (base + timedelta(days=offset)).isoformat()
    return None


def _find_time(text: str) -> str | None:
    raw = _normalized_text(text)
    m = re.search(r"([01]?\d|2[0-3]):([0-5]\d)", raw)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    if "오전" in raw or "오후" in raw:
        ampm = 12 if "오후" in raw else 0
        m = re.search(r"(1[0-2]|0?\d)시(?:\s*(반|30분))?", raw)
        if m:
            hour = int(m.group(1))
            minute = 30 if m.group(2) else 0
            if ampm == 12 and hour < 12:
                hour += 12
            return f"{hour:02d}:{minute:02d}"
    return None


def _extract_symptom_summary(text: str) -> str | None:
    raw = _normalized_text(text)
    if not raw or "예약" in raw or "접수" in raw:
        return None
    return raw[:120]


def build_reservation_draft(
    message: str,
    *,
    history_text: str = "",
    ui_field: str | None = None,
    ui_value: str | None = None,
    location: str | None = None,
    user_context: dict[str, Any] | None = None,
) -> ReservationDraft:
    merged = "\n".join(part for part in (message.strip(), history_text.strip()) if part).strip()
    ui_field = (ui_field or "").strip() or None
    ui_value = (ui_value or "").strip() or None

    draft = ReservationDraft(
        message=message or "",
        location=(location or "").strip() or None,
        source_text=merged,
    )

    if user_context:
        draft.patient_name = draft.patient_name or _context_text(user_context, "name")
        draft.phone = draft.phone or _context_phone(user_context)
        draft.birth_date = draft.birth_date or _context_text(user_context, "birthDate")
        profile_location = _context_text(user_context, "location") or _context_text(user_context, "address")
        if profile_location and not draft.location:
            draft.location = profile_location
        for key in ("addressDetail", "postcode", "gender", "email", "role"):
            val = _context_text(user_context, key)
            if val:
                draft.extras[key] = val

    if ui_field and ui_value:
        if ui_field == "hospital":
            draft.hospital_name = ui_value
        elif ui_field == "department":
            draft.department = _DEPARTMENT_ALIASES.get(ui_value, ui_value)
        elif ui_field == "date":
            draft.reservation_date = ui_value
        elif ui_field == "time":
            draft.reservation_time = ui_value
        elif ui_field == "patient_name":
            draft.patient_name = ui_value
        elif ui_field == "phone":
            draft.phone = ui_value
        elif ui_field == "birth_date":
            draft.birth_date = ui_value

    draft.department = draft.department or _find_department(merged)
    draft.hospital_name = draft.hospital_name or _find_hospital_name(merged)
    draft.reservation_date = draft.reservation_date or _find_date(merged)
    draft.reservation_time = draft.reservation_time or _find_time(merged)
    draft.patient_name = draft.patient_name or _find_name(merged)
    draft.phone = draft.phone or _find_phone(merged)
    draft.symptom_summary = _extract_symptom_summary(merged)

    if draft.hospital_name and not draft.search_hint:
        draft.search_hint = draft.hospital_name
    elif draft.department and location:
        draft.search_hint = f"{location} {draft.department}".strip()
    elif draft.department:
        draft.search_hint = draft.department
    elif location:
        draft.search_hint = location

    return draft

