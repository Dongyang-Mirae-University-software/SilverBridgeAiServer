from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from app.services import chat_keywords
from app.utils.conversation import remove_personal_info

# 시간 선택지에 붙는 "다른 날짜 선택" 버튼 값
CHANGE_DATE_OPTION = "__change_date__"
_SEOUL_TZ = ZoneInfo("Asia/Seoul")

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
    "정형외과": "정형외과",
    "산부인과": "산부인과",
    "피부과": "피부과",
    "신경과": "신경과",
    "안과": "안과",
    "내과": "내과",
    "외과": "외과",
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

    def apply_time_selection(self, value: str) -> None:
        """시간 버튼 값 적용. 값은 "HH:mm" 또는 자동으로 넘긴 날짜를 담은 "YYYY-MM-DD HH:mm"이고, 날짜 변경 버튼이면 날짜·시간을 비운다."""
        if value == CHANGE_DATE_OPTION:
            self.reservation_date = None
            self.reservation_time = None
            return
        parts = value.split()
        if len(parts) == 2:
            self.reservation_date, self.reservation_time = parts
        else:
            self.reservation_time = value or self.reservation_time

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
    # 텍스트는 "현재 문장 + 이전 대화" 순이므로 가장 앞에 나온 진료과가 사용자의 최신 의도다.
    raw = _normalized_text(text)
    best: tuple[int, str] | None = None
    for alias, normalized in _DEPARTMENT_ALIASES.items():
        idx = raw.find(alias)
        if idx >= 0 and (best is None or idx < best[0]):
            best = (idx, normalized)
    return best[1] if best else None


def _find_hospital_name(text: str) -> str | None:
    raw = _normalized_text(text)
    patterns = (
        r"([가-힣A-Za-z0-9]+(?:병원|의원|클리닉|안과|내과|정형외과|소아과|소아청소년과|이비인후과|가정의학과|피부과|산부인과|신경과|정신건강의학과|외과))",
        r"([가-힣A-Za-z0-9]+(?:메디|의료원|센터))",
    )
    for pattern in patterns:
        for m in re.finditer(pattern, raw):
            name = m.group(1).strip()
            # "정형외과"처럼 진료과 이름만 있는 건 병원명이 아니다(정형+외과로 잘못 잡힌다).
            if name not in _DEPARTMENT_ALIASES:
                return name
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
    # "오늘·내일"은 한국 날짜 기준이다. 서버(컨테이너)는 UTC라 date.today()를 쓰면 KST 00~09시에 하루 밀린다.
    base = base or datetime.now(_SEOUL_TZ).date()
    # 끝은 \b 가 아니라 (?!\d): 한글도 단어 문자라 "2026-10-30에"에서 \b 가 경계를 못 찾는다.
    m = re.search(r"(?<!\d)(20\d{2})[./-](\d{1,2})[./-](\d{1,2})(?!\d)", raw)
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


# 병원에 넘기는 증상 요약(symptomSummary) 규칙(2026-10-28 S-8). 이번 예약 흐름에서 사용자가 직접 친 문장 중 증상 단어가 든 문장만 쓴다 -
# 대화 이력·LLM 추출 값은 쓰지 않는다(이력은 클라이언트가 보내는 값이고, 관련 없는 대화가 예약 기록에 실렸다).
SYMPTOM_SUMMARY_MAX_CHARS = 60
_SYMPTOM_WORDS = chat_keywords.SUMMARY_SYMPTOM_WORDS
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。])\s+|\n+")


def _extract_symptom_summary(text: str) -> str | None:
    """사용자 문장에서 증상 단어가 든 문장만 모아 개인정보 형태(전화·주민번호·이메일)를 지우고 60자에서 자른다. 없으면 None."""
    kept: list[str] = []
    for sentence in _SENTENCE_SPLIT_RE.split(text or ""):
        cleaned = re.sub(r"\s+([.,!?])", r"\1", _normalized_text(remove_personal_info(sentence)))
        if cleaned and any(word in cleaned for word in _SYMPTOM_WORDS):
            kept.append(cleaned)
    summary = " ".join(kept).strip()
    return summary[:SYMPTOM_SUMMARY_MAX_CHARS].strip() or None


def merge_symptom_summary(current: str | None, text: str) -> str | None:
    """이어 쓰는 초안의 증상 요약에 이번 턴 문장의 증상을 덧붙인다(60자 상한). 이번 턴에 증상이 없으면 그대로."""
    added = _extract_symptom_summary(text)
    if not added:
        return current
    if not current:
        return added
    if added in current:
        return current
    return f"{current} {added}"[:SYMPTOM_SUMMARY_MAX_CHARS].strip()


def build_reservation_draft(
    message: str,
    *,
    history_text: str = "",
    ui_field: str | None = None,
    ui_value: str | None = None,
    location: str | None = None,
    user_context: dict[str, Any] | None = None,
    symptom_text: str | None = None,
) -> ReservationDraft:
    """symptom_text = 사용자가 이번 턴에 직접 친 문장(라우터 규격화 문장 아님). 증상 요약은 이것만으로 만든다 - 이력은 쓰지 않는다."""
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
            draft.apply_time_selection(ui_value)
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
    draft.symptom_summary = _extract_symptom_summary(message if symptom_text is None else symptom_text)

    if draft.hospital_name and not draft.search_hint:
        draft.search_hint = draft.hospital_name
    elif draft.department and location:
        draft.search_hint = f"{location} {draft.department}".strip()
    elif draft.department:
        draft.search_hint = draft.department
    elif location:
        draft.search_hint = location

    return draft

