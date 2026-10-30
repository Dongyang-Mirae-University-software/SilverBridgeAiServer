from __future__ import annotations

import json
import re
import logging
from functools import lru_cache
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from app.core.config import Settings, get_settings
from app.services import chat_keywords
from app.core.reservation_prompts import EXTRACTION_SCHEMA, SYSTEM_PROMPT, build_user_prompt
from app.schemas.chat_schema import ChatRequest, UiOption, UiPayload
from app.services.claude_structured_client import ClaudeStructuredClient, failure_cause
from app.services.reservation_api_client import ReservationApiClient
from app.services.reservation_common import _date_label, _digits, _phone_last4, _trace, with_demo_notice
from app.services.reservation_confirm_flow import (
    CONFIRM_FIELDS,
    DEPARTMENT_NOT_OFFERED_FLAG,
    ReservationFlowMixin,
    draft_from_dict,
    fresh_flow_state,
)
from app.services.reservation_intake import (
    CHANGE_DATE_OPTION,
    ReservationDraft,
    _find_date,
    _find_department,
    _find_hospital_name,
    _find_time,
    build_reservation_draft,
    merge_symptom_summary,
)
from app.utils.conversation import format_recent_history, mask_personal_info, normalize_history
from app.utils.json_extract import extract_json_object

_LOG = logging.getLogger(__name__)
_SEOUL_TZ = ZoneInfo("Asia/Seoul")


def _is_past_date(value: str) -> bool:
    try:
        parsed = datetime.strptime(value.strip()[:10], "%Y-%m-%d").date()
    except ValueError:
        return False
    return parsed < datetime.now(_SEOUL_TZ).date()


# 빠진 값을 물을 때 쓰는 문구·예시. 무엇을 넣어야 하는지 바로 알 수 있어야 한다.
_GAP_PROMPTS: dict[str, tuple[str, str | None]] = {
    "patient_name": ("예약하실 분의 성함을 입력해 주세요.", "예: 김영희"),
    "phone": ("연락받을 전화번호를 입력해 주세요.", "예: 010-1234-5678"),
    "department": ("진료과를 알려주세요.", "예: 내과"),
    "hospital": ("병원 이름을 알려주세요.", "예: 강남메디내과"),
    "location": ("지역을 알려주세요.", "예: 서울 강남구"),
}



# 예약 추출(Claude) 입력 줄이기(2026-10-28 S-6·O-3).
EXTRACT_HISTORY_TURNS = 6
EXTRACT_TURN_MAX_CHARS = 300
# 추출 호출이 남겨 둘 시간(초) - 그 뒤 병원 검색·빈 시간 조회 몫.
EXTRACT_RESERVE_SEC = 15.0
_SIDO_SHORT = {
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종", "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
}
_REGION_UNIT_RE = re.compile(r"[가-힣]+(특별시|광역시|특별자치시|특별자치도|도|시|군|구)")


def coarse_region(text: str) -> str | None:
    """주소에서 시도·시군구까지만 남긴다("서울특별시 강남구 테헤란로 1" -> "서울특별시 강남구", "경기 성남시 분당구 …" -> 세 칸).
    동·도로명·번지·상세주소는 버린다. 알아볼 수 없으면 None(보내지 않는다)."""
    tokens = re.sub(r"\s+", " ", (text or "").strip()).split(" ")
    kept: list[str] = []
    for index, token in enumerate(tokens[:3]):
        is_unit = token in _SIDO_SHORT or bool(_REGION_UNIT_RE.fullmatch(token))
        # 셋째 칸은 "OO시 OO구"(일반구)일 때만 - "강남역입구" 같은 낱말이 붙지 않게.
        if not is_unit or (index == 2 and not (kept[-1].endswith("시") and token.endswith("구"))):
            break
        kept.append(token)
    return " ".join(kept) or None


_symptom_department = chat_keywords.symptom_department


class _DepartmentNotOffered(Exception):
    """빈 시간 조회가 400 INVALID_DEPARTMENT - 그 병원에 이 진료과가 없다(관리자가 이름을 바꿨거나 사용 중지). 진료과를 다시 묻는다."""



def _linked_fields(draft: ReservationDraft) -> tuple[str | None, ...]:
    """이어 쓰는 초안에서 바뀌면 다른 칸을 낡게 만드는 값(병원 id, 병원 이름, 진료과, 날짜, 시간)."""
    return (
        None if draft.hospital_id is None else str(draft.hospital_id),
        draft.hospital_name, draft.department, draft.reservation_date, draft.reservation_time,
    )


def _clear_stale_links(before: tuple[str | None, ...], draft: ReservationDraft, meta: dict[str, Any]) -> None:
    """저장된 초안을 이어 쓸 때 이번 턴에 바뀐 값에 엮인 칸을 비운다(R-8).
    진료과가 바뀌었는데 병원이 그대로면 그 병원에 새 진료과가 없을 수 있어 병원을 비우고,
    병원 이름이 바뀌었는데 id 가 그대로면 옛 병원으로 예약되고, 병원·진료과·날짜가 바뀌었는데 시간이 그대로면
    그 시간이 비어 있는지 확인하지 않은 채 확인 카드로 간다. 이번 턴에 함께 정해진 값(날짜를 담은 시간 버튼 등)은 지우지 않는다."""
    old_id, old_hospital, old_dept, old_date, old_time = before
    new_id, new_hospital, new_dept, new_date, new_time = _linked_fields(draft)
    cleared: list[str] = []
    # 진료과를 바꿨는데 병원이 그대로면 그 병원에 새 진료과가 없을 수 있다 - 고치기 흐름처럼 병원을 다시 찾는다(점검 M-1).
    # 병원은 정했는데 진료과가 비어 있다 처음 정해지는 경우(old_dept 없음)는 바뀐 것이 아니다.
    if old_dept and new_dept != old_dept and new_hospital == old_hospital and draft.hospital_name:
        draft.hospital_name = None
        draft.hospital_id = None
        cleared.append("hospital")
        new_hospital, new_id = None, None
    if new_hospital != old_hospital and new_id == old_id and draft.hospital_id is not None:
        draft.hospital_id = None
        cleared.append("hospital_id")
    if (new_hospital != old_hospital or new_dept != old_dept or new_date != old_date) and new_time == old_time and draft.reservation_time:
        draft.reservation_time = None
        cleared.append("reservation_time")
    if cleared:
        _trace(meta, "reservation_draft_links_cleared", cleared=cleared)


def _with_copula(word: str) -> str:
    """'내과예요' / '병원이에요' - 받침 유무로 고른다."""
    code = ord(word[-1]) - 0xAC00
    has_final = 0 <= code < 11172 and code % 28 != 0
    return f"{word}{'이에요' if has_final else '예요'}"


def _gap_label(field: str) -> str:
    return _GAP_PROMPTS.get(field, ("아래에서 값을 선택하거나 입력해 주세요.", None))[0]


def _ui_for_field(field: str) -> UiPayload:
    if field == "date":
        return UiPayload(kind="date", field="date", label="달력에서 진료 날짜를 선택해 주세요")
    if field == "time":
        return UiPayload(kind="select", field="time", label="예약 시간을 선택해 주세요")
    label, placeholder = _GAP_PROMPTS.get(field, ("아래에서 값을 선택하거나 입력해 주세요.", None))
    return UiPayload(kind="text", field=field, label=label, placeholder=placeholder)


def _filter_future_times(slot_date: str | None, slots: list[str]) -> list[str]:
    date_text = (slot_date or "").strip()
    if not date_text:
        return slots

    try:
        target_date = datetime.strptime(date_text, "%Y-%m-%d").date()
    except ValueError:
        return slots

    today = datetime.now(_SEOUL_TZ).date()
    if target_date != today:
        return slots

    now_time = datetime.now(_SEOUL_TZ).time().replace(second=0, microsecond=0)
    filtered: list[str] = []
    for slot in slots:
        text = str(slot).strip()
        if not text:
            continue
        time_text = text.split()[-1]
        try:
            slot_time = datetime.strptime(time_text, "%H:%M").time()
        except ValueError:
            continue
        if slot_time > now_time:
            filtered.append(text)
    return filtered


class ReservationOrchestrator(ReservationFlowMixin):
    RESERVATION_KEYWORDS = chat_keywords.RESERVATION_CONTEXT_WORDS
    LOOKUP_KEYWORDS = ("내 예약", "예약 내역", "예약 조회", "예약 목록")
    SEARCH_KEYWORDS = ("병원 찾아", "근처 병원", "찾아줘", "찾아")
    # 선택한 날짜가 꽉 찼을 때 다음 빈 날을 찾는 범위
    SLOT_LOOKAHEAD_DAYS = 7

    def __init__(self, settings: Settings, loader: ClaudeStructuredClient, reservation: ReservationApiClient) -> None:
        self._settings = settings
        self._loader = loader
        self._reservation = reservation

    def _booked_times(self, api_key: str | None) -> dict[str, set[str]]:
        """사용자가 이미 잡아 둔 날짜별 시간. 예약 서비스는 같은 사람의 같은 날짜·시간 예약을 병원과 무관하게 거절하므로
        선택지에서 미리 뺀다. 조회 실패면 빈 값(fail-open)."""
        try:
            res = self._reservation.list_my_appointments(api_key_override=api_key)
            rows = res.get("data") if int(res.get("httpStatus") or 0) == 200 else None
        except Exception:  # noqa: BLE001
            rows = None
        booked: dict[str, set[str]] = {}
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict) or str(row.get("status") or "").upper() in {"CANCELED", "CANCELLED", "REJECTED"}:
                continue
            day = str(row.get("reservationDate") or row.get("date") or "")[:10]
            tm = str(row.get("reservationTime") or row.get("time") or "").strip()
            if day and tm:
                booked.setdefault(day, set()).add(tm)
        return booked

    def _slots_on(
        self,
        hospital_id: int,
        date_text: str,
        api_key: str | None,
        booked: dict[str, set[str]] | None = None,
        department: str | None = None,
    ) -> list[str] | None:
        """해당 날짜의 빈 시간(오늘이면 지난 시간 제외, 내가 이미 잡은 시간 제외). 조회 실패면 None.
        시간 칸은 진료과마다 따로라 진료과를 알면 함께 넘긴다(모르면 예약 서비스가 병원 전체 기준으로 센다)."""
        res = self._reservation.get_available_slots(hospital_id, date_text, api_key_override=api_key, department=department)
        data = res.get("data")
        status = int(res.get("httpStatus") or 0)
        if status == 400 and isinstance(data, dict) and data.get("code") == "INVALID_DEPARTMENT":
            raise _DepartmentNotOffered()
        if status != 200 or not isinstance(data, dict):
            return None
        slots = [str(s).strip() for s in data.get("availableSlots") or [] if str(s).strip()]
        taken = (booked or {}).get(date_text, set())
        return [s for s in _filter_future_times(date_text, slots) if s.split()[-1] not in taken]

    def _find_slots(
        self,
        hospital_id: int,
        requested_date: str,
        api_key: str | None,
        meta: dict[str, Any],
        department: str | None = None,
    ) -> tuple[str | None, list[str]]:
        """요청 날짜에 빈 시간이 없으면 다음 날부터 SLOT_LOOKAHEAD_DAYS일 안에서 처음 비는 날을 찾는다.

        반환 (날짜, 슬롯). 요청 날짜 조회 자체가 실패하면 (None, [])로 기존 흐름(시간 직접 입력)에 맡긴다."""
        booked = self._booked_times(api_key)
        try:
            slots = self._slots_on(hospital_id, requested_date, api_key, booked, department)
        except _DepartmentNotOffered:
            _trace(meta, "slot_department_not_offered", hospital_id=hospital_id, department=department)
            meta[DEPARTMENT_NOT_OFFERED_FLAG] = True
            return None, []
        _trace(meta, "slot_lookup", hospital_id=hospital_id, slot_count=None if slots is None else len(slots))
        if slots is None:
            return None, []
        if slots:
            return requested_date, slots
        try:
            base = datetime.strptime(requested_date, "%Y-%m-%d").date()
        except ValueError:
            return requested_date, []
        for offset in range(1, self.SLOT_LOOKAHEAD_DAYS + 1):
            candidate = (base + timedelta(days=offset)).isoformat()
            found = self._slots_on(hospital_id, candidate, api_key, booked, department) or []
            if found:
                _trace(meta, "slot_lookahead", requested=requested_date, found=candidate)
                return candidate, found
        return requested_date, []

    def _time_select_reply(
        self, meta: dict[str, Any], draft: ReservationDraft, requested_date: str, slot_date: str, slots: list[str], prefix: str = ""
    ) -> dict[str, Any]:
        """시간 선택지 응답. 요청 날짜와 다른 날로 넘겼으면 그 사실을, prefix가 있으면 그 사유(중복 등)를 앞에 붙인다."""
        label = "예약 시간을 선택해 주세요."
        context_label = self._reservation_context_label(draft)
        if context_label:
            label = f"{context_label} 예약 시간을 선택해 주세요."
        if slot_date != requested_date:
            label = (
                f"{_date_label(requested_date)}은 예약 가능한 시간이 없어 "
                f"{_date_label(slot_date)} 예약 가능 시간을 보여드립니다. {label}"
            )
        if prefix:
            label = f"{prefix} {label}"
        return {
            "reply": label,
            "riskLevel": "medium",
            "recommendedAction": "guardian_contact",
            "reservationRequired": True,
            "intent": "hospital_reservation",
            "engine": "claude",
            "modelName": self._loader.state.model_name,
            "type": "message",
            "ui": UiPayload(
                kind="select",
                field="time",
                label="예약 시간을 선택해 주세요.",
                options=[UiOption(value=f"{slot_date} {s}", label=s) for s in slots]
                + [UiOption(value=CHANGE_DATE_OPTION, label="다른 날짜 선택")],
            ),
            "tool": None,
            "data": None,
            "summary": None,
            "possibleCauses": [],
            "homeCare": [],
            "visitHospitalIf": [],
            "emergencyWarning": [],
            "meta": meta,
            "decisionTrace": meta.get("decisionTrace") or [],
        }

    @staticmethod
    def _normalize_location_hint(location: str | None) -> str | None:
        raw = (location or "").strip()
        if not raw:
            return None
        if re.search(r"[가-힣]", raw):
            return raw
        compact = re.sub(r"[^a-z]", "", raw.lower())
        mapping = {
            "seoul": "서울",
            "busan": "부산",
            "incheon": "인천",
            "daegu": "대구",
            "daejeon": "대전",
            "gwangju": "광주",
            "ulsan": "울산",
            "jeju": "제주",
        }
        return mapping.get(compact)

    def _conversation_context(self, req: ChatRequest) -> tuple[str, str]:
        recent, older_summary = normalize_history(
            history=[h.model_dump() for h in req.history],
            keep_turns=self._settings.chat_history_keep_turns,
        )
        history_text = format_recent_history(recent)
        if older_summary:
            history_text = f"[이전 요약]\n{older_summary}\n\n[최근 대화]\n{history_text}"

        loc = ""
        if req.context and req.context.location:
            loc = str(req.context.location).strip()
        elif req.context and req.context.address:
            loc = str(req.context.address).strip()
        return history_text, loc

    @staticmethod
    def _user_profile_text(req: ChatRequest) -> str:
        """예약 추출(Claude)에 보내는 프로필 - 지역(시·구까지)만 보낸다(2026-10-28 S-6).
        이름·전화·생년월일은 서버가 프로필로 초안을 채우고(피보호자면 _pin_ward_patient 로 고정) 이메일·성별·상세주소는 추출에 쓰지 않는다."""
        if not req.context:
            return "(없음)"
        region = coarse_region(str(req.context.location or req.context.address or ""))
        return f"- region: {region}" if region else "(없음)"

    def _claude_history_text(self, req: ChatRequest) -> str:
        """예약 추출(Claude)에 보내는 이력 - 최근 EXTRACT_HISTORY_TURNS턴, 턴당 EXTRACT_TURN_MAX_CHARS자, 그 이전 요약은 빼고
        전화·주민번호·이메일 형태는 가린다(S-6). 규칙 초안(서버 안)은 예전처럼 _conversation_context 의 이력을 쓴다."""
        recent = [h.model_dump() for h in req.history][-EXTRACT_HISTORY_TURNS:]
        trimmed = [{"role": h.get("role") or "", "content": (h.get("content") or "")[:EXTRACT_TURN_MAX_CHARS]} for h in recent]
        return mask_personal_info(format_recent_history(trimmed))

    NEW_REQUEST_CUES = ("근처", "가까운", "주변", "아파", "아프", "통증", "증상")
    REQUEST_VERBS = ("예약", "잡아", "찾아", "알아봐", "접수")

    @classmethod
    def _is_new_request(cls, message: str) -> bool:
        """요청 동사(예약·잡아·찾아…)와 함께 진료과·병원명·"근처"·증상 중 하나를 말하면 새 요청으로 본다."""
        if not message or not any(verb in message for verb in cls.REQUEST_VERBS):
            return False
        return bool(
            _find_department(message)
            or _find_hospital_name(message)
            or any(cue in message for cue in cls.NEW_REQUEST_CUES)
        )

    @staticmethod
    def _looks_like_reservation(message: str) -> bool:
        text = (message or "").strip()
        return bool(text) and any(keyword in text for keyword in ReservationOrchestrator.RESERVATION_KEYWORDS)

    @staticmethod
    def _reservation_related_ui(field: str | None) -> bool:
        return field in {
            "hospital",
            "hospital_confirm",
            "department",
            "date",
            "time",
            "patient_name",
            "phone",
            "location",
            "location_confirm",
            "reservation_followup",
            "reservation_id",
        } or field in CONFIRM_FIELDS

    STANDARD_DEPARTMENTS = (
        "내과", "이비인후과", "정형외과", "안과", "피부과", "가정의학과",
        "소아청소년과", "산부인과", "신경과", "정신건강의학과", "외과", "응급의학과",
    )

    def _ui_reply(self, meta: dict[str, Any], reply: str, ui: UiPayload) -> dict[str, Any]:
        return {
            "reply": reply,
            "riskLevel": "medium",
            "recommendedAction": "guardian_contact",
            "reservationRequired": True,
            "intent": "hospital_reservation",
            "engine": "claude",
            "modelName": self._loader.state.model_name,
            "type": "ui",
            "ui": ui,
            "tool": None,
            "data": None,
            "summary": None,
            "possibleCauses": [],
            "homeCare": [],
            "visitHospitalIf": [],
            "emergencyWarning": [],
            "meta": meta,
            "decisionTrace": meta.get("decisionTrace") or [],
        }

    def _other_region_ui(self, req: ChatRequest, searched_location: str | None) -> UiPayload | None:
        """프로필 주소로 찾은 결과에만 "다른 지역으로 찾기"를 붙인다(사용자가 직접 말한 지역이면 이미 원하는 곳이다)."""
        profile = {
            re.sub(r"\s+", " ", str(value).strip())
            for value in (req.context.location if req.context else None, req.context.address if req.context else None)
            if value and str(value).strip()
        }
        _, loc = self._conversation_context(req)
        hint = self._normalize_location_hint(loc)
        profile |= {v for v in (hint,) if v}
        if not searched_location:
            return None
        searched = re.sub(r"\s+", " ", searched_location.strip())
        # 추출이 지역을 시·구로 줄여 돌려줘도 프로필 지역과 같은 곳으로 본다(2026-10-29 점검 L-3).
        # 다만 시·구보다 자세한 지역(동 등)은 사용자가 직접 말한 것이라 버튼을 붙이지 않는다(2026-10-29 재점검 L-2).
        coarse_profile = {coarse_region(v) for v in profile} - {None}
        is_profile_coarse = coarse_region(searched) == searched and searched in coarse_profile
        if searched not in profile and not is_profile_coarse:
            return None
        return UiPayload(
            kind="select",
            field="location_confirm",
            label="다른 지역에서 찾으시려면 눌러 주세요",
            options=[UiOption(value="__other__", label="다른 지역으로 찾기")],
        )

    def _ask_location(self, meta: dict[str, Any], department: str | None = None) -> dict[str, Any]:
        _trace(meta, "reservation_gap", missing="location")
        # "병원 어디로 가야 돼요?"에는 먼저 어느 과인지 답하고, 가까운 곳을 찾으려 지역을 묻는다(예약 양식처럼 보이지 않게).
        reply = (
            f"{department}로 가시면 돼요. 가까운 {department}를 찾아드릴게요. 어느 지역에서 찾을까요?"
            if department
            else "어느 지역에서 병원을 찾을까요?"
        )
        return self._ui_reply(
            meta,
            reply,
            UiPayload(kind="text", field="location", label="지역을 입력해 주세요", placeholder="예: 서울 강남구"),
        )

    @staticmethod
    def _location_candidates(req: ChatRequest, draft_location: str | None) -> list[str]:
        raw_candidates: list[str] = []
        if draft_location:
            raw_candidates.append(draft_location)
        if req.context:
            if req.context.location:
                raw_candidates.append(str(req.context.location))
            if req.context.address:
                raw_candidates.append(str(req.context.address))

        seen: set[str] = set()
        candidates: list[str] = []
        for raw in raw_candidates:
            normalized = re.sub(r"\s+", " ", raw.strip())
            if not normalized:
                continue
            tokens = normalized.split(" ")
            for end in range(len(tokens), 0, -1):
                candidate = " ".join(tokens[:end]).strip()
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    candidates.append(candidate)

            if len(tokens) >= 2:
                first = re.sub(r"(특별시|광역시|자치시|특별자치시|특별자치도|광역자치시|도)$", "", tokens[0]).strip()
                second = tokens[1].strip()
                if first and second:
                    alias = f"{first} {second}".strip()
                    if alias and alias not in seen:
                        seen.add(alias)
                        candidates.append(alias)
                if second and second not in seen:
                    seen.add(second)
                    candidates.append(second)

            cleaned = re.sub(r"(특별시|광역시|자치시|특별자치시|특별자치도|광역자치시|도)$", "", normalized.replace(" ", "")).strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                candidates.append(cleaned)
        return candidates

    def _extract_reservation_fields(
        self,
        req: ChatRequest,
        history_text: str,
        location: str,
        meta: dict[str, Any],
        draft_state: ReservationDraft | None = None,
    ) -> dict[str, Any] | None:
        # 저장 초안을 이어 쓰는 턴은 최근 대화 대신 그 초안을 [현재 상태 JSON]으로 준다 - 이력 속 지난 요청의 병원·시간이
        # 추출로 되살아나지 않게 한다(R-8 남은 것). 처음부터 다시 만드는 턴은 예전처럼 이력을 본다.
        state_json = '{"facts":{},"intent":null,"history":[],"resolved":[]}'
        if draft_state is not None:
            facts = {
                k: v for k, v in {
                    "department": draft_state.department, "hospital_name": draft_state.hospital_name,
                    # 초안의 지역은 검색용 전체 값이라 Claude 에는 시·구까지만 보낸다(2026-10-28 점검 M-1).
                    "location": coarse_region(draft_state.location or ""), "reservation_date": draft_state.reservation_date,
                    "reservation_time": draft_state.reservation_time,
                }.items() if v
            }
            state_json = json.dumps({"facts": facts, "intent": "hospital_reservation", "history": [], "resolved": list(facts)}, ensure_ascii=False)
            history_text = "(생략 - 이 예약에서 지금까지 정해진 값은 [현재 상태 JSON]에 있다)"
        try:
            self._loader.ensure_loaded()
            prompt = build_user_prompt(
                message=req.message,
                history_text=history_text if draft_state is not None else (self._claude_history_text(req) or "(없음)"),
                state_json=state_json,
                location=coarse_region(location) or "",
                profile_text=self._user_profile_text(req),
            )
            raw = self._loader.generate_structured_text(
                SYSTEM_PROMPT, prompt, schema=EXTRACTION_SCHEMA, stage="reservation_extract", reserve_sec=EXTRACT_RESERVE_SEC
            )
        except Exception as exc:  # noqa: BLE001
            _trace(meta, "reservation_extraction_failed", error=failure_cause(exc))  # 고정 코드만 - 예외 문구에는 입력·모델 출력이 섞일 수 있다
            return None

        parsed = extract_json_object(raw)
        if not isinstance(parsed, dict):
            _trace(meta, "reservation_extraction_parse_failed")
            return None
        _trace(
            meta,
            "reservation_extraction_ok",
            intent=parsed.get("intent"),
            missing_fields=parsed.get("missing_fields"),
            confidence=parsed.get("confidence"),
            needs_search=parsed.get("needs_search"),
        )
        return parsed

    @staticmethod
    def _apply_extracted(draft: ReservationDraft, extracted: dict[str, Any]) -> None:
        for key in (
            "hospital_name",
            "department",
            "location",
            "reservation_date",
            "reservation_time",
            "patient_name",
            "phone",
            "birth_date",
            "memo",
        ):
            # symptom_summary 는 받지 않는다 - 추출 입력에 이력이 섞여 관련 없는 대화가 예약 기록에 실릴 수 있다(S-8).
            value = extracted.get(key)
            if isinstance(value, str) and value.strip():
                cleaned = value.strip()
                if key == "location":
                    draft.location = cleaned
                elif key == "hospital_name":
                    draft.hospital_name = cleaned
                elif key == "department":
                    draft.department = cleaned
                elif key == "reservation_date":
                    # LLM이 과거 날짜(예: 학습 시점 기준 "내일")를 내면 버리고 규칙 파서 값을 유지한다.
                    if _is_past_date(cleaned):
                        continue
                    draft.reservation_date = cleaned
                elif key == "reservation_time":
                    draft.reservation_time = cleaned
                elif key == "patient_name":
                    draft.patient_name = cleaned
                elif key == "phone":
                    draft.phone = cleaned
                elif key == "birth_date":
                    draft.birth_date = cleaned
                elif key == "memo":
                    draft.memo = cleaned

        hid = extracted.get("hospital_id")
        if isinstance(hid, int) and hid > 0:
            draft.hospital_id = hid

    @staticmethod
    def _shape_reservation_row(raw: dict[str, Any]) -> dict[str, Any]:
        """예약 서비스 행(hospitalName·reservationDate…)에 화면 카드가 읽는 키(hospital_name·date·time)를 덧붙인다."""
        rid = raw.get("reservationId")
        return {
            **raw,
            "reservationId": str(rid) if rid is not None else None,
            "reservation_id": rid,
            "hospital": raw.get("hospitalName"),
            "hospital_name": raw.get("hospitalName"),
            "date": raw.get("reservationDate"),
            "time": raw.get("reservationTime"),
        }

    def _shape_make_appointment(self, raw: dict[str, Any], patient_name: str, phone_in: str) -> dict[str, Any]:
        return {
            **self._shape_reservation_row(raw),
            "patient_name": patient_name.strip(),
            "phone": _digits(phone_in),
            "status": raw.get("status"),
            "message": raw.get("message"),
        }

    @staticmethod
    def _reservation_context_label(draft: ReservationDraft) -> str:
        parts = [
            str(draft.hospital_name or "").strip(),
            str(draft.department or "").strip(),
            str(draft.reservation_date or "").strip(),
            str(draft.reservation_time or "").strip(),
        ]
        readable = [part for part in parts if part]
        return " · ".join(readable) if readable else ""

    @staticmethod
    def _reservation_choice_label(item: dict[str, Any]) -> str:
        hospital = str(item.get("hospitalName") or item.get("hospital") or "병원").strip()
        date = str(item.get("reservationDate") or item.get("date") or "").strip()
        time = str(item.get("reservationTime") or item.get("time") or "").strip()
        status = str(item.get("status") or "").strip()
        parts = [part for part in (hospital, date, time, status) if part]
        return " · ".join(parts) if parts else "예약 항목"

    def _apply_ui_selection(self, draft: ReservationDraft, field: str, value: str) -> bool:
        """선택값을 초안에 반영한다. 지역을 다시 물어야 하면(__other__) True."""
        if field in {"hospital", "hospital_confirm"}:
            draft.hospital_name = value or draft.hospital_name
        elif field == "department":
            draft.department = value or draft.department
        elif field == "date":
            draft.reservation_date = value or draft.reservation_date
        elif field == "time":
            draft.apply_time_selection(value)
        elif field == "patient_name":
            draft.patient_name = value or draft.patient_name
        elif field == "phone":
            draft.phone = value or draft.phone
        elif field == "location":
            draft.location = value or draft.location
        elif field == "location_confirm":
            if value == "__other__":
                return True
            draft.location = value or draft.location
        return False

    def handle(self, req: ChatRequest, meta: dict[str, Any]) -> dict[str, Any] | None:
        reservation_api_key = str(meta.get("reservationApiKey") or "").strip() or None
        ui_field = req.uiSelection.field if req.uiSelection else None
        ui_value = req.uiSelection.value if req.uiSelection else None
        message_text = (req.message or "").strip()
        # 라우터가 규격화한 문장이 아니라 사용자가 실제로 친 문장. 새 요청·조회 판단은 이걸로 한다.
        raw_message = str(meta.get("rawMessage") or message_text).strip()
        intent = "hospital_reservation" if self._looks_like_reservation(message_text) else "general_chat"
        _trace(meta, "reservation_candidate", intent=intent, ui_field=ui_field)

        wants_cancel = not req.uiSelection and self._is_cancel_request(raw_message)
        wants_change = not req.uiSelection and not wants_cancel and self._is_change_request(raw_message)
        asks_how = not req.uiSelection and self._is_cancel_change_question(raw_message)
        should_handle = (
            intent == "hospital_reservation"
            or self._reservation_related_ui(ui_field)
            or wants_cancel
            or wants_change
            or asks_how
        )
        if not should_handle:
            return None
        if asks_how:
            return self._flow_message(meta, self.HOW_TO_REPLY)

        # 확인·취소·변경 버튼은 서버가 저장한 직전 상태(토큰)와 맞을 때만 동작한다. 예약 생성·취소 호출은 여기서만 일어난다.
        if ui_field in CONFIRM_FIELDS:
            return self._handle_flow_selection(req, meta, ui_field, ui_value or "", reservation_api_key)
        if wants_cancel:
            return self._start_pick(meta, reservation_api_key, change=False)
        if wants_change:
            return self._start_pick(meta, reservation_api_key, change=True)

        if ui_field == "reservation_followup":
            if ui_value == "show_list":
                result = self._reservation.list_my_appointments(api_key_override=reservation_api_key)
                st = int(result.get("httpStatus") or 0)
                data = result.get("data")
                if 200 <= st < 300 and isinstance(data, list):
                    reservations = [self._shape_reservation_row(item) for item in data if isinstance(item, dict)]
                    data = reservations
                    options = []
                    for item in reservations:
                        reservation_id = item.get("reservationId") or item.get("id")
                        if reservation_id is None:
                            continue
                        options.append(
                            UiOption(
                                value=str(reservation_id),
                                label=self._reservation_choice_label(item),
                            )
                        )
                    ui = None
                    if options:
                        ui = UiPayload(
                            kind="select",
                            field="reservation_id",
                            label="상세를 볼 예약을 선택해 주세요",
                            options=options,
                        )
                    return {
                        "reply": with_demo_notice("예약 목록을 불러왔습니다. 확인할 예약을 선택해 주세요."),
                        "riskLevel": "medium",
                        "recommendedAction": "guardian_contact",
                        "reservationRequired": True,
                        "intent": "hospital_reservation",
                        "engine": "claude",
                        "modelName": self._loader.state.model_name,
                        "type": "tool_result",
                        "ui": ui,
                        "tool": "list_my_appointments",
                        "data": data,
                        "summary": None,
                        "possibleCauses": [],
                        "homeCare": [],
                        "visitHospitalIf": [],
                        "emergencyWarning": [],
                        "meta": meta,
                        "decisionTrace": meta.get("decisionTrace") or [],
                    }
                _trace(meta, "reservation_lookup_failed", status=st)
                return {
                    "reply": "예약 목록을 불러오지 못했습니다.",
                    "riskLevel": "medium",
                    "recommendedAction": "guardian_contact",
                    "reservationRequired": True,
                    "intent": "hospital_reservation",
                    "engine": "claude",
                    "modelName": self._loader.state.model_name,
                    "type": "tool_error",
                    "ui": None,
                    "tool": "list_my_appointments",
                    "data": data,
                    "error": {"code": "LOOKUP_FAILED", "message": "예약 목록을 불러오지 못했습니다."},
                    "summary": None,
                    "possibleCauses": [],
                    "homeCare": [],
                    "visitHospitalIf": [],
                    "emergencyWarning": [],
                    "meta": meta,
                    "decisionTrace": meta.get("decisionTrace") or [],
                }

            return {
                "reply": "알겠습니다. 다른 예약이나 증상도 말씀해 주세요.",
                "riskLevel": "medium",
                "recommendedAction": "guardian_contact",
                "reservationRequired": True,
                "intent": "hospital_reservation",
                "engine": "claude",
                "modelName": self._loader.state.model_name,
                "type": "message",
                "ui": None,
                "tool": None,
                "data": None,
                "summary": None,
                "possibleCauses": [],
                "homeCare": [],
                "visitHospitalIf": [],
                "emergencyWarning": [],
                "meta": meta,
                "decisionTrace": meta.get("decisionTrace") or [],
            }

        if ui_field == "reservation_id":
            reservation_id = 0
            try:
                reservation_id = int(str(ui_value).strip())
            except ValueError:
                reservation_id = 0
            if reservation_id <= 0:
                return None
            result = self._reservation.get_reservation(reservation_id, api_key_override=reservation_api_key)
            st = int(result.get("httpStatus") or 0)
            data = result.get("data")
            if 200 <= st < 300 and isinstance(data, dict):
                data = self._shape_reservation_row(data)
                return {
                    "reply": with_demo_notice("예약 상세를 불러왔습니다."),
                    "riskLevel": "medium",
                    "recommendedAction": "guardian_contact",
                    "reservationRequired": True,
                    "intent": "hospital_reservation",
                    "engine": "claude",
                    "modelName": self._loader.state.model_name,
                    "type": "tool_result",
                    "ui": None,
                    "tool": "get_reservation",
                    "data": data,
                    "summary": None,
                    "possibleCauses": [],
                    "homeCare": [],
                    "visitHospitalIf": [],
                    "emergencyWarning": [],
                    "meta": meta,
                    "decisionTrace": meta.get("decisionTrace") or [],
                }
            return {
                "reply": "예약 상세를 불러오지 못했습니다.",
                "riskLevel": "medium",
                "recommendedAction": "guardian_contact",
                "reservationRequired": True,
                "intent": "hospital_reservation",
                "engine": "claude",
                "modelName": self._loader.state.model_name,
                "type": "tool_error",
                "ui": None,
                "tool": "get_reservation",
                "data": data,
                "error": {"code": "LOOKUP_FAILED", "message": "예약 상세를 불러오지 못했습니다."},
                "summary": None,
                "possibleCauses": [],
                "homeCare": [],
                "visitHospitalIf": [],
                "emergencyWarning": [],
                "meta": meta,
                "decisionTrace": meta.get("decisionTrace") or [],
            }

        if any(keyword in raw_message for keyword in self.LOOKUP_KEYWORDS):
            result = self._reservation.list_my_appointments(api_key_override=reservation_api_key)
            st = int(result.get("httpStatus") or 0)
            data = result.get("data")
            if 200 <= st < 300 and isinstance(data, list):
                data = [self._shape_reservation_row(item) for item in data if isinstance(item, dict)]
                _trace(meta, "reservation_lookup_success", count=len(data))
                return {
                    "reply": with_demo_notice("예약 내역을 불러왔습니다."),
                    "riskLevel": "medium",
                    "recommendedAction": "guardian_contact",
                    "reservationRequired": True,
                    "intent": "hospital_reservation",
                    "engine": "claude",
                    "modelName": self._loader.state.model_name,
                    "type": "tool_result",
                    "ui": None,
                    "tool": "list_my_appointments",
                    "data": data,
                    "summary": None,
                    "possibleCauses": [],
                    "homeCare": [],
                    "visitHospitalIf": [],
                    "emergencyWarning": [],
                    "meta": meta,
                    "decisionTrace": meta.get("decisionTrace") or [],
                }
            _trace(meta, "reservation_lookup_failed", status=st)

        history_text, loc = self._conversation_context(req)
        normalized_loc = self._normalize_location_hint(loc)

        # 직전 턴이 남긴 초안이 있으면 이력에서 다시 만들지 않고 이어 쓴다(옛 시간·병원이 이력에서 되살아나는 것 방지).
        # 일반 예약 흐름도 매 턴 저장한다(R-8). 이력 재구성은 저장 초안이 없거나(옛 기록·다른 대화가 낌) TTL 이 지났을 때의 폴백이다.
        flow = (meta.get("storedFlow") or {}).get("flowState") if isinstance(meta.get("storedFlow"), dict) else None
        in_flow = (
            fresh_flow_state(flow)
            # 다른 피보호자로 바뀌었으면 이어 쓰지 않고 새로 만든다. wardId 키가 없는 옛 상태(배포 전)도 이어 쓰지 않는다.
            and "wardId" in flow
            and (flow.get("wardId") or "") == self._ward_id(req)
            and bool(req.uiSelection or not self._is_new_request(raw_message))
        )
        replaces: dict[str, Any] | None = None
        before: tuple[str | None, ...] | None = None
        if in_flow:
            draft = draft_from_dict(flow["draft"])  # type: ignore[index]
            before = _linked_fields(draft)
            replaces = flow.get("replaces") if isinstance(flow.get("replaces"), dict) else None  # type: ignore[union-attr]
            if raw_message and not req.uiSelection:
                extracted = self._extract_reservation_fields(req, history_text, normalized_loc or loc, meta, draft_state=draft)
                if extracted:
                    self._apply_extracted(draft, extracted)
                draft.reservation_date = _find_date(raw_message) or draft.reservation_date
                draft.reservation_time = _find_time(raw_message) or draft.reservation_time
                draft.symptom_summary = merge_symptom_summary(draft.symptom_summary, raw_message)
        else:
            draft = build_reservation_draft(
                req.message,
                history_text=history_text,
                ui_field=ui_field,
                ui_value=ui_value,
                location=normalized_loc or loc,
                user_context=req.context.model_dump(exclude_none=True) if req.context else None,
                # 증상 요약은 이번 턴에 사용자가 직접 친 문장으로만 만든다(버튼 턴은 없음, S-8).
                symptom_text="" if req.uiSelection else raw_message,
            )
            extracted = self._extract_reservation_fields(req, history_text, normalized_loc or loc, meta)
            if extracted:
                self._apply_extracted(draft, extracted)

            # 새 예약 요청(버튼이 아닌 문장에 진료과·병원·"근처"·증상이 있음)이면 이전 턴에서 고른 병원·시간은 버린다.
            # 초안은 매 턴 이력 전체에서 다시 만들어지므로, 안 버리면 지난 요청의 병원·시간이 새 요청에 그대로 붙는다.
            if not req.uiSelection and self._is_new_request(raw_message):
                if not _find_hospital_name(raw_message):
                    draft.hospital_name = None
                    draft.hospital_id = None
                if not _find_time(raw_message):
                    draft.reservation_time = None
                # 날짜도 이어받지 않는다: 지난 요청의 날짜(이력의 "오늘은 …" 같은 안내 문구 포함)가 새 요청에 조용히 붙는 것을 막는다.
                # 이번 문장에서 직접 말한 날짜(규칙 파서 또는 LLM 추출)만 남기고, 없으면 날짜를 되묻는다.
                extracted_date = str((extracted or {}).get("reservation_date") or "").strip()
                draft.reservation_date = _find_date(raw_message) or (
                    extracted_date if extracted_date and not _is_past_date(extracted_date) else None
                )
                dept_in_message = _find_department(raw_message) or _symptom_department(raw_message)
                if dept_in_message:
                    draft.department = dept_in_message
                _trace(meta, "new_request_reset", hospital=draft.hospital_name, time=draft.reservation_time, date=draft.reservation_date, department=draft.department)

        if not in_flow:
            # 저장 초안 없이 다시 만든 초안: 이 상담에서 버튼으로 입력했던 환자 이름·전화로 빈 칸만 채운다(2026-10-29 점검 L-2).
            # 대화 이력에는 넣지 않는 값이라(Claude로 나가지 않게) chat_service 가 서버 기록에서 꺼내 넘긴다. 프로필·이번 문장 값이 먼저다.
            contact = meta.get("storedContact") if isinstance(meta.get("storedContact"), dict) else {}
            draft.patient_name = draft.patient_name or (str(contact.get("patient_name") or "").strip() or None)
            draft.phone = draft.phone or (str(contact.get("phone") or "").strip() or None)
        ward_id = self._ward_id(req)
        if req.uiSelection and self._apply_ui_selection(draft, req.uiSelection.field, req.uiSelection.value.strip()):
            return self._remember_flow(self._ask_location(meta, draft.department), meta, draft, replaces, ward_id)
        if before is not None:
            _clear_stale_links(before, draft, meta)

        _trace(
            meta,
            "reservation_draft",
            location=draft.location,
            hospital_name=draft.hospital_name,
            department=draft.department,
            reservation_date=draft.reservation_date,
            reservation_time=draft.reservation_time,
            patient_name_len=len(draft.patient_name or ""),
            phone_last4=_phone_last4(draft.phone),
        )

        out = self._advance(req, meta, draft, reservation_api_key, replaces)
        return self._remember_flow(out, meta, draft, replaces, ward_id)

    def _advance(
        self,
        req: ChatRequest,
        meta: dict[str, Any],
        draft: ReservationDraft,
        reservation_api_key: str | None,
        replaces: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """초안에서 비어 있는 것을 순서대로 채우고(병원 → 날짜 → 시간 → 환자 정보), 다 채워지면 예약한다."""
        self._pin_ward_patient(req, draft)
        if draft.hospital_name and draft.hospital_id is None:
            draft.hospital_id, resolved_name = self._reservation.resolve_hospital(
                draft.hospital_name,
                api_key_override=reservation_api_key,
            )
            if draft.hospital_id is None:
                # 이름이 안 맞으면 병원 없이 검색으로 넘긴다("정형외과 강남"처럼 과만 말한 경우). 엉뚱한 병원을 잡지 않는다.
                draft.hospital_name = None
            else:
                draft.hospital_name = resolved_name or draft.hospital_name  # 사용자가 줄여 말해도 화면엔 실제 병원명
            _trace(meta, "hospital_id_resolved", hospital_name=draft.hospital_name, hospital_id=draft.hospital_id)

        if draft.hospital_id is None and not draft.hospital_name:
            if not draft.department:
                _trace(meta, "reservation_gap", missing="department")
                return self._ui_reply(
                    meta,
                    "어느 진료과를 찾을까요?",
                    UiPayload(
                        kind="select",
                        field="department",
                        label="진료과를 선택해 주세요",
                        options=[UiOption(value=d, label=d) for d in self.STANDARD_DEPARTMENTS],
                    ),
                )
            if not draft.location:
                return self._ask_location(meta, draft.department)
            # 프로필 지역은 되묻지 않고 바로 쓴다(한 턴 절약). 어느 지역으로 찾았는지는 결과 문구에 적고,
            # 다른 지역을 원하면 사용자가 말한 지역이 현재 문장 우선 규칙으로 덮어쓴다.

        if draft.hospital_id is None and not draft.hospital_name and (draft.department or draft.location):
            unique_rows: list[dict[str, Any]] = []
            tried_regions: list[str] = []
            for region in self._location_candidates(req, draft.location):
                tried_regions.append(region)
                _trace(meta, "hospital_search_attempt", region=region, department=draft.department, date=draft.reservation_date)
                cr = self._reservation.search_hospitals(
                    region,
                    draft.department or "",
                    name=None,
                    region=region,
                    slot_date=draft.reservation_date,
                    api_key_override=reservation_api_key,
                )
                if not cr.get("success") or not isinstance(cr.get("rows"), list):
                    _trace(meta, "hospital_search_failed", region=region, reason="upstream_error")
                    continue
                rows = [r for r in cr.get("rows") or [] if isinstance(r, dict)]
                unique_rows = []
                seen: set[str] = set()
                for row in rows:
                    name = str(row.get("hospital") or "").strip()
                    if name and name not in seen:
                        seen.add(name)
                        unique_rows.append(row)
                _trace(meta, "hospital_search_result", region=region, row_count=len(rows), unique_count=len(unique_rows))
                if unique_rows:
                    break
            if not unique_rows:
                _trace(meta, "hospital_search_exhausted", tried_regions=tried_regions, department=draft.department)
                return {
                    "reply": "조건에 맞는 병원을 찾지 못했습니다. 지역이나 진료과를 조금 더 자세히 알려주세요.",
                    "riskLevel": "medium",
                    "recommendedAction": "guardian_contact",
                    "reservationRequired": True,
                    "intent": "hospital_reservation",
                    "engine": "claude",
                    "modelName": self._loader.state.model_name,
                    "type": "message",
                    "ui": None,
                    "tool": None,
                    "data": None,
                    "summary": None,
                    "possibleCauses": [],
                    "homeCare": [],
                    "visitHospitalIf": [],
                    "emergencyWarning": [],
                    "meta": meta,
                    "decisionTrace": meta.get("decisionTrace") or [],
                }
            if len(unique_rows) > 1:
                _trace(meta, "hospital_selection_needed", candidate_count=len(unique_rows))
                return {
                    "reply": with_demo_notice(f"{region} 기준으로 찾은 {_with_copula(draft.department or '병원')}. 예약하려면 병원을 선택해 주세요."),
                    "riskLevel": "medium",
                    "recommendedAction": "guardian_contact",
                    "reservationRequired": True,
                    "intent": "hospital_reservation",
                    "engine": "claude",
                    "modelName": self._loader.state.model_name,
                    "type": "tool_result",
                    "ui": self._other_region_ui(req, draft.location),
                    "tool": "search_hospital",
                    "data": unique_rows,
                    "summary": None,
                    "possibleCauses": [],
                    "homeCare": [],
                    "visitHospitalIf": [],
                    "emergencyWarning": [],
                    "meta": {**meta, "reservationCandidates": unique_rows},
                    "decisionTrace": meta.get("decisionTrace") or [],
                }
            draft.hospital_name = str(unique_rows[0].get("hospital") or "").strip() or draft.hospital_name
            draft.hospital_id, _ = self._reservation.resolve_hospital(
                draft.hospital_name or "",
                api_key_override=reservation_api_key,
            )

        if draft.hospital_id is not None and not draft.department:
            dept_res = self._reservation.list_departments(draft.hospital_id, api_key_override=reservation_api_key)
            dept_st = int(dept_res.get("httpStatus") or 0)
            dept_data = dept_res.get("data")
            if dept_st == 200 and isinstance(dept_data, dict):
                departments_raw = dept_data.get("departments") or []
                departments = [
                    str(item.get("name") or "").strip()
                    for item in departments_raw
                    if isinstance(item, dict) and str(item.get("name") or "").strip()
                ]
                if departments:
                    _trace(meta, "department_selection_needed", hospital_id=draft.hospital_id, count=len(departments))
                    return {
                        "reply": "진료과를 선택해 주세요.",
                        "riskLevel": "medium",
                        "recommendedAction": "guardian_contact",
                        "reservationRequired": True,
                        "intent": "hospital_reservation",
                        "engine": "claude",
                        "modelName": self._loader.state.model_name,
                        "type": "message",
                        "ui": UiPayload(
                            kind="select",
                            field="department",
                            label=f"{draft.hospital_name}의 진료과를 선택해 주세요" if draft.hospital_name else "진료과를 선택해 주세요",
                            options=[UiOption(value=d, label=d) for d in departments],
                        ),
                        "tool": None,
                        "data": None,
                        "summary": None,
                        "possibleCauses": [],
                        "homeCare": [],
                        "visitHospitalIf": [],
                        "emergencyWarning": [],
                        "meta": meta,
                        "decisionTrace": meta.get("decisionTrace") or [],
                    }

        if not draft.reservation_date:
            _trace(meta, "reservation_gap", missing="date")
            label = "달력에서 진료 날짜를 선택해 주세요."
            context_label = self._reservation_context_label(draft)
            if context_label:
                label = f"{context_label} 예약 날짜를 선택해 주세요."
            return {
                "reply": label,
                "riskLevel": "medium",
                "recommendedAction": "guardian_contact",
                "reservationRequired": True,
                "intent": "hospital_reservation",
                "engine": "claude",
                "modelName": self._loader.state.model_name,
                "type": "message",
                "ui": UiPayload(
                    kind="date",
                    field="date",
                    label=label,
                    placeholder=context_label or None,
                ),
                "tool": None,
                "data": None,
                "summary": None,
                "possibleCauses": [],
                "homeCare": [],
                "visitHospitalIf": [],
                "emergencyWarning": [],
                "meta": meta,
                "decisionTrace": meta.get("decisionTrace") or [],
            }

        if draft.hospital_id is not None and not draft.reservation_time:
            requested_date = draft.reservation_date
            slot_date, slots = self._find_slots(draft.hospital_id, requested_date, reservation_api_key, meta, draft.department)
            if meta.pop(DEPARTMENT_NOT_OFFERED_FLAG, False):
                # 그 병원에 없는 진료과 - 시간 직접 입력으로 넘기지 않고 병원의 진료과 목록에서 다시 고르게 한다(점검 L-1)
                draft.department = None
                draft.reservation_time = None
                return self._advance(req, meta, draft, reservation_api_key, replaces)
            if slot_date is not None:
                # 자동으로 넘긴 날짜는 서버에 저장되지 않으므로 시간 버튼 값("YYYY-MM-DD HH:mm")에 담아 다음 턴에 받는다.
                draft.reservation_date = slot_date
                if slots:
                    return self._time_select_reply(meta, draft, requested_date, slot_date, slots)
                return {
                    "reply": f"{_date_label(requested_date)}부터 {self.SLOT_LOOKAHEAD_DAYS}일 안에는 예약 가능한 시간이 없습니다. 다른 날짜를 선택해 주세요.",
                    "riskLevel": "medium",
                    "recommendedAction": "guardian_contact",
                    "reservationRequired": True,
                    "intent": "hospital_reservation",
                    "engine": "claude",
                    "modelName": self._loader.state.model_name,
                    "type": "message",
                    "ui": _ui_for_field("date"),
                    "tool": None,
                    "data": None,
                    "summary": None,
                    "possibleCauses": [],
                    "homeCare": [],
                    "visitHospitalIf": [],
                    "emergencyWarning": [],
                    "meta": meta,
                    "decisionTrace": meta.get("decisionTrace") or [],
                }

        gap = draft.required_gap()
        if gap:
            _trace(meta, "reservation_gap", missing=gap)
            return {
                "reply": _gap_label(gap),
                "riskLevel": "medium",
                "recommendedAction": "guardian_contact",
                "reservationRequired": True,
                "intent": "hospital_reservation",
                "engine": "claude",
                "modelName": self._loader.state.model_name,
                "type": "message",
                "ui": _ui_for_field(gap),
                "tool": None,
                "data": None,
                "summary": None,
                "possibleCauses": [],
                "homeCare": [],
                "visitHospitalIf": [],
                "emergencyWarning": [],
                "meta": {**meta, "validatorDetail": gap},
                "decisionTrace": meta.get("decisionTrace") or [],
            }

        return self._confirm_reply(req, meta, draft, replaces)


@lru_cache
def get_reservation_orchestrator() -> ReservationOrchestrator:
    settings = get_settings()
    loader = ClaudeStructuredClient(settings)
    client = ReservationApiClient(settings)
    return ReservationOrchestrator(settings, loader, client)

