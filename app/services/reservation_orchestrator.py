from __future__ import annotations

import re
import logging
from functools import lru_cache
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from app.core.config import Settings, get_settings
from app.core.reservation_prompts import SYSTEM_PROMPT, build_user_prompt
from app.schemas.chat_schema import ChatRequest, UiOption, UiPayload
from app.services.gpt_structured_client import GptStructuredClient
from app.services.reservation_api_client import ReservationApiClient
from app.services.reservation_intake import ReservationDraft, build_reservation_draft
from app.utils.conversation import format_recent_history, normalize_history
from app.utils.json_extract import extract_json_object

_LOG = logging.getLogger(__name__)
_SEOUL_TZ = ZoneInfo("Asia/Seoul")


def _digits(phone: str) -> str:
    return re.sub(r"\D", "", phone or "")


def _phone_last4(phone: str | None) -> str:
    digits = _digits(phone or "")
    return digits[-4:] if digits else ""


def _trace(meta: dict[str, Any], event: str, **details: Any) -> None:
    trace = meta.setdefault("decisionTrace", [])
    if isinstance(trace, list):
        payload = {"event": event, **{k: v for k, v in details.items() if v is not None}}
        trace.append(payload)
        _LOG.info("[예약트레이스] %s %s", event, {k: v for k, v in details.items() if v is not None})


def _ui_for_field(field: str) -> UiPayload:
    if field == "date":
        return UiPayload(kind="date", field="date", label="달력에서 진료 날짜를 선택해 주세요")
    if field == "time":
        return UiPayload(kind="select", field="time", label="예약 시간을 선택해 주세요")
    return UiPayload(kind="text", field=field, label="아래에서 값을 선택하거나 입력해 주세요.")


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


class ReservationOrchestrator:
    RESERVATION_KEYWORDS = ("예약", "병원", "진료", "접수", "내 예약", "예약 내역", "예약 조회", "예약 목록")
    LOOKUP_KEYWORDS = ("내 예약", "예약 내역", "예약 조회", "예약 목록")
    SEARCH_KEYWORDS = ("병원 찾아", "근처 병원", "찾아줘", "찾아")

    def __init__(self, settings: Settings, loader: GptStructuredClient, reservation: ReservationApiClient) -> None:
        self._settings = settings
        self._loader = loader
        self._reservation = reservation

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
        if not req.context:
            return "(없음)"
        ctx = req.context.model_dump(exclude_none=True)
        keys = ("name", "phone", "gender", "birthDate", "postcode", "address", "addressDetail", "location", "email", "role")
        lines = []
        for key in keys:
            value = ctx.get(key)
            if isinstance(value, str) and value.strip():
                lines.append(f"- {key}: {value.strip()}")
            elif isinstance(value, int):
                lines.append(f"- {key}: {value}")
        return "\n".join(lines) if lines else "(없음)"

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
            "reservation_followup",
            "reservation_id",
        }

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
    ) -> dict[str, Any] | None:
        try:
            self._loader.ensure_loaded()
            prompt = build_user_prompt(
                message=req.message,
                history_text=history_text,
                state_json='{"facts":{},"intent":null,"history":[],"resolved":[]}',
                location=location,
                profile_text=self._user_profile_text(req),
            )
            raw = self._loader.generate_structured_text(SYSTEM_PROMPT, prompt)
        except Exception as exc:  # noqa: BLE001
            _trace(meta, "reservation_extraction_failed", error=str(exc))
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
            "symptom_summary",
            "memo",
        ):
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
                    draft.reservation_date = cleaned
                elif key == "reservation_time":
                    draft.reservation_time = cleaned
                elif key == "patient_name":
                    draft.patient_name = cleaned
                elif key == "phone":
                    draft.phone = cleaned
                elif key == "birth_date":
                    draft.birth_date = cleaned
                elif key == "symptom_summary":
                    draft.symptom_summary = cleaned
                elif key == "memo":
                    draft.memo = cleaned

        hid = extracted.get("hospital_id")
        if isinstance(hid, int) and hid > 0:
            draft.hospital_id = hid

    def _shape_make_appointment(self, raw: dict[str, Any], patient_name: str, phone_in: str) -> dict[str, Any]:
        rid = raw.get("reservationId")
        rdate = str(raw.get("reservationDate", ""))
        display_id = f"{rid}" if rid is None else str(rid)
        return {
            "reservationId": display_id,
            "reservation_id": rid,
            "hospital": raw.get("hospitalName"),
            "department": raw.get("department"),
            "date": raw.get("reservationDate"),
            "time": raw.get("reservationTime"),
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

    def handle(self, req: ChatRequest, meta: dict[str, Any]) -> dict[str, Any] | None:
        reservation_api_key = str(meta.get("reservationApiKey") or "").strip() or None
        ui_field = req.uiSelection.field if req.uiSelection else None
        ui_value = req.uiSelection.value if req.uiSelection else None
        message_text = (req.message or "").strip()
        intent = "hospital_reservation" if self._looks_like_reservation(message_text) else "general_chat"
        _trace(meta, "reservation_candidate", intent=intent, ui_field=ui_field)

        should_handle = intent == "hospital_reservation" or self._reservation_related_ui(ui_field)
        if not should_handle:
            return None

        if ui_field == "reservation_followup":
            if ui_value == "show_list":
                result = self._reservation.list_my_appointments(api_key_override=reservation_api_key)
                st = int(result.get("httpStatus") or 0)
                data = result.get("data")
                if 200 <= st < 300 and isinstance(data, list):
                    reservations = [item for item in data if isinstance(item, dict)]
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
                        "reply": "예약 목록을 불러왔습니다. 확인할 예약을 선택해 주세요.",
                        "riskLevel": "medium",
                        "recommendedAction": "guardian_contact",
                        "reservationRequired": True,
                        "intent": "hospital_reservation",
                        "engine": "gpt",
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
                    "engine": "gpt",
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
                "engine": "gpt",
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
                return {
                    "reply": "예약 상세를 불러왔습니다.",
                    "riskLevel": "medium",
                    "recommendedAction": "guardian_contact",
                    "reservationRequired": True,
                    "intent": "hospital_reservation",
                    "engine": "gpt",
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
                "engine": "gpt",
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

        if any(keyword in message_text for keyword in self.LOOKUP_KEYWORDS):
            result = self._reservation.list_my_appointments(api_key_override=reservation_api_key)
            st = int(result.get("httpStatus") or 0)
            data = result.get("data")
            if 200 <= st < 300 and isinstance(data, list):
                _trace(meta, "reservation_lookup_success", count=len(data))
                return {
                    "reply": "예약 내역을 불러왔습니다.",
                    "riskLevel": "medium",
                    "recommendedAction": "guardian_contact",
                    "reservationRequired": True,
                    "intent": "hospital_reservation",
                    "engine": "gpt",
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
        draft = build_reservation_draft(
            req.message,
            history_text=history_text,
            ui_field=ui_field,
            ui_value=ui_value,
            location=normalized_loc or loc,
            user_context=req.context.model_dump(exclude_none=True) if req.context else None,
        )
        extracted = self._extract_reservation_fields(req, history_text, normalized_loc or loc, meta)
        if extracted:
            self._apply_extracted(draft, extracted)

        if req.uiSelection:
            field = req.uiSelection.field
            value = req.uiSelection.value.strip()
            if field in {"hospital", "hospital_confirm"}:
                draft.hospital_name = value or draft.hospital_name
            elif field == "department":
                draft.department = value or draft.department
            elif field == "date":
                draft.reservation_date = value or draft.reservation_date
            elif field == "time":
                draft.reservation_time = value or draft.reservation_time
            elif field == "patient_name":
                draft.patient_name = value or draft.patient_name
            elif field == "phone":
                draft.phone = value or draft.phone

        _trace(
            meta,
            "reservation_draft",
            location=draft.location,
            hospital_name=draft.hospital_name,
            department=draft.department,
            reservation_date=draft.reservation_date,
            reservation_time=draft.reservation_time,
            patient_name=draft.patient_name,
            phone_last4=_phone_last4(draft.phone),
        )

        if draft.hospital_name and draft.hospital_id is None:
            draft.hospital_id = self._reservation.resolve_hospital_id(
                draft.hospital_name,
                api_key_override=reservation_api_key,
            )
            _trace(meta, "hospital_id_resolved", hospital_name=draft.hospital_name, hospital_id=draft.hospital_id)

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
                    "engine": "gpt",
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
                    "reply": "예약할 병원을 선택해 주세요.",
                    "riskLevel": "medium",
                    "recommendedAction": "guardian_contact",
                    "reservationRequired": True,
                    "intent": "hospital_reservation",
                    "engine": "gpt",
                    "modelName": self._loader.state.model_name,
                    "type": "tool_result",
                    "ui": None,
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
            draft.hospital_id = self._reservation.resolve_hospital_id(
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
                        "engine": "gpt",
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
                "engine": "gpt",
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
            slot_res = self._reservation.get_available_slots(
                draft.hospital_id,
                draft.reservation_date,
                api_key_override=reservation_api_key,
            )
            slot_st = int(slot_res.get("httpStatus") or 0)
            slot_data = slot_res.get("data")
            if slot_st == 200 and isinstance(slot_data, dict):
                slots = [str(s).strip() for s in slot_data.get("availableSlots") or [] if str(s).strip()]
                slots = _filter_future_times(draft.reservation_date, slots)
                _trace(meta, "slot_lookup", hospital_id=draft.hospital_id, slot_count=len(slots))
                if slots:
                    label = "예약 시간을 선택해 주세요."
                    context_label = self._reservation_context_label(draft)
                    if context_label:
                        label = f"{context_label} 예약 시간을 선택해 주세요."
                    return {
                        "reply": label,
                        "riskLevel": "medium",
                        "recommendedAction": "guardian_contact",
                        "reservationRequired": True,
                        "intent": "hospital_reservation",
                        "engine": "gpt",
                        "modelName": self._loader.state.model_name,
                        "type": "message",
                        "ui": UiPayload(
                            kind="select",
                            field="time",
                            label=label,
                            options=[UiOption(value=s, label=s) for s in slots],
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
                return {
                    "reply": "선택한 날짜에는 예약 가능한 시간이 없습니다. 다른 날짜를 선택해 주세요.",
                    "riskLevel": "medium",
                    "recommendedAction": "guardian_contact",
                    "reservationRequired": True,
                    "intent": "hospital_reservation",
                    "engine": "gpt",
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
                "reply": "아래에서 값을 선택하거나 입력해 주세요.",
                "riskLevel": "medium",
                "recommendedAction": "guardian_contact",
                "reservationRequired": True,
                "intent": "hospital_reservation",
                "engine": "gpt",
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

        _trace(
            meta,
            "reservation_submit",
            hospital_id=draft.hospital_id,
            hospital_name=draft.hospital_name,
            department=draft.department,
            reservation_date=draft.reservation_date,
            reservation_time=draft.reservation_time,
            patient_name=draft.patient_name,
            phone_last4=_phone_last4(draft.phone),
        )
        args = draft.to_make_appointment_args()
        if draft.hospital_id is not None:
            args["hospital_id"] = draft.hospital_id
        cr = self._reservation.create_reservation(
            hospital_id=int(args["hospital_id"]),
            department=str(args["department"]),
            reservation_date=str(args["date"]),
            reservation_time=str(args["time"]),
            patient_name=str(args["patient_name"]),
            phone=str(args["phone"]),
            api_key_override=reservation_api_key,
            birth_date=str(args["birth_date"]).strip() if args.get("birth_date") else None,
            symptom_summary=str(args["symptom_summary"]).strip() if args.get("symptom_summary") else None,
            memo=str(args["memo"]).strip() if args.get("memo") else None,
        )
        st = int(cr.get("httpStatus") or 0)
        data = cr.get("data")
        if 200 <= st < 300 and isinstance(data, dict):
            shaped = self._shape_make_appointment(data, str(args["patient_name"]), str(args["phone"]))
            _trace(meta, "reservation_submit_ok", reservation_id=shaped.get("reservation_id"), status=shaped.get("status"))
            return {
                "reply": "예약이 완료되었습니다. 예약 목록을 보여드릴까요?",
                "riskLevel": "medium",
                "recommendedAction": "guardian_contact",
                "reservationRequired": True,
                "intent": "hospital_reservation",
                "engine": "gpt",
                "modelName": self._loader.state.model_name,
                "type": "tool_result",
                "ui": UiPayload(
                    kind="select",
                    field="reservation_followup",
                    label="예약이 완료되었습니다. 예약 목록을 보여드릴까요?",
                    options=[
                        UiOption(value="show_list", label="예약 목록 보기"),
                        UiOption(value="stay", label="계속 상담하기"),
                    ],
                ),
                "tool": "make_appointment",
                "data": shaped,
                "summary": None,
                "possibleCauses": [],
                "homeCare": [],
                "visitHospitalIf": [],
                "emergencyWarning": [],
                "meta": meta,
                "decisionTrace": meta.get("decisionTrace") or [],
            }

        msg = "예약 처리에 실패했습니다."
        code = "REQUEST_FAILED"
        if isinstance(data, dict):
            msg = str(data.get("message") or msg)
            code = str(data.get("code") or code)
        _trace(meta, "reservation_submit_failed", http_status=st, code=code, message=msg[:200])
        return {
            "reply": msg,
            "riskLevel": "medium",
            "recommendedAction": "guardian_contact",
            "reservationRequired": True,
            "intent": "hospital_reservation",
            "engine": "gpt",
            "modelName": self._loader.state.model_name,
            "type": "tool_error",
            "ui": None,
            "tool": "make_appointment",
            "data": data,
            "error": {"code": code, "message": msg},
            "summary": None,
            "possibleCauses": [],
            "homeCare": [],
            "visitHospitalIf": [],
            "emergencyWarning": [],
            "meta": meta,
            "decisionTrace": meta.get("decisionTrace") or [],
        }


@lru_cache
def get_reservation_orchestrator() -> ReservationOrchestrator:
    settings = get_settings()
    loader = GptStructuredClient(settings)
    client = ReservationApiClient(settings)
    return ReservationOrchestrator(settings, loader, client)

