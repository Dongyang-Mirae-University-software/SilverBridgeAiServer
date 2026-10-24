from __future__ import annotations

import json
import re
import logging
import time
from uuid import uuid4
from typing import Any

from sqlalchemy.orm import Session  # pyright: ignore[reportMissingImports]

from app.models.chat_log import ChatLog
from app.core.config import get_settings
from app.schemas.chat_schema import ChatRequest
from app.services.reservation_api_client import get_reservation_api_client
from app.services.reservation_credential_service import get_reservation_credential_service
from app.services.chat_router_service import get_chat_router_service
from app.services.claude_structured_client import failure_cause
from app.services.medical_llm_service import MedicalLlmService, get_medical_llm_service
from app.services.reservation_confirm_flow import CONFIRM_FIELDS
from app.services.reservation_intake import _find_department
from app.services.reservation_orchestrator import ReservationOrchestrator, get_reservation_orchestrator

_LOG = logging.getLogger(__name__)


def _elapsed_ms(since: float) -> int:
    return int((time.perf_counter() - since) * 1000)


def _origin(exc: BaseException) -> str:
    """예외가 난 마지막 프레임의 파일:줄. 예외 문구(값이 섞일 수 있음) 없이 원인 위치만 남긴다."""
    tb = exc.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    if tb is None:
        return "?"
    return f"{tb.tb_frame.f_code.co_filename.rsplit('/', 1)[-1]}:{tb.tb_lineno}"


# 오케스트레이터에 넘긴 입력값이 응답 meta 로 되돌아온다. 챗 기록(context_json)에는 남기지 않는다:
# 예약 API 키(자격 증명), 직전 턴 상태 사본(storedFlow - 같은 기록이 이미 직전 행에 있음), 사용자 원문(message 컬럼에 있음).
_UNSTORED_META_KEYS = ("reservationApiKey", "storedFlow", "rawMessage")


def _persistable_meta(meta: Any) -> dict:
    if not isinstance(meta, dict):
        return {}
    return {k: v for k, v in meta.items() if k not in _UNSTORED_META_KEYS}


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in value]
    return value


class ChatService:
    """
    의료 챗 상담 서비스.
    - 1순위: MedGemma LLM (대화 이력 반영)
    - 2순위: 키워드 기반 fallback (모델 미로드/오류 시)
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    EMERGENCY_KEYWORDS = MedicalLlmService.EMERGENCY_KEYWORDS  # 단일 출처
    HIGH_RISK_KEYWORDS = ("가슴", "호흡곤란", "숨이", "의식", "실신", "출혈", "마비")
    MEDIUM_RISK_KEYWORDS = (
        "어지럽", "어지러", "어지럼", "복통", "발열", "기침", "두통", "구토", "아파", "아프", "통증", "더부룩", "속이", "소화",
        "무릎", "허리", "관절", "저리", "붓", "당뇨", "혈압", "혈당", "불면", "잠이", "수면", "약을", "약 먹", "복용",
        "설사", "변비", "가려", "피부", "기운", "피곤", "식욕",
    )
    RESERVATION_KEYWORDS = ("예약", "병원 찾아", "근처 병원", "진료 가능한", "응급실", "잡아", "접수", "병원 좀", "병원 추천", "진료 받")

    def process_message(self, db: Session, payload: ChatRequest) -> dict:
        started = time.perf_counter()
        # 구간 시간(ms). 거치지 않은 구간은 None. 챗 기록 upstreamMeta.timing 과 [CHAT-TIMING] 로그에만 남는다(응답 필드 아님).
        timing: dict[str, int | None] = {"routerMs": None, "llmMs": None, "reservationMs": None}
        message = (payload.message or "").strip()
        ui_selection = None
        if payload.uiSelection:
            ui_selection = {
                "field": payload.uiSelection.field.strip(),
                "value": payload.uiSelection.value.strip(),
            }
        if not message and not ui_selection:
            raise ValueError("메시지를 입력해 주세요.")

        session_id = (payload.sessionId or "").strip() or f"chat-{uuid4().hex[:12]}"
        user_context = self._build_user_context(payload)
        ward_id = (payload.context.wardId or "").strip() if payload.context and payload.context.wardId else ""
        history = [{"role": item.role, "content": item.content} for item in payload.history]
        log_message = message
        if not log_message and ui_selection:
            log_message = f"[선택] {ui_selection['field']}: {ui_selection['value']}"

        try:
            reservation_credential_service = get_reservation_credential_service()
            reservation_api_client = get_reservation_api_client()
            reservation_ui_fields = {
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
            } | CONFIRM_FIELDS
            # 1차 감별: GPT 라우터. 실패하면 키워드 분류로 내려간다.
            route_info: dict = {}
            # 자해·자살 신호는 진료과 답변·예약·MedGemma보다 먼저 고정 응답으로 보낸다(키워드 판정, 라우터 결과와 무관).
            self_harm_keyword = bool(message) and bool(MedicalLlmService.self_harm_signals(message))
            department_answer = self._department_question_reply(message, history) if message and not self_harm_keyword else None
            if department_answer:
                result = department_answer
            elif self_harm_keyword:
                # 위기 문장은 라우터(외부 API)로 보내지 않는다 - 라우터 결과와 상관없이 고정 응답이 나간다.
                route_info = {"route": "symptom", "engine": "rule", "normalized_message": message, "symptom_summary": "",
                              "emergency": True, "reason": "self-harm-keyword"}
            elif message:
                router_started = time.perf_counter()
                try:
                    route_info = get_chat_router_service().route(
                        message, history, user_context,
                        # 새 예약 요청이면 직전 턴의 병원·시간을 끌어오지 않는다(지난 요청이 새 요청에 붙는 것 방지).
                        prev_reservation=None
                        if ReservationOrchestrator._is_new_request(message)
                        else self._last_reservation_state(db, payload.userId, session_id, ward_id),
                    )
                except Exception as exc:  # noqa: BLE001
                    # 원인은 고정 코드만(max_tokens·parse·api_<상태>…) - 입력·모델 출력은 남기지 않는다.
                    _LOG.warning("[CHAT-ROUTER-FAILED] cause=%s fallback=keywords", failure_cause(exc))
                timing["routerMs"] = _elapsed_ms(router_started)
                # "머리가 아파서 병원 예약하고 싶어요"처럼 증상과 예약이 섞이면 라우터가 증상 상담으로 보내기도 한다 - 예약 요청이 분명하면 예약으로 고정.
                if (
                    route_info
                    and route_info.get("route") != "reservation"
                    and not route_info.get("emergency")
                    and ReservationOrchestrator._is_new_request(message)
                ):
                    route_info = {**route_info, "route": "reservation", "normalized_message": message, "reason": "new-request-override"}
                # "예약 취소할래"·"예약 바꿔줘"도 라우터가 다른 경로로 보내면 예약 흐름으로 고정한다(응급 신호가 있으면 그대로 상담).
                if (
                    route_info
                    and route_info.get("route") != "reservation"
                    and not route_info.get("emergency")
                    and (
                        ReservationOrchestrator._is_cancel_request(message)
                        or ReservationOrchestrator._is_change_request(message)
                        or ReservationOrchestrator._is_cancel_change_question(message)
                    )
                ):
                    route_info = {**route_info, "route": "reservation", "normalized_message": message, "reason": "cancel-change-override"}
                if route_info and route_info.get("route") == "reservation" and ReservationOrchestrator._is_new_request(message):
                    # 새 요청은 라우터 규격화 문장(이전 병원·시간이 딸려옴) 대신 원문을 넘긴다. 다음 턴의 상태 기억도 원문 기준.
                    route_info = {**route_info, "normalized_message": message}
            if department_answer:
                route_info = {"route": "general", "engine": "rule", "normalized_message": message, "symptom_summary": "", "emergency": False}
            elif not route_info and not message and ui_selection and ui_selection["field"] in reservation_ui_fields:
                # UI 블록 선택 턴: 직전 규격화 문장에 선택값을 붙여 상태를 이어간다.
                prev = self._last_reservation_state(db, payload.userId, session_id, ward_id) or "병원 예약:"
                value = ui_selection["value"]
                carry = ui_selection["field"] in {"location", "location_confirm", "department", "hospital", "hospital_confirm", "date", "time"} and value != "__other__"
                route_info = {
                    "route": "reservation",
                    "normalized_message": f"{prev} / {value}" if carry else prev,
                    "symptom_summary": "",
                    "emergency": False,
                    "engine": "ui",
                }
            if not route_info and not department_answer:
                kw_intent = self.classify_intent(message)
                route_info = {
                    "route": {"hospital_reservation": "reservation", "medical_advice": "symptom", "emergency_guidance": "symptom"}.get(kw_intent, "general"),
                    "normalized_message": message,
                    "symptom_summary": message,
                    "emergency": kw_intent == "emergency_guidance",
                    "engine": "keyword",
                }
            route_info = self._force_emergency(route_info, message)
            self_harm_trigger = self._self_harm_trigger(self_harm_keyword, route_info, history)
            self_harm = self_harm_trigger is not None
            # 상담 내용(증상 요약·예약 문장)은 로그에 남기지 않는다 - 길이만.
            _LOG.info(
                "[라우터] route=%s engine=%s text_len=%s",
                route_info["route"],
                route_info.get("engine"),
                len(str(route_info.get("normalized_message") if route_info["route"] == "reservation" else route_info.get("symptom_summary") or "")),
            )
            reservation_flow_hint = not self_harm and (
                route_info["route"] == "reservation" or bool(ui_selection and ui_selection["field"] in reservation_ui_fields)
            )
            reservation_api_key = None
            reservation_started = time.perf_counter()
            # 예약 계정 비밀값이 없으면 예약 기능만 끈다(상담·이상감지 등 나머지는 그대로).
            reservation_disabled = reservation_flow_hint and not self._settings.has_reservation_account_secret
            if reservation_flow_hint:
                # 보호자가 피보호자 대신이면 피보호자 계정으로 예약한다(피보호자 웹·보호자 웹과 같은 목록).
                reservation_api_key = reservation_credential_service.ensure_credential(
                    db,
                    user_id=ward_id or payload.userId,
                    client=reservation_api_client,
                    name=str(user_context.get("name") or ""),
                    phone=str(user_context.get("phone") or ""),
                )
            if department_answer:
                pass  # 이미 result가 있다
            elif self_harm:
                family = MedicalLlmService.about_family(message)
                # 상담 내용은 남기지 않는다 - 경로 플래그만.
                _LOG.info("[CHAT-SAFETY] self_harm=true path=main trigger=%s family=%s", self_harm_trigger, family)
                result = MedicalLlmService.self_harm_response(family=family, trigger=self_harm_trigger)
            elif reservation_flow_hint:
                reservation_orchestrator = get_reservation_orchestrator()
                # 규격화된 문장("병원 예약: 서울 강남구 / 이비인후과 / ...")을 원래 예약 로직에 넘긴다.
                routed_context = payload.context
                if routed_context and routed_context.wardId:
                    # 환자는 피보호자다 - 예약 초안의 환자명·연락처를 피보호자 것으로 바꿔 넘긴다.
                    routed_context = routed_context.model_copy(
                        update={"name": routed_context.wardName or routed_context.name, "phone": routed_context.wardPhone or routed_context.phone}
                    )
                routed_payload = payload.model_copy(update={"message": route_info["normalized_message"], "context": routed_context})
                reservation_result = None if reservation_disabled else reservation_orchestrator.handle(
                    routed_payload,
                    {
                        "rawMessage": message,
                        "model": self._settings.claude_model_name,
                        "schemaVersion": "ai-reservation-v1",
                        "reservationApiKey": reservation_api_key,
                        # 직전 턴이 남긴 확인 대기·고치기 상태. 서버 DB 값이라 클라이언트가 바꿀 수 없다.
                        "storedFlow": self._last_flow_state(db, payload.userId, session_id),
                    },
                )
                timing["reservationMs"] = _elapsed_ms(reservation_started)
                if reservation_result is not None:
                    result = reservation_result
                else:
                    result = {
                        "reply": (
                            "지금은 병원 예약 기능을 사용할 수 없어요. 병원에 직접 전화해 예약해 주세요."
                            if reservation_disabled
                            else "예약 정보를 처리하지 못했습니다. 잠시 후 다시 시도해 주세요."
                        ),
                        "riskLevel": "medium",
                        "recommendedAction": "guardian_contact",
                        "reservationRequired": True,
                        "intent": "hospital_reservation",
                        "engine": "claude",
                        "modelName": self._settings.claude_model_name,
                        "type": "message",
                        "ui": None,
                        "tool": None,
                        "data": None,
                        "summary": None,
                        "possibleCauses": [],
                        "homeCare": [],
                        "visitHospitalIf": [],
                        "emergencyWarning": [],
                        "meta": {"reservationFlowHint": True},
                        "decisionTrace": [{"event": "reservation_disabled" if reservation_disabled else "reservation_flow_fallback"}],
                    }
            else:
                local_llm = get_medical_llm_service()
                llm_started = time.perf_counter()
                try:
                    if route_info["route"] == "symptom":
                        llm_intent = "emergency_guidance" if route_info.get("emergency") else "medical_advice"
                    else:
                        llm_intent = "general_chat"
                    result = local_llm.generate(
                        message,
                        user_context,
                        history,
                        intent=llm_intent,
                        processed_symptom=route_info.get("symptom_summary") if route_info["route"] == "symptom" else None,
                    )
                    timing["llmMs"] = _elapsed_ms(llm_started)
                except Exception as exc:
                    timing["llmMs"] = _elapsed_ms(llm_started)
                    # 옛 챗 서버로 넘기던 폴백은 없앴다(2026-10-22) - 고정 문구로 끝낸다. 로그는 예외 클래스명만(상담 내용 금지).
                    _LOG.warning("[CHAT-LLM-FAILED] %s", type(exc).__name__)
                    result = self._fallback_reply(message, history, emergency=bool(route_info.get("emergency")))
        except Exception as exc:
            # 예외 문구·스택에는 SQL 파라미터·요청 값이 섞일 수 있어 클래스명과 발생 위치만 남긴다.
            _LOG.error("chat process_message fallback: %s at %s", type(exc).__name__, _origin(exc))
            result = self._fallback_reply(message or log_message, history)
            route_info = {}
        post_started = time.perf_counter()
        if route_info:
            result.setdefault("meta", {})
            if isinstance(result["meta"], dict):
                result["meta"]["router"] = {k: route_info.get(k) for k in ("route", "engine", "reason", "normalized_message", "symptom_summary", "emergency", "emergency_signals")}
                # 라우터 기억은 이 상담의 피보호자에게 묶는다 - 다른 피보호자 상담에서 앞 사람의 병원·날짜를 이어 쓰지 않게.
                result["meta"]["router"]["wardId"] = ward_id

        timing["postMs"] = _elapsed_ms(post_started)
        timing["totalMs"] = _elapsed_ms(started)  # 저장 직전까지. 저장 시간은 행을 쓴 뒤에야 알아 로그에만 남긴다.
        result.setdefault("meta", {})
        if isinstance(result["meta"], dict):
            result["meta"]["timing"] = dict(timing)

        context_payload = {
            "sessionId": session_id,
            "userContext": user_context,
            "historyLength": len(history),
            "engine": result.get("engine", "fallback"),
            "modelName": result.get("modelName"),
            "type": result.get("type", "message"),
            "ui": result.get("ui"),
            "tool": result.get("tool"),
            "data": result.get("data"),
            "summary": result.get("summary"),
            "possibleCauses": result.get("possibleCauses") or [],
            "homeCare": result.get("homeCare") or [],
            "visitHospitalIf": result.get("visitHospitalIf") or [],
            "emergencyWarning": result.get("emergencyWarning") or [],
            "decisionTrace": result.get("decisionTrace") or [],
            "upstreamMeta": _persistable_meta(result.get("meta")),
        }

        log = ChatLog(
            chat_no=f"CHAT-{uuid4().hex[:12]}",
            user_id=payload.userId,
            message=log_message,
            context_json=json.dumps(_json_safe(context_payload), ensure_ascii=False),
            reply=result["reply"],
            risk_level=result["riskLevel"],
            recommended_action=result["recommendedAction"],
            reservation_required=bool(result["reservationRequired"]),
        )
        save_started = time.perf_counter()
        db.add(log)
        db.commit()
        db.refresh(log)
        # 경로·엔진·구간 ms 만 남긴다(상담 내용·사용자 문장 금지).
        _LOG.info(
            "[CHAT-TIMING] route=%s engine=%s router_ms=%s llm_ms=%s reservation_ms=%s post_ms=%s save_ms=%s total_ms=%s",
            route_info.get("route") if route_info else None,
            result.get("engine", "fallback"),
            timing["routerMs"],
            timing["llmMs"],
            timing["reservationMs"],
            timing["postMs"],
            _elapsed_ms(save_started),
            _elapsed_ms(started),
        )

        return {
            "sessionId": session_id,
            "reply": result["reply"],
            "riskLevel": result["riskLevel"],
            "recommendedAction": result["recommendedAction"],
            "reservationRequired": bool(result["reservationRequired"]),
            "intent": result["intent"],
            "engine": result.get("engine", "fallback"),
            "modelName": result.get("modelName"),
            "type": result.get("type", "message"),
            "ui": result.get("ui"),
            "tool": result.get("tool"),
            "toolData": result.get("data"),
            "summary": result.get("summary"),
            "possibleCauses": result.get("possibleCauses") or [],
            "homeCare": result.get("homeCare") or [],
            "visitHospitalIf": result.get("visitHospitalIf") or [],
            "emergencyWarning": result.get("emergencyWarning") or [],
            "chatLogId": log.id,
            "chatNo": log.chat_no,
        }

    @staticmethod
    def _last_flow_state(db: Session, user_id: str, session_id: str) -> dict:
        """직전 턴(같은 세션)이 남긴 예약 확인 대기(pendingAction)·고치기 상태(flowState). 그 사이에 다른 대화가 끼면 비어 있다."""
        log = db.query(ChatLog).filter(ChatLog.user_id == user_id).order_by(ChatLog.id.desc()).first()
        if not log:
            return {}
        try:
            ctx = json.loads(log.context_json or "{}")
        except json.JSONDecodeError:
            return {}
        if ctx.get("sessionId") != session_id:
            return {}
        meta = ctx.get("upstreamMeta") or {}
        return {k: meta[k] for k in ("pendingAction", "flowState") if isinstance(meta.get(k), dict)}

    @staticmethod
    def _last_reservation_state(db: Session, user_id: str, session_id: str, ward_id: str = "") -> str | None:
        """같은 세션의 직전 턴이 예약 흐름이었으면 그때의 규격화 문장을 돌려준다 (라우터의 상태 기억).
        직전 턴이 다른 피보호자의 상담이었으면 비운다. wardId 를 남기기 전의 기록은 본인 상담("")으로 본다."""
        log = db.query(ChatLog).filter(ChatLog.user_id == user_id).order_by(ChatLog.id.desc()).first()
        if not log:
            return None
        try:
            ctx = json.loads(log.context_json or "{}")
        except json.JSONDecodeError:
            return None
        if ctx.get("sessionId") != session_id:
            return None
        router = (ctx.get("upstreamMeta") or {}).get("router") or {}
        if (router.get("wardId") or "") != ward_id:
            return None
        return router.get("normalized_message") if router.get("route") == "reservation" else None

    @staticmethod
    def _build_user_context(payload: ChatRequest) -> dict[str, object]:
        if not payload.context:
            return {}
        ctx: dict[str, object] = {}
        if payload.context.age is not None:
            ctx["age"] = payload.context.age
        if payload.context.email:
            ctx["email"] = payload.context.email.strip()
        if payload.context.name:
            ctx["name"] = payload.context.name.strip()
        if payload.context.phone:
            ctx["phone"] = payload.context.phone.strip()
        if payload.context.wardId:
            # 보호자 상담은 피보호자가 환자다 - 예약 환자명·연락처를 피보호자로 둔다.
            ctx["wardId"] = payload.context.wardId.strip()
            if payload.context.wardName:
                ctx["name"] = payload.context.wardName.strip()
            if payload.context.wardPhone:
                ctx["phone"] = payload.context.wardPhone.strip()
        if payload.context.gender:
            ctx["gender"] = payload.context.gender
        if payload.context.birthDate:
            ctx["birthDate"] = payload.context.birthDate.strip()
        if payload.context.postcode:
            ctx["postcode"] = payload.context.postcode.strip()
        if payload.context.address:
            ctx["address"] = payload.context.address.strip()
        if payload.context.addressDetail:
            ctx["addressDetail"] = payload.context.addressDetail.strip()
        if payload.context.guardianId is not None:
            ctx["guardianId"] = payload.context.guardianId
        if payload.context.location:
            ctx["location"] = payload.context.location.strip()
        if payload.context.role:
            ctx["role"] = payload.context.role.strip()
        return ctx

    @staticmethod
    def _force_emergency(route_info: dict, message: str) -> dict:
        """응급 신호가 있으면 라우트와 상관없이 증상 상담(119 안내)으로 보낸다.
        라우터가 emergency=True를 주고도 route를 reservation·general로 두거나, 라우터가 놓친 키워드가 있어도 응급이 묻히지 않게."""
        signals = MedicalLlmService.emergency_signals(message)
        self_harm = MedicalLlmService.self_harm_signals(message)
        if not route_info or not (signals or self_harm or route_info.get("emergency")):
            return route_info
        return {
            **route_info,
            "route": "symptom",
            "emergency": True,
            "emergency_signals": signals,
            "symptom_summary": route_info.get("symptom_summary") or message,
        }

    @staticmethod
    def _self_harm_trigger(keyword: bool, route_info: dict, history: list[dict[str, str]]) -> str | None:
        """위기 고정 응답을 낼 근거. keyword = 이번 문장 키워드 / router = 라우터가 응급으로 보고 요약에 자해·자살(부정형 제외) /
        history = 이번 턴이 응급이고 직전 사용자 턴 2개에 자해 신호. 응급이 아니면 앞 턴 때문에 위기 응답에 가두지 않는다."""
        if keyword:
            return "keyword"
        if not route_info or not route_info.get("emergency"):
            return None
        if MedicalLlmService.self_harm_in_summary(str(route_info.get("symptom_summary") or "")):
            return "router"
        if MedicalLlmService.self_harm_in_history(history):
            return "history"
        return None

    @classmethod
    def classify_intent(cls, message: str) -> str:
        text = (message or "").strip().lower()
        if MedicalLlmService.emergency_signals(text):
            return "emergency_guidance"
        if any(keyword in text for keyword in cls.RESERVATION_KEYWORDS):
            return "hospital_reservation"
        if any(keyword in text for keyword in cls.MEDIUM_RISK_KEYWORDS):
            return "medical_advice"
        return "general_chat"

    @classmethod
    def infer_risk(cls, message: str, intent: str) -> tuple[str, str, bool]:
        text = (message or "").strip().lower()
        if intent == "emergency_guidance" or any(keyword in text for keyword in cls.HIGH_RISK_KEYWORDS):
            return "high", "hospital_visit", True
        if intent in {"hospital_reservation", "medical_advice"} or any(keyword in text for keyword in cls.MEDIUM_RISK_KEYWORDS):
            return "medium", "guardian_contact", intent == "hospital_reservation"
        return "low", "home_monitoring", False

    @staticmethod
    def _department_question_reply(message: str, history: list[dict[str, str]]) -> dict | None:
        """"무슨 과로 가야 해?"에는 직전 상담에서 권한 진료과를 한 줄로 답한다(같은 상담 답변을 반복하지 않게)."""
        if not re.search(r"(무슨|어느|어떤)\s*과", message or ""):
            return None
        last_assistant = next((h.get("content", "") for h in reversed(history) if h.get("role") == "assistant"), "")
        department = _find_department(last_assistant)
        if not department:
            return None
        return {
            "reply": f"{department}로 가시면 돼요. 가까운 {department}를 찾아드릴까요? \"병원 찾아줘\"라고 말씀해 주세요.",
            "riskLevel": "medium",
            "recommendedAction": "guardian_contact",
            "reservationRequired": False,
            "intent": "medical_advice",
            "engine": "rule",
            "modelName": None,
            "summary": None,
            "possibleCauses": [],
            "homeCare": [],
            "visitHospitalIf": [],
            "emergencyWarning": [],
        }

    def _fallback_reply(self, message: str, history: list[dict[str, str]], emergency: bool = False) -> dict:
        trigger = "keyword" if MedicalLlmService.self_harm_signals(message) else (
            "history" if emergency and MedicalLlmService.self_harm_in_history(history) else None
        )
        if trigger:
            family = MedicalLlmService.about_family(message)
            _LOG.info("[CHAT-SAFETY] self_harm=true path=fallback trigger=%s family=%s", trigger, family)
            return MedicalLlmService.self_harm_response(family=family, trigger=trigger)
        intent = self.classify_intent(message)
        if emergency:
            # 라우터가 응급으로 본 문장은 키워드에 안 걸려도 119 안내가 빠지면 안 된다.
            intent = "emergency_guidance"
        risk_level, action, reservation_required = self.infer_risk(message, intent)
        prior_user = [item["content"] for item in history if item.get("role") == "user"][-2:]
        context_hint = ""
        if prior_user:
            context_hint = f" 이전에 말씀하신 내용({prior_user[0][:30]})도 함께 고려해 주세요."

        if risk_level == "high":
            reply = (
                "현재 증상은 즉시 확인이 필요할 수 있습니다. 119 또는 가까운 응급실 방문을 권장합니다."
                f"{context_hint} "
                "이 답변은 일반적인 건강 정보 안내이며, 정확한 진단과 치료는 의료진 상담이 필요합니다."
            )
        elif risk_level == "medium":
            reply = (
                # 보호자·피보호자 모두에게 쓰이는 문구다 - "보호자"로 못 박으면 보호자 본인이나 연결된 보호자가 없는
                # 피보호자에게 맞지 않는다(백엔드 점검 L-4, 2026-10-09).
                "증상이 지속되면 진료가 필요할 수 있습니다. 가족이나 가까운 분과 상태를 공유하고 경과를 관찰해 주세요."
                f"{context_hint} "
                "이 답변은 일반적인 건강 정보 안내이며, 정확한 진단과 치료는 의료진 상담이 필요합니다."
            )
        else:
            reply = (
                f"전달해 주신 내용을 기준으로 우선 안정 상태로 보입니다.{context_hint} "
                "이상 징후가 있으면 증상이 언제부터 시작됐는지, 다른 동반 증상이 있는지 알려주세요. "
                "이 답변은 일반적인 건강 정보 안내이며, 정확한 진단과 치료는 의료진 상담이 필요합니다."
            )

        return {
            "reply": reply,
            "riskLevel": risk_level,
            "recommendedAction": action,
            "reservationRequired": reservation_required,
            "intent": intent,
            "engine": "fallback",
            "modelName": None,
            "summary": None,
            "possibleCauses": [],
            "homeCare": [],
            "visitHospitalIf": [],
            # MedGemma 응급 경로와 같은 경고 박스 - 모델이 실패해도 응급 안내가 빨간 박스에서 빠지지 않게(#33).
            "emergencyWarning": [MedicalLlmService.EMERGENCY_WARNING] if intent == "emergency_guidance" else [],
        }

    @staticmethod
    def delete_logs(db: Session, user_id: str) -> int:
        """해당 회원의 상담 기록을 모두 지우고 지운 건수를 돌려준다(회원 탈퇴 정리용). 다른 회원 기록은 건드리지 않는다."""
        deleted = db.query(ChatLog).filter(ChatLog.user_id == user_id).delete(synchronize_session=False)
        db.commit()
        return int(deleted)

    @staticmethod
    def list_logs(db: Session, user_id: str | None = None) -> list[dict]:
        query = db.query(ChatLog)
        if user_id is not None:
            query = query.filter(ChatLog.user_id == user_id)
        logs = query.order_by(ChatLog.id.desc()).all()
        items: list[dict] = []
        for log in logs:
            context = {}
            try:
                context = json.loads(log.context_json or "{}")
            except json.JSONDecodeError:
                context = {}
            items.append(
                {
                    "id": log.id,
                    "chatNo": log.chat_no,
                    "userId": log.user_id,
                    "message": log.message,
                    "contextJson": log.context_json,
                    "sessionId": context.get("sessionId"),
                    "engine": context.get("engine"),
                    "intent": context.get("conversationState", {}).get("intent") if isinstance(context.get("conversationState"), dict) else context.get("intent"),
                    "modelName": context.get("modelName"),
                    "type": context.get("type"),
                    "tool": context.get("tool"),
                    "toolData": context.get("data"),
                    "ui": context.get("ui"),
                    "reply": log.reply,
                    "riskLevel": log.risk_level,
                    "recommendedAction": log.recommended_action,
                    "reservationRequired": log.reservation_required,
                    # 기록을 다시 열 때 빨간 경고 박스를 복원하려고 내보낸다(BE 기록 허용 목록 emergencyWarning 과 짝).
                    "emergencyWarning": context.get("emergencyWarning") or [],
                    "decisionTrace": context.get("decisionTrace") or [],
                    "upstreamMeta": context.get("upstreamMeta") or {},
                    "createdAt": log.created_at.isoformat(),
                },
            )
        return items

    @staticmethod
    def get_log(db: Session, chat_id: int, user_id: str | None = None) -> dict | None:
        log = db.query(ChatLog).filter(ChatLog.id == chat_id).first()
        if not log:
            return None
        if user_id is not None and log.user_id != user_id:
            return None
        context = {}
        try:
            context = json.loads(log.context_json or "{}")
        except json.JSONDecodeError:
            context = {}
        return {
            "id": log.id,
            "chatNo": log.chat_no,
            "userId": log.user_id,
            "message": log.message,
            "contextJson": context,
            "sessionId": context.get("sessionId"),
            "engine": context.get("engine"),
            "intent": context.get("conversationState", {}).get("intent") if isinstance(context.get("conversationState"), dict) else context.get("intent"),
            "modelName": context.get("modelName"),
            "type": context.get("type"),
            "tool": context.get("tool"),
            "toolData": context.get("data"),
            "ui": context.get("ui"),
            "reply": log.reply,
            "riskLevel": log.risk_level,
            "recommendedAction": log.recommended_action,
            "reservationRequired": log.reservation_required,
            "emergencyWarning": context.get("emergencyWarning") or [],
            "decisionTrace": context.get("decisionTrace") or [],
            "upstreamMeta": context.get("upstreamMeta") or {},
            "createdAt": log.created_at.isoformat(),
        }
