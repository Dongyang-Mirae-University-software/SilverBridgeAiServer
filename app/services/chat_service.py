from __future__ import annotations

import json
import logging
from uuid import uuid4
from typing import Any

from sqlalchemy.orm import Session  # pyright: ignore[reportMissingImports]

from app.models.chat_log import ChatLog
from app.core.config import get_settings
from app.schemas.chat_schema import ChatRequest
from app.services.reservation_api_client import get_reservation_api_client
from app.services.reservation_credential_service import get_reservation_credential_service
from app.services.chat_upstream_service import get_chat_upstream_service
from app.services.medical_llm_service import get_medical_llm_service
from app.services.reservation_orchestrator import get_reservation_orchestrator

_LOG = logging.getLogger(__name__)


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

    EMERGENCY_KEYWORDS = ("숨이 안 쉬어", "호흡곤란", "가슴 통증", "의식 없음", "쓰러짐", "피가 멈추지", "경련")
    HIGH_RISK_KEYWORDS = ("가슴", "호흡곤란", "숨이", "의식", "실신", "출혈", "마비")
    MEDIUM_RISK_KEYWORDS = ("어지럽", "복통", "발열", "기침", "두통", "구토")
    RESERVATION_KEYWORDS = ("예약", "병원 찾아", "근처 병원", "진료 가능한", "응급실")

    def process_message(self, db: Session, payload: ChatRequest) -> dict:
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
                "reservation_followup",
                "reservation_id",
            }
            reservation_flow_hint = self.classify_intent(message) == "hospital_reservation" or bool(
                ui_selection and ui_selection["field"] in reservation_ui_fields
            )
            reservation_api_key = None
            if reservation_flow_hint:
                reservation_api_key = reservation_credential_service.ensure_credential(
                    db,
                    user_id=payload.userId,
                    reservation_email=payload.context.email if payload.context and payload.context.email else None,
                    client=reservation_api_client,
                )
            if reservation_flow_hint:
                reservation_orchestrator = get_reservation_orchestrator()
                reservation_result = reservation_orchestrator.handle(
                    payload,
                    {
                        "model": self._settings.gpt_model_name,
                        "schemaVersion": "ai-reservation-v1",
                        "reservationApiKey": reservation_api_key,
                    },
                )
                if reservation_result is not None:
                    result = reservation_result
                else:
                    result = {
                        "reply": "예약 정보를 처리하지 못했습니다. 잠시 후 다시 시도해 주세요.",
                        "riskLevel": "medium",
                        "recommendedAction": "guardian_contact",
                        "reservationRequired": True,
                        "intent": "hospital_reservation",
                        "engine": "gpt",
                        "modelName": self._settings.gpt_model_name,
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
                        "decisionTrace": [{"event": "reservation_flow_fallback"}],
                    }
            else:
                local_llm = get_medical_llm_service()
                try:
                    result = local_llm.generate(message, user_context, history)
                except Exception:
                    upstream = get_chat_upstream_service()
                    if upstream.enabled:
                        result = upstream.generate(
                            message,
                            session_id,
                            payload.userId,
                            user_context,
                            history,
                            ui_selection=ui_selection,
                        )
                    else:
                        raise
        except Exception as exc:
            _LOG.exception("chat process_message fallback: %s", exc)
            result = self._fallback_reply(message or log_message, history)

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
            "upstreamMeta": result.get("meta") or {},
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
        db.add(log)
        db.commit()
        db.refresh(log)

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
            "data": result.get("data"),
            "summary": result.get("summary"),
            "possibleCauses": result.get("possibleCauses") or [],
            "homeCare": result.get("homeCare") or [],
            "visitHospitalIf": result.get("visitHospitalIf") or [],
            "emergencyWarning": result.get("emergencyWarning") or [],
            "chatLogId": log.id,
            "chatNo": log.chat_no,
        }

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

    @classmethod
    def classify_intent(cls, message: str) -> str:
        text = (message or "").strip().lower()
        if any(keyword in text for keyword in cls.EMERGENCY_KEYWORDS):
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

    def _fallback_reply(self, message: str, history: list[dict[str, str]]) -> dict:
        intent = self.classify_intent(message)
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
                "증상이 지속되면 진료가 필요할 수 있습니다. 보호자와 상태를 공유하고 경과를 관찰해 주세요."
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
            "emergencyWarning": [],
        }

    @staticmethod
    def list_logs(db: Session, user_id: int | None = None) -> list[dict]:
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
                    "data": context.get("data"),
                    "ui": context.get("ui"),
                    "reply": log.reply,
                    "riskLevel": log.risk_level,
                    "recommendedAction": log.recommended_action,
                    "reservationRequired": log.reservation_required,
                    "decisionTrace": context.get("decisionTrace") or [],
                    "upstreamMeta": context.get("upstreamMeta") or {},
                    "createdAt": log.created_at.isoformat(),
                },
            )
        return items

    @staticmethod
    def get_log(db: Session, chat_id: int, user_id: int | None = None) -> dict | None:
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
            "data": context.get("data"),
            "ui": context.get("ui"),
            "reply": log.reply,
            "riskLevel": log.risk_level,
            "recommendedAction": log.recommended_action,
            "reservationRequired": log.reservation_required,
            "decisionTrace": context.get("decisionTrace") or [],
            "upstreamMeta": context.get("upstreamMeta") or {},
            "createdAt": log.created_at.isoformat(),
        }
