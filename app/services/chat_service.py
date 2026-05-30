from __future__ import annotations

import json
from uuid import uuid4

from sqlalchemy.orm import Session

from app.models.chat_log import ChatLog
from app.schemas.chat_schema import ChatRequest


class ChatService:
    """
    ChatSilverBridge의 의료 챗 목적을 유지하기 위한 MVP 서비스.
    - 증상 기반 위험도 분류
    - 권장 행동 생성
    - 로그 저장
    """

    EMERGENCY_KEYWORDS = ("숨이 안 쉬어", "호흡곤란", "가슴 통증", "의식 없음", "쓰러짐", "피가 멈추지", "경련")
    HIGH_RISK_KEYWORDS = ("가슴", "호흡곤란", "숨이", "의식", "실신", "출혈", "마비")
    MEDIUM_RISK_KEYWORDS = ("어지럽", "복통", "발열", "기침", "두통", "구토")
    RESERVATION_KEYWORDS = ("예약", "병원 찾아", "근처 병원", "진료 가능한", "응급실")

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
            return ("high", "hospital_visit", True)
        if intent in {"hospital_reservation", "medical_advice"} or any(keyword in text for keyword in cls.MEDIUM_RISK_KEYWORDS):
            return ("medium", "guardian_contact", False)
        return ("low", "home_monitoring", False)

    def process_message(self, db: Session, payload: ChatRequest) -> dict:
        intent = self.classify_intent(payload.message)
        risk_level, action, reservation_required = self.infer_risk(payload.message, intent)
        reply = self._build_reply(risk_level=risk_level, message=payload.message)
        context_json = payload.context.model_dump_json() if payload.context else "{}"

        log = ChatLog(
            chat_no=f"CHAT-{uuid4().hex[:12]}",
            user_id=payload.userId,
            message=payload.message,
            context_json=context_json,
            reply=reply,
            risk_level=risk_level,
            recommended_action=action,
            reservation_required=reservation_required,
        )
        db.add(log)
        db.commit()
        db.refresh(log)

        return {
            "reply": reply,
            "riskLevel": risk_level,
            "recommendedAction": action,
            "reservationRequired": reservation_required,
            "intent": intent,
            "chatLogId": log.id,
            "chatNo": log.chat_no,
        }

    @staticmethod
    def _build_reply(risk_level: str, message: str) -> str:
        if risk_level == "high":
            return (
                "현재 증상은 즉시 확인이 필요할 수 있습니다. "
                "가까운 병원 방문 또는 보호자 연락을 권장합니다. "
                "이 답변은 일반적인 건강 정보 안내이며, 정확한 진단과 치료는 의료진 상담이 필요합니다."
            )
        if risk_level == "medium":
            return (
                "증상이 지속되면 진료가 필요할 수 있습니다. "
                "보호자와 상태를 공유하고 경과를 관찰해 주세요. "
                "이 답변은 일반적인 건강 정보 안내이며, 정확한 진단과 치료는 의료진 상담이 필요합니다."
            )
        return (
            f"전달된 내용({message[:40]})을 기준으로 우선 안정 상태로 보입니다. 이상 징후 시 즉시 재문의하세요. "
            "이 답변은 일반적인 건강 정보 안내이며, 정확한 진단과 치료는 의료진 상담이 필요합니다."
        )

    @staticmethod
    def list_logs(db: Session) -> list[dict]:
        logs = db.query(ChatLog).order_by(ChatLog.id.desc()).all()
        items: list[dict] = []
        for log in logs:
            items.append(
                {
                    "id": log.id,
                    "chatNo": log.chat_no,
                    "userId": log.user_id,
                    "message": log.message,
                    "contextJson": log.context_json,
                    "reply": log.reply,
                    "riskLevel": log.risk_level,
                    "recommendedAction": log.recommended_action,
                    "reservationRequired": log.reservation_required,
                    "createdAt": log.created_at.isoformat(),
                },
            )
        return items

    @staticmethod
    def get_log(db: Session, chat_id: int) -> dict | None:
        log = db.query(ChatLog).filter(ChatLog.id == chat_id).first()
        if not log:
            return None
        return {
            "id": log.id,
            "chatNo": log.chat_no,
            "userId": log.user_id,
            "message": log.message,
            "contextJson": json.loads(log.context_json or "{}"),
            "reply": log.reply,
            "riskLevel": log.risk_level,
            "recommendedAction": log.recommended_action,
            "reservationRequired": log.reservation_required,
            "createdAt": log.created_at.isoformat(),
        }
