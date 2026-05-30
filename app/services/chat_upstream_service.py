from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from app.core.config import Settings, get_settings


class ChatUpstreamService:
    """ChatSilverBridge 등 외부 MedGemma 챗 API 프록시."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def enabled(self) -> bool:
        return bool((self._settings.chat_upstream_url or "").strip())

    def generate(
        self,
        message: str,
        session_id: str,
        user_id: int,
        user_context: dict[str, Any] | None,
        history: list[dict[str, str]],
        ui_selection: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("CHAT_UPSTREAM_URL 미설정")

        base_url = self._settings.chat_upstream_url.rstrip("/")
        payload: dict[str, Any] = {
            "sessionId": session_id,
            "externalSessionId": f"ai-server-user-{user_id}",
            "message": message,
            "history": history,
        }
        if user_context:
            upstream_context: dict[str, Any] = {}
            if user_context.get("age") is not None:
                upstream_context["age"] = user_context["age"]
            if user_context.get("email"):
                upstream_context["email"] = str(user_context["email"]).strip()
            if user_context.get("name"):
                upstream_context["name"] = str(user_context["name"]).strip()
            if user_context.get("phone"):
                upstream_context["phone"] = str(user_context["phone"]).strip()
            if user_context.get("gender"):
                upstream_context["gender"] = user_context["gender"]
            if user_context.get("birthDate"):
                upstream_context["birthDate"] = str(user_context["birthDate"]).strip()
            if user_context.get("postcode"):
                upstream_context["postcode"] = str(user_context["postcode"]).strip()
            if user_context.get("address"):
                upstream_context["address"] = str(user_context["address"]).strip()
            if user_context.get("addressDetail"):
                upstream_context["addressDetail"] = str(user_context["addressDetail"]).strip()
            if user_context.get("location"):
                upstream_context["location"] = user_context["location"]
            if user_context.get("role"):
                upstream_context["role"] = str(user_context["role"]).strip()
            if upstream_context:
                payload["userContext"] = upstream_context
        if ui_selection:
            payload["uiSelection"] = ui_selection

        request = urllib.request.Request(
            f"{base_url}/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._settings.chat_upstream_timeout_sec) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"upstream HTTP {exc.code}: {detail[:200]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"upstream 연결 실패: {exc.reason}") from exc

        return self._normalize_response(body, message)

    @staticmethod
    def _normalize_response(body: dict[str, Any], message: str) -> dict[str, Any]:
        if not body.get("success", True):
            error = body.get("error") or {}
            raise RuntimeError(str(error.get("message") or error or "upstream chat failed"))

        reply = str(body.get("message") or "").strip()
        response_type = str(body.get("type") or "message")
        meta = body.get("meta") or {}
        conversation_state = meta.get("conversationState") or {}
        intent = str(conversation_state.get("intent") or "general_chat")
        model_name = str(meta.get("model") or "upstream-medgemma")
        ui = body.get("ui")
        tool = body.get("tool")
        tool_data = body.get("data")

        if response_type == "tool_result" and tool_data is not None:
            if not reply:
                if tool == "search_hospital":
                    reply = "병원 검색 결과입니다."
                elif tool == "make_appointment":
                    reply = "예약 처리 결과입니다."
                else:
                    reply = f"[{tool or 'tool'}] 결과"

        if response_type == "tool_error":
            error = body.get("error") or {}
            raise RuntimeError(str(error.get("message") or "upstream tool error"))

        if not reply and response_type != "tool_result":
            raise RuntimeError("upstream empty reply")

        risk_level, recommended_action, reservation_required = ChatUpstreamService._infer_risk(
            message,
            intent,
            response_type,
            tool,
        )

        return {
            "reply": reply,
            "riskLevel": risk_level,
            "recommendedAction": recommended_action,
            "reservationRequired": reservation_required,
            "intent": intent,
            "engine": "medgemma",
            "modelName": model_name,
            "type": response_type,
            "ui": ui,
            "tool": tool,
            "data": tool_data,
            "summary": None,
            "possibleCauses": [],
            "homeCare": [],
            "visitHospitalIf": [],
            "emergencyWarning": [],
            "meta": meta,
            "decisionTrace": meta.get("decisionTrace") or [],
        }

    @staticmethod
    def _infer_risk(
        message: str,
        intent: str,
        response_type: str,
        tool: Any,
    ) -> tuple[str, str, bool]:
        text = (message or "").strip().lower()
        high_keywords = ("숨이", "호흡", "가슴", "의식", "실신", "출혈", "경련", "쓰러")
        medium_keywords = ("어지럽", "복통", "발열", "기침", "두통", "구토", "아프")

        if intent == "emergency_guidance" or any(keyword in text for keyword in high_keywords):
            return "high", "hospital_visit", True
        if response_type == "tool_result" or tool:
            return "medium", "guardian_contact", True
        if intent in {"hospital_reservation", "medical_advice"} or any(keyword in text for keyword in medium_keywords):
            return "medium", "guardian_contact", intent == "hospital_reservation"
        return "low", "home_monitoring", False


def get_chat_upstream_service() -> ChatUpstreamService:
    return ChatUpstreamService(get_settings())
