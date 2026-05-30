from __future__ import annotations

from functools import lru_cache
import re
from typing import Any

from app.core.config import Settings, get_settings
from app.models.medgemma_loader import MedGemmaLoader
from app.prompts.medical_chat_prompts import SYSTEM_PROMPT, build_user_prompt
from app.utils.conversation import format_recent_history, normalize_history
from app.utils.json_extract import extract_json_object


class MedicalLlmService:
    EMERGENCY_KEYWORDS = (
        "숨이 안 쉬어",
        "호흡곤란",
        "가슴 통증",
        "의식 없음",
        "쓰러짐",
        "피가 멈추지",
        "경련",
    )
    RESERVATION_KEYWORDS = ("예약", "병원 찾", "근처 병원", "진료 가능", "응급실")
    MEDICAL_KEYWORDS = ("아파", "아프", "통증", "열", "기침", "두통", "어지럽", "구토", "충혈")

    def __init__(self, settings: Settings, loader: MedGemmaLoader) -> None:
        self._settings = settings
        self._loader = loader

    @classmethod
    def classify_intent(cls, message: str) -> str:
        text = (message or "").strip().lower()
        if any(keyword in text for keyword in cls.EMERGENCY_KEYWORDS):
            return "emergency_guidance"
        if any(keyword in text for keyword in cls.RESERVATION_KEYWORDS):
            return "hospital_reservation"
        if any(keyword in text for keyword in cls.MEDICAL_KEYWORDS):
            return "medical_advice"
        return "general_chat"

    @staticmethod
    def infer_risk(message: str, intent: str, emergency_warning: list[str]) -> tuple[str, str, bool]:
        text = (message or "").strip().lower()
        if intent == "emergency_guidance" or emergency_warning:
            return "high", "hospital_visit", True
        if intent in {"hospital_reservation", "medical_advice"}:
            return "medium", "guardian_contact", intent == "hospital_reservation"
        if any(keyword in text for keyword in MedicalLlmService.MEDICAL_KEYWORDS):
            return "medium", "guardian_contact", False
        return "low", "home_monitoring", False

    @staticmethod
    def format_answer(data: dict[str, Any]) -> str:
        parts: list[str] = []
        final_message = str(data.get("finalMessage") or "").strip()
        if final_message:
            parts.append(final_message)

        summary = str(data.get("summary") or "").strip()
        if summary and summary not in final_message:
            parts.append(summary)

        def append_section(title: str, key: str) -> None:
            items = [str(item).strip() for item in (data.get(key) or []) if str(item).strip()]
            if items:
                parts.append(f"{title}\n- " + "\n- ".join(items))

        append_section("가능한 원인", "possibleCauses")
        append_section("자가 관리", "homeCare")
        append_section("병원 방문이 필요한 경우", "visitHospitalIf")
        append_section("응급 주의", "emergencyWarning")

        disclaimer = "이 답변은 일반적인 건강 정보 안내이며, 정확한 진단과 치료는 의료진 상담이 필요합니다."
        body = "\n\n".join(parts).strip() or "증상을 조금 더 구체적으로 말씀해 주시면 도와드릴게요."
        if disclaimer not in body:
            body = f"{body}\n\n{disclaimer}"
        return body

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value).strip()
        if isinstance(value, list):
            for item in value:
                text = MedicalLlmService._normalize_text(item)
                if text:
                    return text
        return ""

    @staticmethod
    def _normalize_list(value: Any) -> list[str]:
        if isinstance(value, str):
            raw_items = re.split(r"[\n•·,;/]+", value)
        elif isinstance(value, list):
            raw_items = value
        else:
            return []

        items: list[str] = []
        for item in raw_items:
            text = MedicalLlmService._normalize_text(item)
            if text and text.lower() not in {"none", "null"}:
                items.append(text)
        return items[:3]

    @staticmethod
    def _normalize_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on", "예", "네"}
        return False

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        chunks = re.split(r"(?<=[.!?。])\s+|\n+", text or "")
        return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]

    @classmethod
    def _sanitize_unstructured_text(cls, raw_text: str) -> str:
        lines: list[str] = []
        for line in (raw_text or "").splitlines():
            s = line.strip()
            if not s:
                continue
            if s.startswith("<unused") or s.startswith("```"):
                break
            if s.startswith("Constraint Checklist") or s.startswith("Mental Sandbox"):
                break
            if s.startswith("MedGemma") or s.startswith("위험도"):
                break
            if s.startswith("The user") or s.startswith("I need to"):
                break
            lines.append(s)
        return " ".join(lines).strip()

    @classmethod
    def _fallback_payload(cls, message: str, intent: str, emergency_hint: bool, raw_text: str) -> dict[str, Any]:
        sanitized = cls._sanitize_unstructured_text(raw_text)
        sentence_source = sanitized or message or ""
        sentences = cls._split_sentences(sentence_source)
        first_sentence = sentences[0] if sentences else ""
        second_sentence = sentences[1] if len(sentences) > 1 else ""

        if emergency_hint or intent == "emergency_guidance":
            return {
                "summary": "응급 가능성을 우선 확인하세요.",
                "possibleCauses": [],
                "homeCare": [],
                "visitHospitalIf": [],
                "emergencyWarning": ["응급 증상이 의심됩니다. 즉시 119 또는 가까운 응급실로 도움을 요청하세요."],
                "finalMessage": "즉시 119 또는 가까운 응급실에 도움을 요청하세요.",
                "needsReservation": False,
            }

        summary = first_sentence or "입력된 증상을 확인했습니다."
        final_message = first_sentence or "증상을 조금 더 구체적으로 말씀해 주시면 도와드릴게요."
        possible_causes = [second_sentence] if second_sentence else []
        home_care = ["충분히 쉬고 수분을 보충하세요."]
        visit_hospital_if = ["통증이 심하거나 지속되면 의료진 상담을 받으세요."]
        needs_reservation = intent == "hospital_reservation"
        if intent == "general_chat" and not first_sentence:
            summary = "일반적인 건강 안내입니다."

        return {
            "summary": summary,
            "possibleCauses": possible_causes,
            "homeCare": home_care,
            "visitHospitalIf": visit_hospital_if,
            "emergencyWarning": [],
            "finalMessage": final_message,
            "needsReservation": needs_reservation,
        }

    @classmethod
    def _normalize_payload(
        cls,
        parsed: dict[str, Any] | None,
        raw_text: str,
        message: str,
        intent: str,
        emergency_hint: bool,
    ) -> dict[str, Any]:
        base = cls._fallback_payload(message, intent, emergency_hint, raw_text)
        if not isinstance(parsed, dict):
            return base

        summary = cls._normalize_text(parsed.get("summary"))
        final_message = cls._normalize_text(parsed.get("finalMessage") or parsed.get("reply"))
        possible_causes = cls._normalize_list(parsed.get("possibleCauses") or parsed.get("possible_causes"))
        home_care = cls._normalize_list(parsed.get("homeCare") or parsed.get("home_care"))
        visit_hospital_if = cls._normalize_list(parsed.get("visitHospitalIf") or parsed.get("visit_hospital_if"))
        emergency_warning = cls._normalize_list(parsed.get("emergencyWarning") or parsed.get("emergency_warning"))

        needs_reservation = parsed.get("needsReservation")
        if needs_reservation is None:
            needs_reservation = parsed.get("reservationRequired")
        if needs_reservation is None:
            needs_reservation = parsed.get("needs_reservation")
        needs_reservation = cls._normalize_bool(needs_reservation)

        return {
            "summary": summary or base["summary"],
            "possibleCauses": possible_causes or base["possibleCauses"],
            "homeCare": home_care or base["homeCare"],
            "visitHospitalIf": visit_hospital_if or base["visitHospitalIf"],
            "emergencyWarning": emergency_warning or base["emergencyWarning"],
            "finalMessage": final_message or base["finalMessage"],
            "needsReservation": needs_reservation or base["needsReservation"],
        }

    def generate(
        self,
        message: str,
        user_context: dict[str, Any] | None,
        history: list[dict[str, str]],
    ) -> dict[str, Any]:
        intent = self.classify_intent(message)
        emergency_hint = any(keyword in message.lower() for keyword in self.EMERGENCY_KEYWORDS)

        recent, summary = normalize_history(
            history,
            keep_turns=self._settings.chat_history_keep_turns,
        )
        history_lines = format_recent_history(recent)
        if summary:
            history_lines = f"(이전 요약) {summary}\n{history_lines}".strip()

        user_prompt = build_user_prompt(
            message=message,
            user_context=user_context,
            history_text=history_lines or "(없음)",
            intent=intent,
            emergency_hint=emergency_hint,
        )

        if not self._settings.chat_enable_llm:
            raise RuntimeError("CHAT_ENABLE_LLM=false")

        self._loader.ensure_loaded()
        raw = self._loader.generate_structured_text(SYSTEM_PROMPT, user_prompt)
        parsed = extract_json_object(raw)
        normalized = self._normalize_payload(parsed, raw, message, intent, emergency_hint)

        emergency_warning = normalized["emergencyWarning"]
        if emergency_hint and not emergency_warning:
            emergency_warning = ["현재 응급 위험 신호가 의심됩니다. 즉시 119 또는 가까운 응급실로 연락하세요."]

        risk_level, recommended_action, reservation_required = self.infer_risk(
            message,
            intent,
            emergency_warning,
        )
        needs_reservation = bool(normalized.get("needsReservation")) or reservation_required

        return {
            "reply": self.format_answer(normalized),
            "riskLevel": risk_level,
            "recommendedAction": recommended_action,
            "reservationRequired": needs_reservation,
            "intent": intent,
            "engine": "medgemma",
            "modelName": self._loader.state.model_name,
            "summary": normalized.get("summary"),
            "possibleCauses": normalized.get("possibleCauses") or [],
            "homeCare": normalized.get("homeCare") or [],
            "visitHospitalIf": normalized.get("visitHospitalIf") or [],
            "emergencyWarning": emergency_warning,
        }


@lru_cache
def get_medgemma_loader() -> MedGemmaLoader:
    return MedGemmaLoader(get_settings())


@lru_cache
def get_medical_llm_service() -> MedicalLlmService:
    settings = get_settings()
    return MedicalLlmService(settings, get_medgemma_loader())
