"""챗봇 안전 규칙(2026-10-24, C단계): 자해·자살 신호 고정 응답 / 약 문의+증상 병원 권유 보장 / 폴백 응급 경고 / 진료과 키워드."""
from __future__ import annotations

import json
import logging
import tempfile
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.models.chat_log import ChatLog
from app.schemas.chat_schema import ChatContext, ChatRequest
from app.services import chat_service as cs
from app.services.medical_llm_service import MedicalLlmService
from app.services.reservation_orchestrator import _symptom_department

# 응답 계약(변경 전과 같은 키 집합)
RESPONSE_KEYS = {
    "sessionId", "reply", "riskLevel", "recommendedAction", "reservationRequired", "intent", "engine", "modelName",
    "type", "ui", "tool", "toolData", "summary", "possibleCauses", "homeCare", "visitHospitalIf", "emergencyWarning",
    "chatLogId", "chatNo",
}
SELF_HARM = ["죽고 싶어요", "살기 싫어", "약을 다 먹어버릴까", "그만 살고 싶어", "자살하고 싶어요", "다 죽어버리고 싶다"]
# 일상·과장 표현 - 위기 응답이 나오면 안 된다
NOT_SELF_HARM = ["이제 그만 살 빼야지", "밥 다 먹어버릴까", "무릎 아파 죽겠어", "다 그만두고 싶어"]


class _Router:
    def __init__(self, emergency: bool = False) -> None:
        self.emergency = emergency

    def route(self, message: str, history, ctx, prev_reservation=None) -> dict[str, Any]:  # noqa: ARG002
        return {"route": "symptom", "normalized_message": message, "symptom_summary": message,
                "emergency": self.emergency, "reason": "t", "engine": "fake"}


class _Llm:
    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def generate(self, *a, **k) -> dict[str, Any]:  # noqa: ARG002
        self.calls += 1
        if self.fail:
            raise RuntimeError("민감한 상담 문장이 섞인 예외 원문")
        return {"reply": "모델 답변", "riskLevel": "medium", "recommendedAction": "guardian_contact",
                "reservationRequired": False, "intent": "medical_advice", "engine": "medgemma", "modelName": "m",
                "summary": "s", "possibleCauses": [], "homeCare": [], "visitHospitalIf": [], "emergencyWarning": []}


@pytest.fixture()
def env(monkeypatch):
    engine = create_engine(f"sqlite:///{tempfile.mkdtemp(prefix='sb_chat_safety_')}/t.db")
    ChatLog.__table__.create(bind=engine)
    db = sessionmaker(bind=engine)()
    holder = SimpleNamespace(llm=_Llm(), router=_Router())
    monkeypatch.setattr(cs, "get_chat_router_service", lambda: holder.router)
    monkeypatch.setattr(cs, "get_reservation_credential_service", lambda: object())
    monkeypatch.setattr(cs, "get_reservation_api_client", lambda: object())
    monkeypatch.setattr(cs, "get_medical_llm_service", lambda: holder.llm)
    svc = cs.ChatService()
    svc._settings = SimpleNamespace(has_reservation_account_secret=True, claude_model_name="fake")
    holder.svc, holder.db = svc, db
    yield holder
    db.close()


def _send(env, message: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    payload = ChatRequest(userId="user-1", message=message, sessionId="s1", context=ChatContext(name="홍길동"),
                          history=history or [])
    return env.svc.process_message(env.db, payload)


# ---- C-1 자해·자살 신호 ------------------------------------------------------------------------------


@pytest.mark.parametrize("message", SELF_HARM)
def test_self_harm_signal_detected(message: str) -> None:
    assert MedicalLlmService.self_harm_signals(message)


@pytest.mark.parametrize("message", NOT_SELF_HARM)
def test_everyday_phrases_are_not_self_harm(message: str) -> None:
    assert MedicalLlmService.self_harm_signals(message) == []


@pytest.mark.parametrize("message", SELF_HARM)
def test_self_harm_gets_fixed_reply_without_symptom_block(env, message: str) -> None:
    out = _send(env, message)
    assert env.llm.calls == 0  # MedGemma 증상 정리 틀을 거치지 않는다
    assert set(out) == RESPONSE_KEYS
    assert "· 증상:" not in out["reply"]
    assert "119" in out["reply"] and "109" in out["reply"]
    assert out["riskLevel"] == "high"
    assert out["intent"] == "emergency_guidance"
    assert out["emergencyWarning"] and "109" in out["emergencyWarning"][0]


@pytest.mark.parametrize("message", ["이제 그만 살 빼야지", "밥 다 먹어버릴까", "무릎 아파 죽겠어"])
def test_everyday_phrases_do_not_get_crisis_reply(env, message: str) -> None:
    out = _send(env, message)
    assert env.llm.calls == 1
    assert "109" not in out["reply"]
    assert not any("109" in w for w in out["emergencyWarning"])


def test_self_harm_wins_over_router_non_emergency_and_department_question(env) -> None:
    history = [{"role": "user", "content": "머리가 아파요"}, {"role": "assistant", "content": "신경과 진료를 권해드려요."}]
    out = _send(env, "죽고 싶은데 무슨 과로 가야 해", history)
    assert "109" in out["reply"] and env.llm.calls == 0


def test_general_emergency_keeps_existing_path(env) -> None:
    _send(env, "가슴이 조이고 식은땀이 나요")
    assert env.llm.calls == 1  # 일반 응급은 기존 MedGemma 응급 경로 그대로


def test_self_harm_log_has_no_message_text(env, caplog) -> None:
    with caplog.at_level(logging.INFO):
        _send(env, "살기 싫어 홍길순 010-5555-6666")
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "[CHAT-SAFETY] self_harm=true path=main" in joined
    assert "살기" not in joined and "홍길순" not in joined and "010-5555" not in joined


def test_fallback_path_also_returns_fixed_self_harm_reply(env) -> None:
    out = env.svc._fallback_reply("그만 살고 싶어", [])
    assert "109" in out["reply"] and out["riskLevel"] == "high" and out["emergencyWarning"]


# ---- C-3 폴백 응급 경고 ------------------------------------------------------------------------------


def test_fallback_emergency_fills_warning(env) -> None:
    env.llm = _Llm(fail=True)
    out = _send(env, "가슴이 너무 아파서 숨을 못 쉬겠어요")
    assert out["engine"] == "fallback"
    assert out["emergencyWarning"] == [MedicalLlmService.EMERGENCY_WARNING]


def test_fallback_router_only_emergency_fills_warning(env) -> None:
    env.llm, env.router = _Llm(fail=True), _Router(emergency=True)
    out = _send(env, "엄마가 아까부터 좀 이상해요 대답을 잘 못하세요")
    assert out["riskLevel"] == "high" and out["emergencyWarning"] == [MedicalLlmService.EMERGENCY_WARNING]


def test_fallback_non_emergency_has_no_warning(env) -> None:
    env.llm = _Llm(fail=True)
    out = _send(env, "어제부터 기침이 계속 나요")
    assert out["emergencyWarning"] == []


# ---- C-2 약 문의 + 증상 병원 권유 --------------------------------------------------------------------


class _Loader:
    def __init__(self, advice: str) -> None:
        self.advice = advice
        self.state = SimpleNamespace(model_name="fake-medgemma")

    def ensure_loaded(self) -> None:
        return None

    def generate_structured_text(self, system_prompt: str, user_prompt: str, prefill: str = "") -> str:  # noqa: ARG002
        return json.dumps({"empathy": "걱정되셨겠어요.", "symptoms": "약 복용 문의", "features": "",
                           "advice": self.advice, "reply": ""}, ensure_ascii=False)


def _generate(message: str, advice: str) -> dict[str, Any]:
    return MedicalLlmService(Settings(), _Loader(advice)).generate(message, {}, [], intent="medical_advice")


CLINIC = "증상이 계속되거나 걱정되시면 병원 진료를 받아 보세요."


def test_medication_question_with_symptom_gets_clinic_sentence() -> None:
    out = _generate("열이 나는데 병원 가지 말고 남은 항생제를 먹어도 되죠?", "복용 전에 의사나 약사에게 문의해 주세요.")
    assert out["reply"].endswith(CLINIC)


def test_clinic_sentence_not_added_when_reply_already_recommends_clinic() -> None:
    out = _generate("무릎이 너무 아파서 진통제를 몇 알까지 먹어도 돼?", "약사에게 묻고 정형외과 진료를 받아 보세요.")
    assert CLINIC not in out["reply"]


@pytest.mark.parametrize("message", ["혈압약을 깜빡하고 안 먹었어요", "수면제를 약국에서 그냥 살 수 있나요", "어제부터 기침이 나요"])
def test_clinic_sentence_only_for_medication_question_with_symptom(message: str) -> None:
    out = _generate(message, "의사나 약사에게 물어봐 주세요.")
    assert CLINIC not in out["reply"]


def test_clinic_sentence_not_added_on_emergency() -> None:
    out = MedicalLlmService(Settings(), _Loader("x")).generate(
        "가슴이 아픈데 진통제 먹어도 돼?", {}, [], intent="emergency_guidance")
    assert CLINIC not in out["reply"]


# ---- C-4 진료과 키워드 --------------------------------------------------------------------------------


def test_dizzy_conjugation_maps_to_neurology() -> None:
    assert _symptom_department("요즘 일어날 때 어지럽고 눈앞이 캄캄해져.") == "신경과"
