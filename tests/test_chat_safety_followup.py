"""챗봇 안전 규칙 점검 후속(2026-10-24, BE docs/(2026-10-24) audit-ai-chat-safety-rules.md M-1·L-1·L-2·L-3·I-1·I-2)."""
from __future__ import annotations

import logging
import tempfile
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.chat_log import ChatLog
from app.schemas.chat_schema import ChatContext, ChatRequest
from app.services import chat_service as cs
from app.services.medical_llm_service import MedicalLlmService

NEW_SELF_HARM = ["자해하고 싶어", "살기가 싫어", "사는 게 싫어", "살고 싶지가 않아", "목숨 끊고 싶어", "뛰어내리고 싶어",
                 "약을 다 먹을까", "수면제를 다 먹어버릴까", "수면제를 한꺼번에 먹으면", "농약 마시고 싶어", "극단적인 생각이 들어"]
# 일상 표현 - 위기 응답이 나오면 안 된다
EVERYDAY = ["극단적인 다이어트 해도 돼?", "처방받은 약을 다 먹었어요", "수면제를 다 먹어서 다시 받아야 해요",
            "농약 뿌리고 나서 손이 따가워요", "이제 그만 살 빼야지", "밥 다 먹어버릴까", "무릎 아파 죽겠어", "다 그만두고 싶어"]


class _Router:
    def __init__(self, emergency: bool = False, summary: str | None = None) -> None:
        self.emergency, self.summary, self.calls = emergency, summary, 0

    def route(self, message: str, history, ctx, prev_reservation=None) -> dict[str, Any]:  # noqa: ARG002
        self.calls += 1
        return {"route": "symptom", "normalized_message": message, "symptom_summary": self.summary or message,
                "emergency": self.emergency, "reason": "t", "engine": "claude"}


class _Llm:
    def __init__(self, fail: bool = False) -> None:
        self.calls, self.fail = 0, fail

    def generate(self, *a, **k) -> dict[str, Any]:  # noqa: ARG002
        self.calls += 1
        if self.fail:
            raise RuntimeError("x")
        return {"reply": "모델 답변", "riskLevel": "high", "recommendedAction": "hospital_visit", "reservationRequired": True,
                "intent": "emergency_guidance", "engine": "medgemma", "modelName": "m", "summary": "s", "possibleCauses": [],
                "homeCare": [], "visitHospitalIf": [], "emergencyWarning": ["경고"]}


@pytest.fixture()
def env(monkeypatch):
    engine = create_engine(f"sqlite:///{tempfile.mkdtemp(prefix='sb_chat_safety2_')}/t.db")
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


def _is_crisis(out: dict[str, Any]) -> bool:
    return "109" in out["reply"] and "· 증상:" not in out["reply"] and out["riskLevel"] == "high"


# ---- M-1 ① 키워드 보강 / L-1 과잉 키워드 -----------------------------------------------------------------


@pytest.mark.parametrize("message", NEW_SELF_HARM)
def test_new_self_harm_keywords(message: str) -> None:
    assert MedicalLlmService.self_harm_signals(message)


@pytest.mark.parametrize("message", EVERYDAY)
def test_everyday_phrases_are_not_self_harm(message: str) -> None:
    assert MedicalLlmService.self_harm_signals(message) == []


def test_extreme_diet_is_not_emergency_any_more() -> None:
    assert MedicalLlmService.emergency_signals("극단적인 다이어트 해도 돼?") == []
    assert MedicalLlmService.emergency_signals("극단적인 선택을 할까 봐요")


@pytest.mark.parametrize("message", ["극단적인 다이어트 해도 돼?", "처방받은 약을 다 먹었어요"])
def test_everyday_phrases_get_normal_reply(env, message: str) -> None:
    out = _send(env, message)
    assert env.llm.calls == 1 and "109" not in out["reply"]


# ---- I-1 라우터 생략 / I-2 예약 버튼 값 -------------------------------------------------------------------


def test_keyword_self_harm_skips_router(env, caplog) -> None:
    with caplog.at_level(logging.INFO):
        out = _send(env, "자해하고 싶어")
    assert env.router.calls == 0 and env.llm.calls == 0
    assert _is_crisis(out)
    assert out["reservationRequired"] is False
    assert "trigger=keyword" in "\n".join(r.getMessage() for r in caplog.records)


# ---- M-1 ③ 라우터 요약 -----------------------------------------------------------------------------------


def test_router_emergency_with_self_harm_summary_gets_crisis_reply(env) -> None:
    env.router = _Router(emergency=True, summary="자살 의도가 의심되는 위기 표현")
    out = _send(env, "다 끝내고 싶어")
    assert env.router.calls == 1 and env.llm.calls == 0
    assert _is_crisis(out)


def test_negated_self_harm_summary_is_ignored(env) -> None:
    env.router = _Router(emergency=True, summary="가슴 통증, 자해 의도 없음")
    _send(env, "가슴이 조이고 식은땀이 나요")
    assert env.llm.calls == 1  # 일반 응급 경로


def test_non_emergency_summary_with_self_harm_word_is_ignored(env) -> None:
    env.router = _Router(emergency=False, summary="무릎 통증으로 일상이 힘듦, 자해")
    _send(env, "무릎이 아파서 다 그만두고 싶어")
    assert env.llm.calls == 1


# ---- M-1 ② 다중 턴 ---------------------------------------------------------------------------------------

PREV = [{"role": "user", "content": "죽고 싶어요"}, {"role": "assistant", "content": "많이 힘드시군요."}]


def test_history_self_harm_with_emergency_turn_gets_crisis_reply(env) -> None:
    env.router = _Router(emergency=True, summary="정서적 고통")
    out = _send(env, "어떻게 해야 할지 모르겠어", PREV)
    assert _is_crisis(out) and env.llm.calls == 0


def test_history_self_harm_without_emergency_turn_goes_normal(env) -> None:
    env.router = _Router(emergency=False, summary="일반 대화")
    _send(env, "고마워요", PREV)
    assert env.llm.calls == 1  # 앞 턴 때문에 위기 응답에 가두지 않는다


def test_fallback_uses_history_when_emergency(env) -> None:
    out = env.svc._fallback_reply("어떻게 해야 할지 모르겠어", PREV, emergency=True)
    assert _is_crisis(out)
    out = env.svc._fallback_reply("어떻게 해야 할지 모르겠어", PREV, emergency=False)
    assert "109" not in out["reply"]


# ---- L-2 가족(제3자) 문구 ---------------------------------------------------------------------------------


@pytest.mark.parametrize("message", ["엄마가 죽고 싶다고 하세요", "아버지께서 자살 얘기를 하세요", "남편이 살기가 싫대요"])
def test_family_subject_gets_family_reply(env, message: str) -> None:
    out = _send(env, message)
    assert out["reply"].startswith("많이 놀라셨겠어요.")
    assert "가족도 24시간" in out["reply"] and "119" in out["reply"]
    assert out["emergencyWarning"] and "109" in out["emergencyWarning"][0]


def test_own_crisis_from_guardian_gets_personal_reply(env) -> None:
    payload = ChatRequest(userId="g-1", message="나도 죽고 싶어요", sessionId="s2",
                          context=ChatContext(name="보호자", role="GUARDIAN", wardId="w1"))
    out = env.svc.process_message(env.db, payload)
    assert out["reply"].startswith("많이 힘드시군요.")  # 역할값이 아니라 주어로 가른다


# ---- L-3 기록 응답의 emergencyWarning ---------------------------------------------------------------------


def test_chat_log_responses_include_emergency_warning(env) -> None:
    _send(env, "죽고 싶어요")
    _send(env, "어제부터 기침이 나요")
    items = cs.ChatService.list_logs(env.db, "user-1")
    by_message = {item["message"]: item for item in items}
    assert "109" in by_message["죽고 싶어요"]["emergencyWarning"][0]
    assert by_message["어제부터 기침이 나요"]["emergencyWarning"] == ["경고"]
    one = cs.ChatService.get_log(env.db, by_message["죽고 싶어요"]["id"], "user-1")
    assert one["emergencyWarning"] == by_message["죽고 싶어요"]["emergencyWarning"]
