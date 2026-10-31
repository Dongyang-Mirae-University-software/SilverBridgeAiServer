"""증상 상담 되묻기(2026-10-29 C-1, 설계 docs/design/2026-10-29-chat-followup-question.md A안).
정보가 부족한 일반 증상 한마디에만 한 번 되묻고, 다음 턴은 앞 질문과 이어 답한다.
응급·자해·약·약 + 술·생활관리·예약·위기 뒤·응급 다음 턴·되묻기 다음 턴에는 되묻지 않는다."""
from __future__ import annotations

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

INTRO = "말씀해 주셔서 고마워요. 더 정확히 안내해 드리려고 몇 가지 여쭤볼게요."


class _Router:
    def __init__(self) -> None:
        self.route_value = "symptom"
        self.emergency = False

    def route(self, message: str, history, ctx, prev_reservation=None) -> dict[str, Any]:  # noqa: ARG002
        return {"route": self.route_value, "normalized_message": message, "symptom_summary": message,
                "emergency": self.emergency, "self_harm": False, "reason": "t", "engine": "fake"}


class _Llm:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, message, ctx, history, intent=None, processed_symptom=None, proxy=False) -> dict[str, Any]:  # noqa: ARG002
        self.calls.append({"message": message, "intent": intent, "processed_symptom": processed_symptom, "proxy": proxy})
        risk = "high" if intent == "emergency_guidance" else "medium"  # 실제 서비스처럼 응급 의도면 high
        return {"reply": "신경과 진료를 권해드려요.", "riskLevel": risk, "recommendedAction": "guardian_contact",
                "reservationRequired": False, "intent": intent, "engine": "medgemma", "modelName": "m",
                "summary": "s", "possibleCauses": [], "homeCare": [], "visitHospitalIf": [], "emergencyWarning": []}


@pytest.fixture()
def env(monkeypatch):
    engine = create_engine(f"sqlite:///{tempfile.mkdtemp(prefix='sb_chat_followup_')}/t.db")
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


def _send(env, message: str, session: str = "s1", ward: str | None = None) -> dict[str, Any]:
    ctx = ChatContext(name="홍길동", wardId=ward) if ward else ChatContext(name="홍길동")
    return env.svc.process_message(env.db, ChatRequest(userId="user-1", message=message, sessionId=session, context=ctx))


# ---- 문장 조건 -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("message", ["어지러워", "기침이 나요", "머리가 아파요", "배가 아파요", "허리가 아파요", "속이 안 좋아요", "귀가 아파요",
                                     "피부가 가려워요", "가려워요"])
def test_short_symptom_needs_followup(message: str) -> None:
    assert MedicalLlmService.needs_followup_question(message)


@pytest.mark.parametrize("message", [
    "며칠째 기침이 나고 목이 아파요",  # 기간
    "어제부터 기침이 나요",
    "아침에 일어나면 어지러워요",  # 상황
    "머리가 너무 아파요",  # 정도
    "열이 38도예요",  # 숫자
    "배가 아프고 설사해요",  # 동반 증상
    "가슴이 아파요",  # 응급 키워드
    "가슴이 답답해요",  # 위험 신호 단어 - 119 한 줄이 나가는 상담으로 둔다(v7 bd.02)
    "숨이 차요",
    "식은땀이 나요",
    "죽고 싶어요",  # 자해
    "혈압약 먹었어요",  # 약
    "관절염이 있는데 어떤 운동이 좋아요",  # 생활관리
    "머리가 아픈데 무슨 과 가요?",  # 진료과 질문
    "오늘 날씨가 좋네요",  # 증상 없음
    "무릎 아파 죽겠어",  # 감정·괴로움 표현 - 위기 신호일 수 있어 라우터·MedGemma 로
    "무릎이 아파서 다 그만두고 싶어",
    "배가 아파서 힘들어요",
])
def test_other_sentences_do_not_need_followup(message: str) -> None:
    assert not MedicalLlmService.needs_followup_question(message)


# ---- 대화 흐름 -----------------------------------------------------------------------------------------


def test_first_turn_asks_without_calling_medgemma(env) -> None:
    out = _send(env, "머리가 아파요")
    assert out["reply"].startswith(INTRO)
    assert "· 언제부터 그러셨나요?" in out["reply"]
    assert "신경과 진료를 생각해 보셔도 좋아요." in out["reply"]  # 진료과 한 줄(Q2)
    assert "119" not in out["reply"] and not out["emergencyWarning"] and out["riskLevel"] == "medium"
    assert env.llm.calls == []


def test_next_turn_answers_with_previous_question(env) -> None:
    _send(env, "머리가 아파요")
    env.router.route_value = "general"  # 짧은 답을 라우터가 일반 대화로 보내도
    out = _send(env, "어제부터요, 열도 조금 있어요")
    assert len(env.llm.calls) == 1
    call = env.llm.calls[0]
    assert call["message"] == "머리가 아파요 / 어제부터요, 열도 조금 있어요"
    assert call["intent"] == "medical_advice"
    assert INTRO not in out["reply"]


def test_never_asks_twice_in_a_row(env) -> None:
    _send(env, "머리가 아파요")
    out = _send(env, "배가 아파요")  # 또 짧은 증상이어도 연속으로 묻지 않는다
    assert INTRO not in out["reply"] and len(env.llm.calls) == 1


def test_thanks_after_followup_is_not_joined(env) -> None:
    _send(env, "머리가 아파요")
    env.router.route_value = "general"
    _send(env, "고마워요")
    assert env.llm.calls[0]["message"] == "고마워요" and env.llm.calls[0]["intent"] == "general_chat"


def test_proxy_question_wording(env) -> None:
    out = _send(env, "기침이 나요", ward="ab12")
    assert "· 어르신께서 언제부터 그러셨나요?" in out["reply"]
    out = _send(env, "엄마가 어지러워해요", session="s2")
    assert "· 어르신께서 언제부터 그러셨나요?" in out["reply"]


def test_router_emergency_never_asks(env) -> None:
    env.router.emergency = True
    out = _send(env, "머리가 아파요")
    assert INTRO not in out["reply"] and len(env.llm.calls) == 1
    assert env.llm.calls[0]["intent"] == "emergency_guidance"


def test_reservation_and_general_routes_never_ask(env) -> None:
    env.router.route_value = "general"
    out = _send(env, "머리가 아파요")
    assert INTRO not in out["reply"]


def test_followup_meta_is_recorded(env) -> None:
    import json
    _send(env, "머리가 아파요")
    row = env.db.query(ChatLog).order_by(ChatLog.id.desc()).first()
    meta = json.loads(row.context_json)["upstreamMeta"]
    assert meta.get("followupAsked") is True


def test_different_session_or_ward_does_not_continue(env) -> None:
    _send(env, "머리가 아파요", session="s1")
    out = _send(env, "배가 아파요", session="s9")  # 다른 세션 - 되묻기 다음 턴이 아니라 새 질문
    assert out["reply"].startswith(INTRO)


def test_followup_log_has_flags_only(env, caplog) -> None:
    import logging
    caplog.set_level(logging.INFO)
    _send(env, "머리가 아파요")
    msgs = [r.getMessage() for r in caplog.records if "followup_asked" in r.getMessage()]
    assert msgs == ["[CHAT-QUALITY] followup_asked=true proxy=False"]


def test_no_followup_right_after_crisis_reply(env) -> None:
    out = _send(env, "죽고 싶어요")  # 위기 고정 응답(키워드)
    assert out["riskLevel"] == "high"
    out = _send(env, "머리가 아파요")  # 위기 직후 첫 일반 턴 - 109 한 줄 + 상담, 되묻지 않는다
    assert INTRO not in out["reply"] and "109" in out["reply"]
    assert len(env.llm.calls) == 1


def test_no_followup_right_after_emergency_reply(env) -> None:
    env.router.emergency = True
    _send(env, "갑자기 머리가 깨질 듯 아파요")
    env.router.emergency = False
    env.llm.calls.clear()
    out = _send(env, "머리가 아파요")  # 응급 답 다음 턴은 되묻지 않는다
    assert INTRO not in out["reply"] and len(env.llm.calls) == 1


# ---- 대리 상담 여부 전달(2026-10-31 구조 정리 B) --------------------------------------------------------------


@pytest.mark.parametrize("message, ward, expected", [
    ("며칠째 기침이 나고 목이 아파요", None, False),
    ("며칠째 기침이 나고 목이 아파요", "ward-1", True),  # 보호자 대리 상담 화면(wardId)
    ("어머니가 며칠째 기침을 하고 목이 아프세요", None, True),  # 가족이 주어
])
def test_generation_gets_proxy_flag(env, message: str, ward: str | None, expected: bool) -> None:
    _send(env, message, ward=ward)
    assert env.llm.calls and env.llm.calls[-1]["proxy"] is expected


def test_followup_answer_turn_uses_previous_question_for_proxy(env) -> None:
    # 되묻기에 답한 턴은 "앞 질문 / 이번 답"으로 대리 판정한다(2026-11-01 점검 M-1).
    _send(env, "어머니가 기침을 하세요")
    assert env.llm.calls == []  # 첫 턴은 되묻기(MedGemma 안 부름)
    _send(env, "어제부터요")
    assert env.llm.calls and env.llm.calls[-1]["proxy"] is True
