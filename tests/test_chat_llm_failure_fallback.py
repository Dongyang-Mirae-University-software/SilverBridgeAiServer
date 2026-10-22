"""MedGemma 생성이 실패하면 옛 챗 서버로 넘기지 않고 고정 문구 폴백으로 끝나는지(2026-10-22 S-5)."""
from __future__ import annotations

import importlib.util
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

# 키워드 층에는 걸리지 않고 라우터(LLM)만 응급으로 판단할 수 있는 문장
ROUTER_ONLY_EMERGENCY = "엄마가 아까부터 좀 이상해요 대답을 잘 못하세요"


class _Router:
    def __init__(self, emergency: bool) -> None:
        self.emergency = emergency

    def route(self, message: str, history, ctx, prev_reservation=None) -> dict[str, Any]:  # noqa: ARG002
        return {"route": "symptom", "normalized_message": message, "symptom_summary": message,
                "emergency": self.emergency, "reason": "t", "engine": "fake"}


class _FailingLlm:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *a, **k) -> dict[str, Any]:  # noqa: ARG002
        self.calls += 1
        raise RuntimeError("민감한 상담 문장이 섞인 예외 원문")


@pytest.fixture()
def env(monkeypatch):
    engine = create_engine(f"sqlite:///{tempfile.mkdtemp(prefix='sb_chat_fb_')}/t.db")
    ChatLog.__table__.create(bind=engine)
    db = sessionmaker(bind=engine)()
    llm = _FailingLlm()
    router = _Router(emergency=False)
    monkeypatch.setattr(cs, "get_chat_router_service", lambda: router)
    monkeypatch.setattr(cs, "get_reservation_credential_service", lambda: object())
    monkeypatch.setattr(cs, "get_reservation_api_client", lambda: object())
    monkeypatch.setattr(cs, "get_medical_llm_service", lambda: llm)
    svc = cs.ChatService()
    svc._settings = SimpleNamespace(has_reservation_account_secret=True, claude_model_name="fake")
    yield SimpleNamespace(svc=svc, db=db, llm=llm, router=router)
    db.close()


def _send(env, message: str) -> dict[str, Any]:
    payload = ChatRequest(userId="user-1", message=message, sessionId="s1", context=ChatContext(name="홍길동"))
    return env.svc.process_message(env.db, payload)


def test_upstream_fallback_module_is_gone() -> None:
    # 옛 챗 서버(127.0.0.1:6012)로 보내는 경로 자체가 없어야 한다
    assert importlib.util.find_spec("app.services.chat_upstream_service") is None
    assert not hasattr(cs, "get_chat_upstream_service")


def test_llm_failure_returns_fixed_fallback(env) -> None:
    out = _send(env, "어제부터 기침이 계속 나요")
    assert env.llm.calls == 1
    assert out["engine"] == "fallback"
    assert "진료가 필요할 수 있습니다" in out["reply"]


def test_llm_failure_on_keyword_emergency_still_guides_119(env) -> None:
    out = _send(env, "가슴이 너무 아파서 숨을 못 쉬겠어요")
    assert out["engine"] == "fallback"
    assert "119" in out["reply"]
    assert out["riskLevel"] == "high"


def test_llm_failure_on_router_only_emergency_still_guides_119(env) -> None:
    # 전제: 키워드만으로는 응급이 아니다(라우터가 잡아야 하는 문장)
    assert cs.ChatService.classify_intent(ROUTER_ONLY_EMERGENCY) != "emergency_guidance"
    env.router.emergency = True
    out = _send(env, ROUTER_ONLY_EMERGENCY)
    assert out["engine"] == "fallback"
    assert "119" in out["reply"]
    assert out["riskLevel"] == "high"


def test_llm_failure_log_has_exception_class_only(env, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger=cs.__name__):
        _send(env, "어제부터 기침이 계속 나요")
    failed = [r.getMessage() for r in caplog.records if "[CHAT-LLM-FAILED]" in r.getMessage()]
    assert failed == ["[CHAT-LLM-FAILED] RuntimeError"]
    assert all("기침" not in r.getMessage() and "예외 원문" not in r.getMessage() for r in caplog.records)
