"""챗봇 라우터 점검 후속(2026-10-25, BE docs/(2026-10-24) audit-ai-chat-router-stability.md H-1·M-1·M-2·I-1)."""
from __future__ import annotations

import logging
import tempfile
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core import config
from app.core.config import Settings
from app.models.chat_log import ChatLog
from app.schemas.chat_schema import ChatContext, ChatRequest
from app.services import chat_service as cs
from app.services.chat_router_service import ChatRouterService
from app.services.claude_structured_client import ClaudeStructuredClient, StructuredCallError, failure_cause
from app.services.reservation_api_client import ReservationApiClient
from app.services.reservation_common import _trace, log_len


# ---- H-1 라우터·예약 추출 출력 상한 ---------------------------------------------------------------------


def test_claude_max_tokens_default_and_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLAUDE_MAX_TOKENS", raising=False)
    assert config._as_int_in_range("CLAUDE_MAX_TOKENS", 2048, 256, 8192) == 2048
    monkeypatch.setenv("CLAUDE_MAX_TOKENS", "4096")
    assert config._as_int_in_range("CLAUDE_MAX_TOKENS", 2048, 256, 8192) == 4096
    for bad in ("100", "99999", "abc"):
        monkeypatch.setenv("CLAUDE_MAX_TOKENS", bad)
        assert config._as_int_in_range("CLAUDE_MAX_TOKENS", 2048, 256, 8192) == 2048


def test_claude_request_uses_its_own_limit_not_medgemma() -> None:
    settings = Settings(anthropic_api_key="test-key", max_new_tokens=512, claude_max_tokens=2048)
    client = ClaudeStructuredClient(settings, "claude-opus-5-5")
    assert client._request_kwargs("s", "u", None)["max_tokens"] == 2048


class _FakeResp(SimpleNamespace):
    pass


def _client_returning(stop_reason: str, text: str) -> ClaudeStructuredClient:
    client = ClaudeStructuredClient(Settings(anthropic_api_key="test-key"), "claude-opus-5-5")
    blocks = [SimpleNamespace(type="text", text=text)] if text else []
    resp = _FakeResp(stop_reason=stop_reason, content=blocks)
    client._client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=lambda **k: resp)))
    return client


@pytest.mark.parametrize(("stop", "text", "cause"), [
    ("max_tokens", '{"route": "sym', "max_tokens"),
    ("refusal", "", "refusal"),
    ("end_turn", "", "empty"),
])
def test_claude_failures_carry_fixed_cause(stop: str, text: str, cause: str) -> None:
    with pytest.raises(StructuredCallError) as info:
        _client_returning(stop, text).generate_structured_text("s", "u")
    assert info.value.cause == cause and failure_cause(info.value) == cause


def test_not_ready_cause() -> None:
    client = ClaudeStructuredClient(Settings(anthropic_api_key=""), "claude-opus-5-5")
    with pytest.raises(StructuredCallError) as info:
        client.generate_structured_text("s", "u")
    assert info.value.cause == "not_ready"


class _Loader:
    def __init__(self, raw: str) -> None:
        self.raw = raw
        self.state = SimpleNamespace(model_name="fake")

    def generate_structured_text(self, *a, **k) -> str:  # noqa: ARG002
        return self.raw


@pytest.mark.parametrize(("raw", "cause"), [("not json", "parse"), ('{"route": "weird"}', "unknown_route")])
def test_router_parse_failures_carry_cause(raw: str, cause: str) -> None:
    with pytest.raises(StructuredCallError) as info:
        ChatRouterService(_Loader(raw), 8).route("x", [], {})
    assert info.value.cause == cause


def test_router_prompt_keeps_reason_short() -> None:
    from app.services.chat_router_service import SYSTEM_PROMPT
    assert "reason: 판단 근거 한 문장" in SYSTEM_PROMPT


# ---- 챗 서비스: 라우터 실패 로그(M-1)·위기 턴 뒤 예약 재개(M-2) ------------------------------------------


class _Router:
    def __init__(self, emergency: bool = False, summary: str | None = None, fail: str | None = None) -> None:
        self.emergency, self.summary, self.fail, self.calls = emergency, summary, fail, 0

    def route(self, message: str, history, ctx, prev_reservation=None) -> dict[str, Any]:  # noqa: ARG002
        self.calls += 1
        if self.fail:
            raise StructuredCallError(self.fail)
        return {"route": "symptom", "normalized_message": message, "symptom_summary": self.summary or message,
                "emergency": self.emergency, "reason": "t", "engine": "claude"}


class _Llm:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *a, **k) -> dict[str, Any]:  # noqa: ARG002
        self.calls += 1
        return {"reply": "모델 답변", "riskLevel": "medium", "recommendedAction": "guardian_contact", "reservationRequired": False,
                "intent": "medical_advice", "engine": "medgemma", "modelName": "m", "summary": "s", "possibleCauses": [],
                "homeCare": [], "visitHospitalIf": [], "emergencyWarning": []}


class _Orchestrator:
    def __init__(self) -> None:
        self.calls = 0

    def handle(self, req, meta) -> dict[str, Any]:  # noqa: ARG002
        self.calls += 1
        return {"reply": "어느 진료과를 찾을까요?", "riskLevel": "medium", "recommendedAction": "guardian_contact",
                "reservationRequired": True, "intent": "hospital_reservation", "engine": "claude", "modelName": "fake",
                "type": "message", "ui": None, "tool": None, "data": None, "summary": None, "possibleCauses": [],
                "homeCare": [], "visitHospitalIf": [], "emergencyWarning": []}


@pytest.fixture()
def env(monkeypatch):
    engine = create_engine(f"sqlite:///{tempfile.mkdtemp(prefix='sb_chat_router_')}/t.db")
    ChatLog.__table__.create(bind=engine)
    db = sessionmaker(bind=engine)()
    holder = SimpleNamespace(llm=_Llm(), router=_Router(), orch=_Orchestrator())
    monkeypatch.setattr(cs, "get_chat_router_service", lambda: holder.router)
    monkeypatch.setattr(cs, "get_reservation_credential_service", lambda: SimpleNamespace(ensure_credential=lambda *a, **k: None))
    monkeypatch.setattr(cs, "get_reservation_api_client", lambda: object())
    monkeypatch.setattr(cs, "get_medical_llm_service", lambda: holder.llm)
    monkeypatch.setattr(cs, "get_reservation_orchestrator", lambda: holder.orch)
    svc = cs.ChatService()
    svc._settings = SimpleNamespace(has_reservation_account_secret=True, claude_model_name="fake")
    holder.svc, holder.db = svc, db
    yield holder
    db.close()


def _send(env, message: str, session: str = "s1", history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    payload = ChatRequest(userId="user-1", message=message, sessionId=session, context=ChatContext(name="홍길동"),
                          history=history or [])
    return env.svc.process_message(env.db, payload)


def test_router_failure_logs_fixed_cause_only(env, caplog) -> None:
    env.router = _Router(fail="max_tokens")
    with caplog.at_level(logging.WARNING):
        _send(env, "요즘 잠이 안 와요 홍길순 010-5555-6666")
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "[CHAT-ROUTER-FAILED] cause=max_tokens fallback=keywords" in joined
    assert "홍길순" not in joined and "010-5555" not in joined and "잠이" not in joined


CRISIS_HISTORY = [{"role": "user", "content": "죽고 싶어"}, {"role": "assistant", "content": "많이 힘드시군요."}]


def _crisis_then(env, message: str, session: str = "s1", **router_kwargs) -> dict[str, Any]:
    first = _send(env, "죽고 싶어", session)  # 키워드 → 위기 고정 응답(기록에 selfHarm)
    assert "109" in first["reply"]
    env.router = _Router(**({"emergency": True, "summary": "자살 위기 이후 대화"} | router_kwargs))
    return _send(env, message, session, CRISIS_HISTORY)


def test_resume_reservation_after_crisis(env, caplog) -> None:
    with caplog.at_level(logging.INFO):
        out = _crisis_then(env, "응 예약해줘")
    assert env.orch.calls == 1
    assert out["reply"].startswith(cs._CRISIS_FOLLOWUP)
    assert out["intent"] == "hospital_reservation"
    assert "resume_reservation_after_crisis=true trigger=history" in "\n".join(r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("message", ["가슴이 답답해서 예약해줘", "응급실 예약해줘", "죽고 싶어 예약해줘", "고마워요"])
def test_no_resume_when_symptom_emergency_selfharm_or_no_reservation_intent(env, message: str) -> None:
    out = _crisis_then(env, message)
    assert env.orch.calls == 0
    assert "109" in out["reply"]  # 위기 응답 유지


def test_no_resume_when_previous_turn_was_not_crisis(env) -> None:
    _send(env, "어제부터 기침이 나요")
    env.router = _Router(emergency=True, summary="자살 위기 표현")
    out = _send(env, "응 예약해줘", "s1", CRISIS_HISTORY)
    assert env.orch.calls == 0 and "109" in out["reply"]


def test_no_resume_across_sessions(env) -> None:
    _send(env, "죽고 싶어", "other-session")
    env.router = _Router(emergency=True, summary="자살 위기 이후 대화")
    out = _send(env, "응 예약해줘", "s1", CRISIS_HISTORY)
    assert env.orch.calls == 0 and "109" in out["reply"]


# ---- I-1 예약 로그 원문 ------------------------------------------------------------------------------------


def test_trace_log_masks_user_values_but_keeps_decision_trace(caplog) -> None:
    meta: dict[str, Any] = {}
    with caplog.at_level(logging.INFO, logger="app.services.reservation_orchestrator"):
        _trace(meta, "reservation_draft", location="서울 강남구", hospital_name="행복내과의원", department="내과",
               reservation_date="2026-10-30", reservation_time="14:00", tried_regions=["서울 강남구", "서울 서초구"])
    line = "\n".join(r.getMessage() for r in caplog.records)
    for raw in ("강남구", "행복내과", "2026-10-30", "14:00", "서초구"):
        assert raw not in line
    assert "내과" in line and "len=" in line and "count=2" in line
    assert meta["decisionTrace"][0]["location"] == "서울 강남구"  # DB 기록은 그대로


def test_log_len() -> None:
    assert log_len(None) == "(empty)" and log_len("") == "(empty)" and log_len([]) == "(empty)"
    assert log_len("서울 강남구") == "len=6" and log_len(["a", "b"]) == "count=2"


def test_search_log_masks_region_name_date(caplog) -> None:
    client = ReservationApiClient.__new__(ReservationApiClient)
    client._request = lambda *a, **k: (200, {"success": True, "data": []})  # type: ignore[method-assign]
    client._resolve_api_key = lambda *a, **k: "k"  # type: ignore[method-assign]
    with caplog.at_level(logging.INFO, logger="app.services.reservation_api_client"):
        try:
            client.search_hospitals("서울 강남구", "내과", name="행복내과의원", slot_date="2026-10-30")
        except Exception:  # noqa: BLE001 - 응답 파싱이 달라도 로그 줄은 이미 남았다
            pass
    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "[예약검색]" in line
    for raw in ("강남구", "행복내과", "2026-10-30"):
        assert raw not in line
