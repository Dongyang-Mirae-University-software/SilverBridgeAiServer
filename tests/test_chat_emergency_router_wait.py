"""응급 키워드 문장의 라우터 대기 상한(2026-10-29 점검 L-4): 라우터를 건너뛰지 않되 15초까지만 기다린다.
넘기면 라우터 실패 경로(키워드 분류 → 119 안내). 일반 문장은 그대로. 시간은 가짜 시계로 흘린다."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.schemas.chat_schema import ChatContext, ChatRequest
from app.services import chat_deadline as dl
from app.services import chat_service as cs
from app.services import claude_structured_client as csc
from app.services.chat_deadline import chat_deadline, remaining_sec, tighter_deadline
from app.services.chat_router_service import ChatRouterService
from tests.test_chat_privacy_deadline import _Clock, _claude, _SlowAnthropic
from tests.test_reservation_confirm_chat_service import env  # noqa: F401


class _JsonAnthropic(_SlowAnthropic):
    def __init__(self, clock: _Clock, slow: float, payload: dict) -> None:
        super().__init__(clock, slow)
        self.payload = payload

    def _create(self, **kwargs):
        out = super()._create(**kwargs)
        out.content[0].text = json.dumps(self.payload)
        return out


ROUTE = {"route": "symptom", "normalized_message": "", "symptom_summary": "", "emergency": True, "self_harm": False, "reason": "t"}


@pytest.fixture()
def clock(monkeypatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(dl, "time", SimpleNamespace(perf_counter=c))
    monkeypatch.setattr(csc, "time", SimpleNamespace(sleep=c.advance))
    return c


def _router(monkeypatch, anthropic_client) -> None:
    client = _claude()
    client._client = anthropic_client
    router = ChatRouterService(client, keep_turns=8)
    monkeypatch.setattr(cs, "get_chat_router_service", lambda: router)
    monkeypatch.setattr(cs, "chat_deadline", lambda: chat_deadline(110))


def _send(env, message: str) -> dict:  # noqa: F811
    payload = ChatRequest(userId="user-1", message=message, sessionId="s1", context=ChatContext(name="홍길동"))
    return env.svc.process_message(env.db, payload)


def test_emergency_keyword_waits_router_at_most_15s(env, monkeypatch, clock) -> None:  # noqa: F811
    slow = _SlowAnthropic(clock, slow=500)
    _router(monkeypatch, slow)
    out = _send(env, "숨을 못 쉬겠어요")
    assert slow.timeouts == [15]  # 재시도할 시간이 남지 않는다
    assert clock.now - 1000 <= 16
    assert out["riskLevel"] == "high" and "119" in out["reply"] and env.llm.calls == 1  # 키워드 분류 → 응급 상담


def test_ordinary_message_keeps_normal_router_wait(env, monkeypatch, clock) -> None:  # noqa: F811
    slow = _SlowAnthropic(clock, slow=500)
    _router(monkeypatch, slow)
    _send(env, "어제부터 기침이 나요")
    assert slow.timeouts == [60, pytest.approx(19.5)]


def test_emergency_keyword_still_uses_router_judgement_when_fast(env, monkeypatch, clock) -> None:  # noqa: F811
    _router(monkeypatch, _JsonAnthropic(clock, 1, {**ROUTE, "self_harm": True}))
    out = _send(env, "약을 한꺼번에 많이 먹었어요")
    assert "109" in out["reply"] and env.llm.calls == 0  # 라우터의 자해 판정이 그대로 위기 응답으로


def test_tighter_deadline_restores_and_is_noop_outside_chat(clock) -> None:
    with tighter_deadline(5):
        assert remaining_sec() is None  # 챗 밖에서는 마감이 생기지 않는다
    with chat_deadline(110):
        with tighter_deadline(45):
            assert remaining_sec() == pytest.approx(45)
        with tighter_deadline(500):
            assert remaining_sec() == pytest.approx(110)  # 원래 마감보다 늦추지 않는다
        assert remaining_sec() == pytest.approx(110)
