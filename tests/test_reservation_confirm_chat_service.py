"""ChatService 전체 경로(라우터 → 예약 오케스트레이터 → chat_log 저장 → 다음 턴 복원)에서 확인 단계가 지켜지는지."""
from __future__ import annotations

import tempfile
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.chat_log import ChatLog
from app.schemas.chat_schema import ChatContext, ChatRequest, UiSelection
from app.services import chat_service as cs
from tests.reservation_flow_helpers import FUTURE, FakeReservation, make_orchestrator


class _Router:
    def route(self, message: str, history, ctx, prev_reservation=None) -> dict[str, Any]:  # noqa: ARG002
        return {"route": "reservation", "normalized_message": message, "symptom_summary": "", "emergency": False, "reason": "t", "engine": "fake"}


class _Credentials:
    def ensure_credential(self, db, **kwargs):  # noqa: ARG002
        return "fake-key"


class _Llm:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, *a, **k) -> dict[str, Any]:  # noqa: ARG002
        self.calls += 1
        return {
            "reply": "지금 바로 119에 전화해 주세요.", "riskLevel": "high", "recommendedAction": "emergency", "reservationRequired": False,
            "intent": "emergency_guidance", "engine": "fake", "modelName": "fake", "type": "message", "ui": None, "tool": None, "data": None,
        }


@pytest.fixture()
def env(monkeypatch):
    engine = create_engine(f"sqlite:///{tempfile.mkdtemp(prefix='sb_chat_')}/t.db")
    ChatLog.__table__.create(bind=engine)
    db = sessionmaker(bind=engine)()
    fake = FakeReservation()
    llm = _Llm()
    orch = make_orchestrator(fake, {"reservation_date": FUTURE})
    monkeypatch.setattr(cs, "get_chat_router_service", lambda: _Router())
    monkeypatch.setattr(cs, "get_reservation_credential_service", lambda: _Credentials())
    monkeypatch.setattr(cs, "get_reservation_api_client", lambda: object())
    monkeypatch.setattr(cs, "get_reservation_orchestrator", lambda: orch)
    monkeypatch.setattr(cs, "get_medical_llm_service", lambda: llm)
    svc = cs.ChatService()
    svc._settings = SimpleNamespace(has_reservation_account_secret=True, claude_model_name="fake")
    yield SimpleNamespace(svc=svc, db=db, fake=fake, llm=llm)
    db.close()


def _send(env, message: str = "", ui: tuple[str, str] | None = None) -> dict[str, Any]:
    payload = ChatRequest(
        userId="user-1", message=message, sessionId="s1",
        context=ChatContext(name="홍길동", phone="010-1234-5678", location="서울 강남구"),
        uiSelection=UiSelection(field=ui[0], value=ui[1]) if ui else None,
    )
    return env.svc.process_message(env.db, payload)


def _token(out: dict[str, Any], prefix: str) -> str:
    return next(o.value for o in out["ui"].options if o.value.startswith(prefix))


def test_full_chain_confirm_then_book_once(env) -> None:
    card = _send(env, "강남메디내과 내과 14:00 예약해줘")
    assert card["ui"].field == "reservation_confirm" and env.fake.names() == []
    done = _send(env, ui=("reservation_confirm", _token(card, "confirm:")))
    assert env.fake.names() == ["create"] and done["tool"] == "make_appointment"
    replay = _send(env, ui=("reservation_confirm", _token(card, "confirm:")))
    assert env.fake.names() == ["create"] and "시간이 지났" in replay["reply"]


def test_unrelated_turn_in_between_invalidates_the_pending_confirm(env) -> None:
    card = _send(env, "강남메디내과 내과 14:00 예약해줘")
    _send(env, "숨을 못 쉬겠어요")  # 응급: 예약 흐름이 아니라 상담으로 간다
    assert env.llm.calls == 1
    late = _send(env, ui=("reservation_confirm", _token(card, "confirm:")))
    assert env.fake.names() == [] and "시간이 지났" in late["reply"]


def test_cancel_sentence_goes_through_chain_and_needs_confirmation(env) -> None:
    env.fake.appointments = {5: {"reservationId": 5, "hospitalName": "강남메디내과", "department": "내과", "reservationDate": FUTURE, "reservationTime": "09:00", "status": "BOOKED"}}
    pick = _send(env, "예약 취소할래")
    assert pick["ui"].field == "reservation_cancel_pick"
    confirm = _send(env, ui=("reservation_cancel_pick", "5"))
    assert env.fake.names() == []
    _send(env, ui=("reservation_cancel_confirm", _token(confirm, "yes:")))
    assert env.fake.names() == ["cancel"]
