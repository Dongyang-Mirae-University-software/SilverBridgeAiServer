"""탈퇴 정리 점검 반영(2026-10-30): 대리 상담 기록 삭제(M-1) · 정리 직후 예약 계정 재생성 보류(L-2) · 결과 불명 진료과 비교(L-3)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.base import Base
from app.database.session import get_db
from app.models.chat_log import ChatLog
from app.routers import chat_router
from app.services import reservation_credential_service as rcs
from app.services.reservation_credential_service import ReservationCredentialService
from tests.reservation_flow_helpers import FakeReservation, make_orchestrator
from app.services import chat_service as cs
from tests.test_chat_history_server_side import _router
from tests.test_reservation_confirm_cancel import FULL, _confirm_card, _turn, _value
from tests.test_reservation_confirm_chat_service import _send, env  # noqa: F401

# ---- M-1 대리 상담 기록 ----------------------------------------------------------------------------------------


def _ctx(ward: str | None = None, router_ward: str | None = None, note: str = "") -> str:
    ctx: dict = {"sessionId": "s1", "userContext": {"role": "GUARDIAN", "name": "피보호자", "phone": "010-0000-0002"}, "note": note}
    if ward is not None:
        ctx["userContext"]["wardId"] = ward
    if router_ward is not None:
        ctx["upstreamMeta"] = {"router": {"wardId": router_ward}}
    return json.dumps(ctx, ensure_ascii=False)


@pytest.fixture()
def http():
    engine = create_engine("sqlite+pysqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add_all([
        ChatLog(chat_no="w-own", user_id="ward01", message="m", reply="r", context_json=_ctx()),        # 피보호자 본인
        ChatLog(chat_no="g-proxy", user_id="guard1", message="m", reply="r", context_json=_ctx("ward01")),  # 대리 상담
        ChatLog(chat_no="g-router", user_id="guard1", message="m", reply="r", context_json=_ctx(router_ward="ward01")),
        ChatLog(chat_no="g-self", user_id="guard1", message="m", reply="r", context_json=_ctx()),      # 보호자 본인 상담
        ChatLog(chat_no="g-other", user_id="guard1", message="m", reply="r", context_json=_ctx("ward02")),  # 다른 피보호자
        ChatLog(chat_no="g-text", user_id="guard1", message="m", reply="r", context_json=_ctx(note="ward01")),  # ID가 다른 칸에만
    ])
    session.commit()
    app = FastAPI()
    app.include_router(chat_router.router)
    app.dependency_overrides[get_db] = lambda: session
    with TestClient(app) as client:
        yield client, session
    session.close()


def test_ward_withdrawal_deletes_guardian_proxy_logs_about_that_ward(http) -> None:
    client, session = http
    res = client.delete("/api/v1/chat/logs", params={"userId": "ward01"})
    assert res.status_code == 200
    assert res.json()["data"] == {"deleted": 3, "proxyDeleted": 2}
    assert sorted(r.chat_no for r in session.query(ChatLog).all()) == ["g-other", "g-self", "g-text"]
    assert "010-" not in res.text


def test_guardian_withdrawal_deletes_only_own_logs(http) -> None:
    client, session = http
    res = client.delete("/api/v1/chat/logs", params={"userId": "guard1"})
    assert res.json()["data"] == {"deleted": 5, "proxyDeleted": 0}
    assert [r.chat_no for r in session.query(ChatLog).all()] == ["w-own"]


def test_broken_context_json_is_left_alone(http) -> None:
    client, session = http
    session.add(ChatLog(chat_no="bad", user_id="guard9", message="m", reply="r", context_json="ward01{not json"))
    session.commit()
    client.delete("/api/v1/chat/logs", params={"userId": "ward01"})
    assert session.query(ChatLog).filter_by(chat_no="bad").count() == 1


# ---- L-2 정리 직후 재생성 보류 ------------------------------------------------------------------------------------


class _Client:
    def __init__(self) -> None:
        self._settings = SimpleNamespace(has_reservation_account_secret=True, reservation_account_secret="s")
        self.calls: list[str] = []

    def has_valid_reservation_key(self, key):  # noqa: ARG002
        return False

    def login_account(self, *a):  # noqa: ARG002
        self.calls.append("login")
        return {"httpStatus": 401, "data": {}}

    def signup_account(self, *a):  # noqa: ARG002
        self.calls.append("signup")
        return {"httpStatus": 201, "data": {}}

    def issue_api_key(self, *a):  # noqa: ARG002
        return {"httpStatus": 201, "data": {"apiKey": "sbk_x.y"}}


def _session():
    engine = create_engine("sqlite+pysqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_purged_id_does_not_get_a_new_reservation_account(monkeypatch) -> None:
    client = _Client()
    rcs.mark_purged("ward01")
    assert ReservationCredentialService().ensure_credential(_session(), user_id="ward01", client=client) is None
    assert client.calls == []  # 로그인·가입을 시도하지 않는다
    assert rcs.recently_purged("ward01") and not rcs.recently_purged("ward02")


def test_hold_expires(monkeypatch) -> None:
    now = {"t": 1000.0}
    monkeypatch.setattr(rcs.time, "monotonic", lambda: now["t"])
    rcs.mark_purged("ward01")
    now["t"] += rcs._PURGE_HOLD_SEC + 1
    assert not rcs.recently_purged("ward01")
    client = _Client()
    ReservationCredentialService().ensure_credential(_session(), user_id="ward01", client=client)
    assert "signup" in client.calls


def test_credential_delete_api_marks_the_id(monkeypatch) -> None:
    from app.routers import reservation_credential_router

    monkeypatch.setattr(reservation_credential_router, "get_reservation_api_client",
                        lambda: SimpleNamespace(_settings=SimpleNamespace(has_reservation_account_secret=False)))
    app = FastAPI()
    app.include_router(reservation_credential_router.router)
    session = _session()
    app.dependency_overrides[get_db] = lambda: session
    with TestClient(app) as client:
        assert client.delete("/api/v1/reservation-credentials", params={"userId": "ward01"}).status_code == 200
    assert rcs.recently_purged("ward01")


def test_chat_turn_for_purged_id_treats_reservation_as_disabled(env, monkeypatch) -> None:  # noqa: F811
    # 실제 계정 확보 서비스 - 보류 중인 ID면 예약 서비스(env.svc 의 가짜 client)를 건드리기 전에 None 을 돌려준다.
    monkeypatch.setattr(cs, "get_reservation_credential_service", lambda: ReservationCredentialService())
    _router(env, monkeypatch, route="reservation")
    rcs.mark_purged("user-1")
    out = _send(env, "내과 예약해 주세요")
    assert env.fake.calls == []  # 예약 서비스를 부르지 않는다(공용 키 등 다른 경로로 새지 않음)
    assert "지금은 병원 예약 기능을 사용할 수 없어요" in out["reply"]


# ---- L-3 진료과 비교 ---------------------------------------------------------------------------------------------


def test_same_slot_other_department_is_not_the_new_booking() -> None:
    fake = FakeReservation()
    fake.create_result = {"httpStatus": 0, "data": {"message": "timeout", "code": "UPSTREAM_TIMEOUT"}}
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    other = {"reservationId": 9, "hospitalName": "강남메디내과", "department": "정형외과",
             "reservationDate": FULL["reservation_date"], "reservationTime": "14:00", "status": "BOOKED"}
    seen = {"n": 0}

    def listing(api_key_override=None):  # noqa: ARG001
        seen["n"] += 1
        return {"httpStatus": 200, "data": [other] if seen["n"] > 1 else []}

    fake.list_my_appointments = listing  # type: ignore[method-assign]
    out = _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")))
    assert out["error"]["code"] == "RESULT_UNKNOWN"


# ---- L-1 대조 도구 ---------------------------------------------------------------------------------------------


def test_leftover_counts_splits_members_and_eval_ids() -> None:
    from tools.withdrawal_leftover_check import leftover_counts

    out = leftover_counts({"aaaaaa", "bbbbbb"}, {
        "chat": ["aaaaaa", "gone01", "eval-x", None, "eval-x"],
        "empty": [],
    })
    assert out == {"chat": {"total": 3, "notInBe": 1, "eval": 1}, "empty": {"total": 0, "notInBe": 0, "eval": 0}}
