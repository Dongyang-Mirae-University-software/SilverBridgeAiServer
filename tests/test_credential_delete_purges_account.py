"""회원 예약 키 행을 지울 때 예약 서비스의 그 회원 계정(예약·키 포함)도 지운다(2026-10-30) - 탈퇴 정리·측정 정리 공통.
예약 서비스가 계정 삭제 경로를 모르면(옛 버전) AI-{userId} 키 폐기로 대신한다(#122)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.base import Base
from app.database.session import get_db
from app.models.reservation_credential import ReservationCredential
from app.routers import reservation_credential_router
from app.services.chat_deadline import ChatDeadlineExceeded
from app.services.reservation_credential_service import ReservationCredentialService


def _key(kid: int, name: str, active: bool = True) -> dict:
    return {"id": kid, "name": name, "keyPrefix": f"p{kid}", "isActive": active}


class _Client:
    def __init__(self, keys=None, *, secret: bool = True, login_status: int = 200, list_status: int = 200,
                 revoke_status: int = 200, delete_status: int = 200) -> None:
        self._settings = SimpleNamespace(has_reservation_account_secret=secret, reservation_account_secret="s" if secret else "")
        self.keys = keys or []
        self.login_status = login_status
        self.list_status = list_status
        self.revoke_status = revoke_status
        self.delete_status = delete_status
        self.calls: list[str] = []
        self.revoked: list[int] = []

    def login_account(self, email, password):  # noqa: ARG002
        self.calls.append("login")
        return {"httpStatus": self.login_status, "data": {"accessToken": "jwt"} if self.login_status == 200 else {}}

    def signup_account(self, *a):  # noqa: ARG002
        self.calls.append("signup")
        return {"httpStatus": 201, "data": {}}

    def delete_my_account(self, token):  # noqa: ARG002
        self.calls.append("delete_me")
        if self.delete_status == 200:
            return {"httpStatus": 200, "data": {"deleted": True, "reservations": 3, "apiKeys": 2}}
        return {"httpStatus": self.delete_status, "data": {}}

    def list_api_keys(self, token):  # noqa: ARG002
        self.calls.append("list")
        return {"httpStatus": self.list_status, "data": {"keys": self.keys}}

    def revoke_api_key(self, token, kid):  # noqa: ARG002
        self.calls.append("revoke")
        self.revoked.append(kid)
        return {"httpStatus": self.revoke_status, "data": {}}


def _app(monkeypatch, client):
    engine = create_engine("sqlite+pysqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add_all([
        ReservationCredential(user_id="aaaaaa", api_key_prefix="p1", api_key="sbk_p1.secret1"),
        ReservationCredential(user_id="bbbbbb", api_key_prefix="p9", api_key="sbk_p9.secret9"),
    ])
    session.commit()
    app = FastAPI()
    app.include_router(reservation_credential_router.router)
    app.dependency_overrides[get_db] = lambda: session
    monkeypatch.setattr(reservation_credential_router, "get_reservation_api_client", lambda: client)
    return TestClient(app), session


def test_delete_removes_the_reservation_account(monkeypatch) -> None:
    client = _Client([_key(1, "AI-aaaaaa")])
    http, session = _app(monkeypatch, client)
    res = http.delete("/api/v1/reservation-credentials", params={"userId": "aaaaaa"})
    assert res.status_code == 200
    assert res.json()["data"] == {"deleted": 1, "accountDeleted": True, "revokedKeys": 2}
    assert client.calls == ["login", "delete_me"] and client.revoked == []
    assert [c.user_id for c in session.query(ReservationCredential).all()] == ["bbbbbb"]
    assert "sbk_" not in res.text and "jwt" not in res.text


@pytest.mark.parametrize("status", [404, 405, 500])
def test_old_reservation_service_falls_back_to_revoking_that_users_ai_keys_only(monkeypatch, status) -> None:
    client = _Client([_key(1, "AI-aaaaaa"), _key(2, "AI-aaaaaa"), _key(3, "AI-aaaaaa", active=False), _key(4, "기본 API 키")],
                     delete_status=status)
    http, session = _app(monkeypatch, client)
    res = http.delete("/api/v1/reservation-credentials", params={"userId": "aaaaaa"})
    assert res.status_code == 200
    assert res.json()["data"] == {"deleted": 1, "accountDeleted": None, "revokedKeys": 2}
    assert client.revoked == [1, 2]
    assert [c.user_id for c in session.query(ReservationCredential).all()] == ["bbbbbb"]
    assert "sbk_" not in res.text and "jwt" not in res.text


def test_row_is_deleted_before_revoking(monkeypatch) -> None:
    seen: dict[str, int] = {}
    client = _Client([_key(1, "AI-aaaaaa")])
    http, session = _app(monkeypatch, client)
    original = client.login_account

    def login(email, password):
        seen["rows"] = session.query(ReservationCredential).filter_by(user_id="aaaaaa").count()
        return original(email, password)

    client.login_account = login  # type: ignore[method-assign]
    http.delete("/api/v1/reservation-credentials", params={"userId": "aaaaaa"})
    assert seen["rows"] == 0


@pytest.mark.parametrize("client,expected", [
    (_Client(secret=False), {"accountDeleted": None, "revokedKeys": None}),   # 비밀값 없음 - 건너뜀
    (_Client(login_status=401), {"accountDeleted": False, "revokedKeys": 0}),  # 예약 계정 없음 - 가입하지 않는다
    (_Client(login_status=500), {"accountDeleted": None, "revokedKeys": None}),
    (_Client([_key(1, "AI-aaaaaa")], delete_status=404, list_status=500), {"accountDeleted": None, "revokedKeys": None}),
    (_Client([_key(1, "AI-aaaaaa")], delete_status=404, revoke_status=500), {"accountDeleted": None, "revokedKeys": None}),
])
def test_revoke_failures_keep_the_row_deleted(monkeypatch, client, expected) -> None:
    http, session = _app(monkeypatch, client)
    res = http.delete("/api/v1/reservation-credentials", params={"userId": "aaaaaa"})
    assert res.status_code == 200
    assert res.json()["data"] == {"deleted": 1, **expected}
    assert session.query(ReservationCredential).filter_by(user_id="aaaaaa").count() == 0
    assert "signup" not in client.calls


def test_revoke_exception_or_deadline_keeps_the_row_deleted(monkeypatch) -> None:
    client = _Client()

    def boom(email, password):  # noqa: ARG001
        raise ChatDeadlineExceeded("reservation_api:POST:/auth/login")

    client.login_account = boom  # type: ignore[method-assign]
    http, _ = _app(monkeypatch, client)
    res = http.delete("/api/v1/reservation-credentials", params={"userId": "aaaaaa"})
    assert res.status_code == 200 and res.json()["data"] == {"deleted": 1, "accountDeleted": None, "revokedKeys": None}


def test_purge_runs_under_its_own_short_deadline(monkeypatch) -> None:
    from app.services import chat_deadline as cd

    budgets: list[float | None] = []
    client = _Client([_key(1, "AI-aaaaaa")])

    def login(email, password):  # noqa: ARG001
        budgets.append(cd.remaining_sec())
        return {"httpStatus": 200, "data": {"accessToken": "jwt"}}

    client.login_account = login  # type: ignore[method-assign]
    ReservationCredentialService().purge_reservation_account(client, "aaaaaa")  # type: ignore[arg-type]
    assert budgets and budgets[0] is not None and budgets[0] <= 8.0
