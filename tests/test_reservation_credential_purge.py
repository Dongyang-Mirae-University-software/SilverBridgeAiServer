from __future__ import annotations

from contextlib import contextmanager

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.base import Base
from app.database.session import get_db
from app.models.reservation_credential import ReservationCredential
from app.routers import reservation_credential_router


@contextmanager
def _client():
    engine = create_engine("sqlite+pysqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add_all([
        ReservationCredential(user_id="aaaaaa", api_key_prefix="p1", api_key="sbk_p1.secret1"),
        ReservationCredential(user_id="bbbbbb", api_key_prefix="p2", api_key="sbk_p2.secret2"),
    ])
    session.commit()
    app = FastAPI()
    app.include_router(reservation_credential_router.router)
    app.dependency_overrides[get_db] = lambda: session
    try:
        with TestClient(app) as client:
            yield client, session
    finally:
        session.close()


def test_delete_removes_only_that_users_credential_and_never_echoes_the_key() -> None:
    with _client() as (client, session):
        res = client.delete("/api/v1/reservation-credentials", params={"userId": "aaaaaa"})
        assert res.status_code == 200, res.text
        assert res.json()["data"] == {"deleted": 1}
        assert "secret" not in res.text
        assert [row.user_id for row in session.query(ReservationCredential).all()] == ["bbbbbb"]


def test_delete_without_user_id_is_rejected_and_deletes_nothing() -> None:
    with _client() as (client, session):
        for params in ({}, {"userId": ""}, {"userId": "   "}):
            res = client.delete("/api/v1/reservation-credentials", params=params)
            assert res.status_code == 422, res.text
        assert session.query(ReservationCredential).count() == 2


def test_delete_unknown_user_is_idempotent() -> None:
    with _client() as (client, session):
        res = client.delete("/api/v1/reservation-credentials", params={"userId": "zzzzzz"})
        assert res.status_code == 200
        assert res.json()["data"] == {"deleted": 0}
        assert session.query(ReservationCredential).count() == 2


def test_credential_router_is_protected_by_api_key() -> None:
    source = open("app/main.py", encoding="utf-8").read()
    assert "app.include_router(reservation_credential_router, dependencies=[Depends(require_api_key)])" in source
