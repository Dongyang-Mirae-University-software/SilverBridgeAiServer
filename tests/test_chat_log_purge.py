from __future__ import annotations

from contextlib import contextmanager

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


@contextmanager
def _client():
    engine = create_engine("sqlite+pysqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add_all([
        ChatLog(chat_no="c-1", user_id="aaaaaa", message="m", reply="r"),
        ChatLog(chat_no="c-2", user_id="aaaaaa", message="m", reply="r"),
        ChatLog(chat_no="c-3", user_id="bbbbbb", message="m", reply="r"),
    ])
    session.commit()
    app = FastAPI()
    app.include_router(chat_router.router)
    app.dependency_overrides[get_db] = lambda: session
    try:
        with TestClient(app) as client:
            yield client, session
    finally:
        session.close()


def test_delete_removes_only_that_users_logs() -> None:
    with _client() as (client, session):
        res = client.delete("/api/v1/chat/logs", params={"userId": "aaaaaa"})
        assert res.status_code == 200, res.text
        assert res.json()["data"] == {"deleted": 2}
        assert [row.user_id for row in session.query(ChatLog).all()] == ["bbbbbb"]


def test_delete_without_user_id_is_rejected_and_deletes_nothing() -> None:
    with _client() as (client, session):
        for params in ({}, {"userId": ""}, {"userId": "   "}):
            res = client.delete("/api/v1/chat/logs", params=params)
            assert res.status_code == 422, res.text
        assert session.query(ChatLog).count() == 3


def test_delete_unknown_user_is_idempotent() -> None:
    with _client() as (client, session):
        res = client.delete("/api/v1/chat/logs", params={"userId": "zzzzzz"})
        assert res.status_code == 200
        assert res.json()["data"] == {"deleted": 0}
        assert session.query(ChatLog).count() == 3


def test_chat_router_is_protected_by_api_key() -> None:
    # 라우터 등록부(main.py)가 챗 라우터 전체에 키 검사를 건다 - 삭제 엔드포인트도 같은 라우터라 예외가 없다.
    source = open("app/main.py", encoding="utf-8").read()
    assert "app.include_router(chat_router, dependencies=[Depends(require_api_key)])" in source
