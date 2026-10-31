"""연결이 끝난 보호자의 대리 상담 기록 삭제 (BE 2026-10 전체 점검 L-13).

보호자(GD0001)가 피보호자(WD0001)를 대신해 상담한 기록만 지우고, 보호자 본인 상담·다른 피보호자 대리 상담·
다른 보호자 기록은 남긴다. 대리 상담 여부는 context_json 의 userContext.wardId / upstreamMeta.router.wardId 로 판단한다.
"""
from __future__ import annotations

import json
from contextlib import contextmanager

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.base import Base
from app.database.session import get_db
from app.models.chat_log import ChatLog
from app.routers import chat_router
from app.services import reservation_credential_service as purge


def _ctx(ward_id: str | None, via_router: bool = False) -> str:
    if ward_id is None:
        return json.dumps({"sessionId": "s"})
    if via_router:
        return json.dumps({"upstreamMeta": {"router": {"wardId": ward_id}}})
    return json.dumps({"userContext": {"wardId": ward_id}})


@contextmanager
def _client():
    engine = create_engine("sqlite+pysqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add_all([
        ChatLog(chat_no="p-1", user_id="GD0001", message="m", reply="r", context_json=_ctx("WD0001")),
        ChatLog(chat_no="p-2", user_id="GD0001", message="m", reply="r", context_json=_ctx("WD0001", via_router=True)),
        ChatLog(chat_no="own", user_id="GD0001", message="m", reply="r", context_json=_ctx(None)),
        ChatLog(chat_no="other-ward", user_id="GD0001", message="m", reply="r", context_json=_ctx("WD0002")),
        ChatLog(chat_no="other-guardian", user_id="GD0002", message="m", reply="r", context_json=_ctx("WD0001")),
        ChatLog(chat_no="ward-own", user_id="WD0001", message="m", reply="r", context_json=_ctx(None)),
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


def _remaining(session) -> list[str]:
    return sorted(row.chat_no for row in session.query(ChatLog).all())


def test_deletes_only_that_guardians_proxy_logs_for_that_ward() -> None:
    with _client() as (client, session):
        res = client.delete("/api/v1/chat/logs/proxy", params={"guardianId": "GD0001", "wardId": "WD0001"})
        assert res.status_code == 200, res.text
        assert res.json()["data"] == {"deleted": 2}
        assert _remaining(session) == ["other-guardian", "other-ward", "own", "ward-own"]


def test_both_ids_are_required_and_nothing_is_deleted_without_them() -> None:
    with _client() as (client, session):
        for params in ({"guardianId": "GD0001"}, {"wardId": "WD0001"}, {"guardianId": " ", "wardId": "WD0001"}, {}):
            res = client.delete("/api/v1/chat/logs/proxy", params=params)
            assert res.status_code == 422, (params, res.text)
        assert session.query(ChatLog).count() == 6


def test_repeat_is_idempotent() -> None:
    with _client() as (client, session):
        client.delete("/api/v1/chat/logs/proxy", params={"guardianId": "GD0001", "wardId": "WD0001"})
        res = client.delete("/api/v1/chat/logs/proxy", params={"guardianId": "GD0001", "wardId": "WD0001"})
        assert res.json()["data"] == {"deleted": 0}


def test_delete_marks_pair_hold_without_touching_member_hold() -> None:
    with _client() as (client, _):
        client.delete("/api/v1/chat/logs/proxy", params={"guardianId": "GD0003", "wardId": "WD0003"})
    assert purge.recently_proxy_purged("GD0003", "WD0003")
    # 쌍 보류는 회원 단위 보류(탈퇴 정리)와 다른 키다 - 보호자·피보호자 각자의 상담·예약은 막지 않는다
    assert not purge.recently_purged("GD0003")
    assert not purge.recently_purged("WD0003")
    assert not purge.recently_proxy_purged("GD0003", "WD0004")
