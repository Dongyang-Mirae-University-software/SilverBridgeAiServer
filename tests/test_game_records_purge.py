"""회원 탈퇴 정리용 게임 기록 삭제 (BE 게임 기록 중계 점검 M-1, 2026-11-20).

- 그 회원의 진행·풀이 기록만 지운다(다른 회원 기록과 게임 목록은 그대로)
- userId 필수(없거나 1 미만이면 422, 아무것도 지우지 않음)
- GAME_REQUIRE_API_KEY 를 꺼도 삭제는 항상 키를 요구한다
"""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import get_settings
from app.database.base import Base
from app.database.session import get_db
from app.models.game import GameAttempt, GameCatalog, GameProgress
from app.routers import game_router
from app.services.game_service import GameService

KEY = "test-key"


@contextmanager
def _client(game_require_api_key: bool = True):
    engine = create_engine("sqlite+pysqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    service = GameService(session)
    service.start_game(111, "maze")
    service.start_game(111, "arithmetic")
    service.start_game(222, "maze")
    session.add_all([
        GameAttempt(user_id=111, game_slug="maze", stage_no=1, is_correct=True, score_delta=10),
        GameAttempt(user_id=111, game_slug="maze", stage_no=2),
        GameAttempt(user_id=222, game_slug="maze", stage_no=1),
    ])
    session.commit()
    app = FastAPI()
    app.include_router(game_router.router)
    app.dependency_overrides[get_db] = lambda: session
    app.dependency_overrides[get_settings] = lambda: SimpleNamespace(
        api_key=KEY, game_require_api_key=game_require_api_key)
    try:
        with TestClient(app) as client:
            yield client, session
    finally:
        session.close()


def _count(session, model, user_id: int) -> int:
    return session.query(model).filter(model.user_id == user_id).count()


def test_delete_removes_only_that_users_records() -> None:
    with _client() as (client, session):
        catalogs = session.query(GameCatalog).count()
        res = client.delete("/api/v1/games/records", params={"userId": 111}, headers={"X-API-Key": KEY})
        assert res.status_code == 200, res.text
        assert res.json()["data"] == {"progressDeleted": 2, "attemptsDeleted": 2}
        assert _count(session, GameProgress, 111) == 0
        assert _count(session, GameAttempt, 111) == 0
        assert _count(session, GameProgress, 222) == 1
        assert _count(session, GameAttempt, 222) == 1
        assert session.query(GameCatalog).count() == catalogs


def test_delete_unknown_user_is_idempotent() -> None:
    with _client() as (client, session):
        res = client.delete("/api/v1/games/records", params={"userId": 999}, headers={"X-API-Key": KEY})
        assert res.status_code == 200
        assert res.json()["data"] == {"progressDeleted": 0, "attemptsDeleted": 0}
        assert session.query(GameProgress).count() == 3


def test_delete_requires_valid_user_id() -> None:
    with _client() as (client, session):
        for params in ({}, {"userId": 0}, {"userId": -1}, {"userId": "abc"}):
            res = client.delete("/api/v1/games/records", params=params, headers={"X-API-Key": KEY})
            assert res.status_code == 422, (params, res.text)
        assert session.query(GameAttempt).count() == 3


def test_delete_always_requires_api_key_even_when_game_key_switch_is_off() -> None:
    with _client(game_require_api_key=False) as (client, session):
        for headers in ({}, {"X-API-Key": "wrong"}):
            res = client.delete("/api/v1/games/records", params={"userId": 111}, headers=headers)
            assert res.status_code == 401, res.text
        assert _count(session, GameProgress, 111) == 2
        # 같은 설정에서 조회 경로는 스위치를 따라 키 없이 열린다 - 삭제만 예외로 막힌 것을 확인
        assert client.get("/api/v1/games/progress", params={"userId": 111}).status_code == 200
