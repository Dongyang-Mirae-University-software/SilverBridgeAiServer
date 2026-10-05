"""QA(종합 점검) 후속 - 링버퍼 바이트 상한, 챗 로그 소유자 검증, 게임 API 키 플래그.

FE 저장소는 이번에 바뀌지 않는다. 그래서 FE 가 지금 부르는 형태(목록 ?userId= / 상세 userId 없음 /
게임 키 없음)를 그대로 재현해, 기본 설정에서 깨지지 않음을 고정한다.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timedelta

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings, get_settings
from app.core.response import error_response
from app.core.security import game_router_dependencies
from app.database.base import Base
from app.database.session import get_db
from app.models import game as game_models  # noqa: F401
from app.models.chat_log import ChatLog
from app.services.clip_buffer import ClipFrameBuffer

T0 = datetime(2026, 10, 5, 1, 0, 0)
MB = 1024 * 1024
API_KEY = "test-only-key"


def _at(sec: float) -> datetime:
    return T0 + timedelta(seconds=sec)


def _sum_bytes(buf: ClipFrameBuffer, session_id: str) -> int:
    return sum(len(data) for _, data in buf.snapshot(session_id, _at(-1000), _at(1000)))


# ---------------------------------------------------------------- ① 링버퍼 바이트 상한


def test_session_cap_drops_oldest_frames_first() -> None:
    buf = ClipFrameBuffer(buffer_seconds=10, max_frames=100, session_max_bytes=10)
    for i in range(5):
        buf.add("s1", _at(i * 0.1), bytes([i]) * 4)

    frames = buf.snapshot("s1", _at(-1), _at(10))
    assert [data[0] for _, data in frames] == [3, 4]  # 4바이트 × 2 = 8 ≤ 10, 오래된 0·1·2 제거
    assert buf.session_bytes("s1") == 8
    assert buf.total_bytes() == 8


def test_total_cap_trims_largest_session_and_keeps_small_one() -> None:
    buf = ClipFrameBuffer(buffer_seconds=10, max_frames=100, total_max_bytes=20)
    for i in range(3):
        buf.add("big", _at(i * 0.1), b"B" * 6)  # 18
    buf.add("small", _at(0.5), b"s" * 4)  # 합계 22 > 20

    assert buf.session_bytes("big") == 12  # 가장 큰 세션의 가장 오래된 한 장만 제거
    assert buf.session_bytes("small") == 4
    assert buf.total_bytes() == 16


def test_total_cap_does_not_drop_the_frame_just_added() -> None:
    buf = ClipFrameBuffer(buffer_seconds=10, max_frames=100, total_max_bytes=20)
    for i in range(3):
        buf.add("a", _at(i * 0.1), b"a" * 6)  # 18
    buf.add("b", _at(0.5), b"b" * 8)  # b 가 더 크지만 방금 들어온 한 장뿐 → a 부터 줄인다

    assert buf.session_bytes("b") == 8
    assert buf.session_bytes("a") == 12
    assert buf.total_bytes() == 20


def test_frame_larger_than_cap_is_not_buffered_and_logs_only_id_and_size(caplog: pytest.LogCaptureFixture) -> None:
    buf = ClipFrameBuffer(buffer_seconds=10, max_frames=100, session_max_bytes=10, total_max_bytes=100)
    buf.add("s1", _at(0), b"ok")
    secret_payload = b"SECRETFRAME"  # 11바이트 > 10
    with caplog.at_level(logging.WARNING, logger="app.services.clip_buffer"):
        buf.add("s1", _at(0.1), secret_payload)
        buf.add("s1", _at(0.2), secret_payload)  # 같은 사유는 30초에 한 번만

    assert [data for _, data in buf.snapshot("s1", _at(-1), _at(1))] == [b"ok"]
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 1
    assert "s1" in warnings[0] and "bytes=11" in warnings[0]
    assert "SECRETFRAME" not in warnings[0]


def test_byte_accounting_survives_time_eviction_max_frames_clear_and_sweep() -> None:
    buf = ClipFrameBuffer(buffer_seconds=2, max_frames=5, session_max_bytes=MB, total_max_bytes=MB)
    for i in range(20):  # 0~9.5초, 0.5초 간격 → 시간 제거와 장수 제거가 함께 돈다
        buf.add("s1", _at(i * 0.5), b"x" * (i + 1))
        buf.add("s2", _at(i * 0.5), b"y" * 3)

    assert buf.session_bytes("s1") == _sum_bytes(buf, "s1")
    assert buf.session_bytes("s2") == _sum_bytes(buf, "s2")
    assert buf.total_bytes() == _sum_bytes(buf, "s1") + _sum_bytes(buf, "s2")
    assert len(buf.snapshot("s1", _at(-1), _at(100))) == 5

    buf.clear("s1")
    assert buf.total_bytes() == _sum_bytes(buf, "s2")

    # s2 는 더 이상 프레임이 오지 않는다 → 다른 세션의 프레임이 들어올 때 sweep 이 지운다
    buf._last_sweep = 0.0
    buf.add("s3", _at(100), b"z")
    assert buf.session_bytes("s2") == 0
    assert buf.total_bytes() == 1


def test_fhd_100_frames_stay_within_caps() -> None:
    # 전제의 최악 경우(장당 6.5MB, 세션당 100장). 같은 bytes 객체를 재사용해 테스트 메모리는 늘지 않는다.
    frame = b"\xff" * int(6.5 * MB)
    buf = ClipFrameBuffer(buffer_seconds=10, max_frames=100, session_max_bytes=64 * MB, total_max_bytes=512 * MB)
    for sid in range(10):
        for i in range(100):
            buf.add(f"s{sid}", _at(i * 0.1), frame)
            assert buf.session_bytes(f"s{sid}") <= 64 * MB
            assert buf.total_bytes() <= 512 * MB

    # 세션마다 64MB 이하로 최근 9장이 남고, 10세션 합계(약 585MB)는 전체 상한에서 잘린다
    assert buf.total_bytes() <= 512 * MB
    last = buf.snapshot("s9", _at(-1), _at(100))
    assert last[-1][0] == _at(9.9)  # 가장 최근 프레임은 남는다


def test_caps_zero_or_negative_disable_byte_limits() -> None:
    buf = ClipFrameBuffer(buffer_seconds=10, max_frames=100, session_max_bytes=0, total_max_bytes=-1)
    for i in range(50):
        buf.add("s1", _at(i * 0.1), b"x" * 1000)
    assert buf.session_bytes("s1") == 50_000


def test_default_settings_have_byte_caps() -> None:
    settings = Settings()
    assert settings.clip_session_max_bytes == 64 * MB
    assert settings.clip_total_max_bytes == 512 * MB


# ---------------------------------------------------------------- ② 챗 로그 소유자 검증


def _with_handler(app: FastAPI) -> FastAPI:
    # main.py 의 예외 핸들러와 같은 규칙(app.main 은 모델 로딩까지 끌고 와서 직접 붙인다).
    @app.exception_handler(HTTPException)
    async def _handler(_: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "success" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(status_code=exc.status_code, content=error_response(str(exc.detail), "HTTP_ERROR", None))

    return app


@contextmanager
def _db():
    engine = create_engine("sqlite+pysqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def _chat_client(chat_require_user_id: bool):
    from app.routers import chat_router

    with _db() as session:
        mine = ChatLog(chat_no="c-1", user_id="ward-a", message="m1", reply="r1")
        other = ChatLog(chat_no="c-2", user_id="ward-b", message="m2", reply="r2")
        session.add_all([mine, other])
        session.commit()

        app = _with_handler(FastAPI())
        app.include_router(chat_router.router)
        app.dependency_overrides[get_db] = lambda: session
        # 다른 테스트가 app.core.config 를 reload 하므로, 라우터가 실제로 참조하는 get_settings 로 덮어쓴다.
        app.dependency_overrides[chat_router.get_settings] = lambda: chat_router.get_settings().model_copy(
            update={"chat_require_user_id": chat_require_user_id},
        )
        with TestClient(app) as client:
            yield client, mine.id, other.id


@pytest.mark.parametrize("params", [{}, {"userId": ""}, {"userId": "   "}])
def test_chat_list_without_user_id_is_rejected(params: dict) -> None:
    with _chat_client(chat_require_user_id=False) as (client, _, _):
        res = client.get("/api/v1/chat/logs", params=params)

    assert res.status_code == 422
    assert res.json()["errorCode"] == "CHAT_USER_ID_REQUIRED"
    assert res.json()["success"] is False


@pytest.mark.parametrize("flag", [False, True])
def test_chat_fe_call_shapes(flag: bool) -> None:
    # FE: 목록 = /chat/logs?userId=…, 상세 = /chat/logs/{id} (userId 없음)
    with _chat_client(chat_require_user_id=flag) as (client, mine_id, _):
        listed = client.get("/api/v1/chat/logs", params={"userId": "ward-a"})
        detail = client.get(f"/api/v1/chat/logs/{mine_id}")

    assert listed.status_code == 200
    assert [item["userId"] for item in listed.json()["data"]] == ["ward-a"]
    if flag:
        assert detail.status_code == 422
        assert detail.json()["errorCode"] == "CHAT_USER_ID_REQUIRED"
    else:
        assert detail.status_code == 200  # 기본 OFF: 지금 FE 형태 그대로 동작
        assert detail.json()["data"]["userId"] == "ward-a"


@pytest.mark.parametrize("flag", [False, True])
def test_chat_detail_owner_match_and_mismatch(flag: bool) -> None:
    with _chat_client(chat_require_user_id=flag) as (client, mine_id, other_id):
        ok = client.get(f"/api/v1/chat/logs/{mine_id}", params={"userId": "ward-a"})
        mismatch = client.get(f"/api/v1/chat/logs/{other_id}", params={"userId": "ward-a"})
        missing = client.get("/api/v1/chat/logs/999999", params={"userId": "ward-a"})

    assert ok.status_code == 200
    assert ok.json()["data"]["id"] == mine_id
    # 남의 기록과 없는 기록은 응답이 같아야 존재 여부가 드러나지 않는다
    assert mismatch.status_code == 404
    assert mismatch.json() == missing.json()
    assert mismatch.json()["errorCode"] == "CHAT_LOG_NOT_FOUND"


def test_chat_require_user_id_defaults_off() -> None:
    assert Settings().chat_require_user_id is False


# ---------------------------------------------------------------- ③ 게임 API 키 플래그


@contextmanager
def _game_client(game_require_api_key: bool):
    from app.core import security
    from app.routers import game_router

    settings = get_settings().model_copy(update={"api_key": API_KEY, "game_require_api_key": game_require_api_key})
    with _db() as session:
        app = _with_handler(FastAPI())
        app.include_router(game_router.router, dependencies=game_router_dependencies(settings))
        app.dependency_overrides[get_db] = lambda: session
        app.dependency_overrides[security.get_settings] = lambda: settings  # require_api_key 가 참조하는 쪽
        with TestClient(app) as client:
            yield client


# 피보호자 게임 화면이 브라우저에서 키 없이 부르는 형태(iframe + embed 내부 JS)와 보호자 프록시 경로
_GAME_CALLS = [
    ("GET", "/api/v1/games/embed?userId=7&gameSlug=maze", None),
    ("GET", "/api/v1/games/maze/state?userId=7", None),
    ("POST", "/api/v1/games/maze/reset", {"userId": 7, "gameSlug": "maze"}),
    ("GET", "/api/v1/games/progress?userId=7", None),
    ("GET", "/api/v1/games/activity?userId=7", None),
]


@pytest.mark.parametrize(("method", "path", "body"), _GAME_CALLS)
def test_game_flag_off_keeps_keyless_calls_working(method: str, path: str, body: dict | None) -> None:
    with _game_client(game_require_api_key=False) as client:
        res = client.request(method, path, json=body)
    assert res.status_code == 200, res.text


@pytest.mark.parametrize(("method", "path", "body"), _GAME_CALLS)
def test_game_flag_on_requires_api_key(method: str, path: str, body: dict | None) -> None:
    with _game_client(game_require_api_key=True) as client:
        no_key = client.request(method, path, json=body)
        wrong_key = client.request(method, path, json=body, headers={"X-API-Key": "wrong"})
        with_key = client.request(method, path, json=body, headers={"X-API-Key": API_KEY})

    assert no_key.status_code == 401
    assert no_key.json()["errorCode"] == "AUTH_INVALID_KEY"
    assert wrong_key.status_code == 401
    assert with_key.status_code == 200, with_key.text


def test_game_require_api_key_defaults_off() -> None:
    assert Settings().game_require_api_key is False
    assert game_router_dependencies(Settings()) == []
