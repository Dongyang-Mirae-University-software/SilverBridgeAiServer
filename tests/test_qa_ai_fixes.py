"""QA AI-1(동기 엔드포인트 WS 방송 유실) · AI-2(쿼리 API 키) · AI-3(문자열 회원 ID) 회귀 테스트."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from contextlib import contextmanager

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.websockets import WebSocketDisconnect

import app.models.ai_model  # noqa: F401  (AnalysisResult FK 대상)
import app.models.camera  # noqa: F401
from app.core.config import get_settings
from app.core.security import require_api_key
from app.database.base import Base
from app.database.schema_upgrades import upgrade_member_id_columns
from app.models.chat_log import ChatLog
from app.routers import live_stream_router, live_ws_router
from app.schemas.chat_schema import ChatRequest
from app.schemas.common import MemberId
from app.services.live_ws_manager import LiveWebSocketManager, live_ws_manager
from app.services.reservation_credential_service import ReservationCredentialService
from app.utils.logger import ApiKeyRedactingFilter, redact_api_key

API_KEY = "test-qa-key"


def _settings(**overrides):
    return get_settings().model_copy(update={"api_key": API_KEY, **overrides})


# ---------------------------------------------------------------- AI-1


class _FakeSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def accept(self) -> None:
        return None

    async def send_json(self, payload: dict) -> None:
        self.sent.append(payload)


def test_broadcast_from_worker_thread_reaches_clients() -> None:
    manager = LiveWebSocketManager()
    sock = _FakeSocket()

    async def run() -> None:
        await manager.connect(sock)  # type: ignore[arg-type]
        loop = asyncio.get_running_loop()
        # 동기 def 엔드포인트처럼 실행 중인 루프가 없는 스레드에서 호출한다
        await loop.run_in_executor(None, manager.broadcast_nowait, {"type": "live_streams", "data": []})
        for _ in range(50):
            if sock.sent:
                return
            await asyncio.sleep(0.01)

    asyncio.run(run())
    assert sock.sent == [{"type": "live_streams", "data": []}]


def test_broadcast_without_loop_or_clients_is_silent(caplog: pytest.LogCaptureFixture) -> None:
    manager = LiveWebSocketManager()
    with caplog.at_level(logging.WARNING):
        manager.broadcast_nowait({"type": "live_streams"})  # 예외 없이, 경고도 없이(보낼 곳이 없다)
    assert "WS-BROADCAST-DROPPED" not in caplog.text


def _ws_app(monkeypatch: pytest.MonkeyPatch, **overrides) -> FastAPI:
    from app.database.session import engine

    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(live_ws_router, "get_settings", lambda: _settings(**overrides))
    app = FastAPI()
    app.include_router(live_stream_router.router, dependencies=[Depends(require_api_key)])
    app.include_router(live_ws_router.router)
    app.dependency_overrides[get_settings] = lambda: _settings(**overrides)
    return app


def _receive_until(ws, wanted: str, timeout: float = 3.0) -> dict | None:
    """TestClient WS 수신에는 제한 시간이 없어 별도 스레드로 기다린다."""
    box: queue.Queue = queue.Queue()

    def reader() -> None:
        try:
            while True:
                msg = ws.receive_json()
                if msg.get("type") == wanted:
                    box.put(msg)
                    return
        except Exception as exc:  # noqa: BLE001
            box.put(exc)

    threading.Thread(target=reader, daemon=True).start()
    try:
        got = box.get(timeout=timeout)
    except queue.Empty:
        return None
    return got if isinstance(got, dict) else None


def test_create_session_broadcasts_live_streams_to_ws_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _ws_app(monkeypatch)
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/ws/live", headers={"x-api-key": API_KEY}) as ws:
            assert ws.receive_json()["type"] == "connected"
            res = client.post(
                "/api/v1/stream-sessions",
                json={"sessionId": "qa_ai1_session", "cameraIdentifier": "cam"},
                headers={"X-API-Key": API_KEY},
            )
            assert res.status_code == 200
            msg = _receive_until(ws, "live_streams")
            client.post("/api/v1/stream-sessions/qa_ai1_session/stop", headers={"X-API-Key": API_KEY})

    assert msg is not None, "세션 생성(동기 엔드포인트) 방송이 WS 클라이언트에 도착해야 한다"
    assert any(item["sessionId"] == "qa_ai1_session" for item in msg["data"])


# ---------------------------------------------------------------- AI-2


def test_ws_query_key_allowed_with_warning_without_key_value(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    app = _ws_app(monkeypatch, ws_allow_query_api_key=True)
    with caplog.at_level(logging.WARNING, logger="app.routers.live_ws_router"):
        with TestClient(app) as client:
            with client.websocket_connect(f"/api/v1/ws/live?apiKey={API_KEY}") as ws:
                assert ws.receive_json()["type"] == "connected"
    assert "WS-QUERY-KEY" in caplog.text
    assert API_KEY not in caplog.text


def test_ws_query_key_rejected_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _ws_app(monkeypatch, ws_allow_query_api_key=False)
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(f"/api/v1/ws/live?apiKey={API_KEY}") as ws:
                ws.receive_json()
        assert exc_info.value.code == 1008
        # 헤더는 계속 된다
        with client.websocket_connect("/api/v1/ws/live", headers={"x-api-key": API_KEY}) as ws:
            assert ws.receive_json()["type"] == "connected"


def test_empty_server_api_key_rejects_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    app = _ws_app(monkeypatch, api_key="")
    with TestClient(app) as client:
        assert client.get("/api/v1/live-streams", headers={"X-API-Key": ""}).status_code == 401
        assert client.get("/api/v1/live-streams").status_code == 401
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/api/v1/ws/live", headers={"x-api-key": ""}) as ws:
                ws.receive_json()


def test_config_has_no_hardcoded_api_key_default(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    import app.core.config as config

    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    reloaded = importlib.reload(config)
    try:
        assert reloaded.Settings().api_key == ""
    finally:
        importlib.reload(config)


def test_access_log_redacts_query_api_key() -> None:
    assert redact_api_key("GET /api/v1/ws/live?apiKey=abc123&x=1") == "GET /api/v1/ws/live?apiKey=***&x=1"
    record = logging.LogRecord(
        "uvicorn.error", logging.INFO, __file__, 1,
        '%s - "WebSocket %s" [accepted]', ("1.2.3.4:5", "/api/v1/ws/live?apikey=SECRET"), None,
    )
    ApiKeyRedactingFilter().filter(record)
    assert "SECRET" not in record.getMessage()
    assert "apikey=***" in record.getMessage()


# ---------------------------------------------------------------- AI-3


@pytest.mark.parametrize(("raw", "expected"), [(1, "1"), ("e2eg01", "e2eg01"), (" e2eg01 ", "e2eg01"), (990001, "990001")])
def test_member_id_accepts_string_and_legacy_number(raw, expected) -> None:
    assert TypeAdapter(MemberId).validate_python(raw) == expected


@pytest.mark.parametrize("raw", [True, "", "   ", "x" * 65, None, 1.5])
def test_member_id_rejects_invalid(raw) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(MemberId).validate_python(raw)


def test_chat_request_takes_member_id() -> None:
    assert ChatRequest(userId="e2eg01", message="안녕").userId == "e2eg01"
    assert ChatRequest(userId=1, message="안녕").userId == "1"  # 기존 FE(숫자) 하위 호환


@contextmanager
def _memory_db():
    engine = create_engine("sqlite+pysqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield engine, session
    finally:
        session.close()


def test_chat_logs_endpoint_accepts_string_member_id(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.database.session import get_db
    from app.routers import chat_router

    with _memory_db() as (_, session):
        session.add(ChatLog(chat_no="c-1", user_id="e2eg01", message="m", reply="r"))
        session.add(ChatLog(chat_no="c-2", user_id="1", message="m", reply="r"))
        session.commit()

        app = FastAPI()
        app.include_router(chat_router.router)
        app.dependency_overrides[get_db] = lambda: session
        with TestClient(app) as client:
            res = client.get("/api/v1/chat/logs", params={"userId": "e2eg01"})

    assert res.status_code == 200, res.text  # 예전에는 422
    items = res.json()["data"]
    assert [item["userId"] for item in items] == ["e2eg01"]


def test_reservation_credential_upsert_with_member_id() -> None:
    with _memory_db() as (_, session):
        service = ReservationCredentialService()
        summary = service.upsert_credential(session, user_id="e2eg01", reservation_email=None, api_key="sbk_abc.def123456")
        assert summary.userId == "e2eg01"
        assert service.get_api_key(session, "e2eg01") == "sbk_abc.def123456"


class _FakeConn:
    def __init__(self, types: dict[str, str]) -> None:
        self.types = types
        self.statements: list[str] = []

    def execute(self, stmt, params=None):
        sql = str(stmt)
        self.statements.append(sql)
        value = self.types.get(params["t"]) if params else None
        return type("R", (), {"scalar": lambda _self: value})()


class _FakeEngine:
    def __init__(self, dialect: str, types: dict[str, str]) -> None:
        self.dialect = type("D", (), {"name": dialect})()
        self.conn = _FakeConn(types)

    @contextmanager
    def begin(self):
        yield self.conn


def test_member_id_column_upgrade_alters_only_integer_columns_on_postgres() -> None:
    engine = _FakeEngine("postgresql", {"chat_logs": "integer", "reservation_credentials": "character varying"})
    changed = upgrade_member_id_columns(engine)  # type: ignore[arg-type]
    assert changed == ["chat_logs.user_id"]
    alters = [s for s in engine.conn.statements if s.startswith("ALTER")]
    assert alters == ['ALTER TABLE "chat_logs" ALTER COLUMN "user_id" TYPE VARCHAR(64) USING "user_id"::text']

    again = _FakeEngine("postgresql", {"chat_logs": "character varying", "reservation_credentials": "character varying"})
    assert upgrade_member_id_columns(again) == []  # type: ignore[arg-type]  (멱등)
    assert upgrade_member_id_columns(_FakeEngine("sqlite", {})) == []  # type: ignore[arg-type]
