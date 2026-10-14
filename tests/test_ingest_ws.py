from __future__ import annotations

import json
import os
import time

import pytest
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

# 라우터가 끌고 오는 모델 메타데이터를 test_clip_endpoint 와 같은 이유로 함께 등록한다.
import app.models.ai_model  # noqa: F401
import app.models.camera  # noqa: F401
from app.core.config import get_settings
from app.core.response import error_response
from app.core.security import require_api_key
from app.routers import live_ingest_ws_router as ingest_module
from app.routers import live_stream_router as stream_module
from app.services import h264_decoder
from app.services.clip_service import parse_jpeg_size
from app.services.stream_session_service import frame_store
from tests.ingest_helpers import frame_message, make_access_units

API_KEY = "test-ingest-key"
HEADERS = {"x-api-key": API_KEY}
_UNITS = make_access_units(seconds=3)  # 45장, 키프레임 15장마다


def _settings(**overrides):
    base = {"api_key": API_KEY, "ingest_ws_enabled": True}
    base.update(overrides)
    return get_settings().model_copy(update=base)


@pytest.fixture
def env(monkeypatch):
    """(client, set_settings) - 라우터가 get_settings()를 직접 부르므로 모듈 참조를 바꾼다."""
    from app.database.base import Base
    from app.database.session import engine

    Base.metadata.create_all(bind=engine)
    state = {"settings": _settings()}
    monkeypatch.setattr(ingest_module, "get_settings", lambda: state["settings"])
    ingest_module._registry.clear()

    app = FastAPI()

    @app.exception_handler(HTTPException)
    async def _handler(_: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "success" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(status_code=exc.status_code, content=error_response(str(exc.detail), "HTTP_ERROR", None))

    app.include_router(stream_module.router, dependencies=[Depends(require_api_key)])
    app.include_router(ingest_module.router)
    app.dependency_overrides[get_settings] = lambda: _settings()

    def configure(**overrides):
        state["settings"] = _settings(**overrides)

    with TestClient(app) as client:
        yield client, configure
    ingest_module._registry.clear()


_counter = 0


def _new_session(client: TestClient) -> str:
    global _counter
    _counter += 1
    session_id = f"ward_ingest{os.getpid()}{_counter:04d}"
    res = client.post(
        "/api/v1/stream-sessions",
        json={"sessionId": session_id, "cameraIdentifier": f"cam-{session_id}", "deviceType": "ipad"},
        headers={"X-API-Key": API_KEY},
    )
    assert res.status_code == 200, res.text
    return session_id


def _config(**overrides) -> str:
    payload = {"type": "config", "codec": "avc1.42E01F", "width": 640, "height": 360, "framerate": 15, "format": "annexb"}
    payload.update(overrides)
    return json.dumps(payload)


def _url(session_id: str) -> str:
    return f"/api/v1/stream-sessions/{session_id}/ingest-ws"


def _expect_close(ws) -> WebSocketDisconnect:
    """close 메시지가 올 때까지 읽는다(그 사이의 텍스트·바이너리는 버린다). TestClient.receive()는 close 를 예외가 아니라 메시지로 준다."""
    while True:
        message = ws.receive()
        if message["type"] == "websocket.close":
            return WebSocketDisconnect(message.get("code", 1000), message.get("reason"))


def _wait(predicate, timeout: float = 8.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


# ---- 단위 ----

def test_parse_config_accepts_contract_and_rejects_others():
    assert ingest_module.parse_config(_config())["framerate"] == 15.0
    assert ingest_module.parse_config(_config(codec="vp8")) is None
    assert ingest_module.parse_config(_config(format="avcc")) is None
    assert ingest_module.parse_config(_config(width=0)) is None
    assert ingest_module.parse_config(_config(width=99999)) is None
    assert ingest_module.parse_config(_config(framerate=0)) is None
    assert ingest_module.parse_config(_config(framerate=True)) is None
    assert ingest_module.parse_config("not json") is None
    assert ingest_module.parse_config(json.dumps({"type": "other"})) is None


def test_split_jpegs_handles_partial_chunks():
    one = b"\xff\xd8" + b"\x01\x02" + b"\xff\xd9"
    two = b"\xff\xd8" + b"\x03" + b"\xff\xd9"
    buf = bytearray(one + two[:3])
    assert h264_decoder.split_jpegs(buf) == [one]
    buf.extend(two[3:])
    assert h264_decoder.split_jpegs(buf) == [two]
    assert len(buf) == 0


# ---- 인증·거절 ----

def test_rejects_wrong_key_and_query_key(env):
    client, _ = env
    sid = _new_session(client)
    with client.websocket_connect(_url(sid), headers={"x-api-key": "wrong"}) as ws:
        assert _expect_close(ws).reason == "AUTH_INVALID_KEY"
    with client.websocket_connect(_url(sid) + f"?apiKey={API_KEY}") as ws:
        assert _expect_close(ws).reason == "AUTH_INVALID_KEY"


def test_disabled_flag_closes_without_touching_frame_path(env):
    client, configure = env
    configure(ingest_ws_enabled=False)
    sid = _new_session(client)
    with client.websocket_connect(_url(sid), headers=HEADERS) as ws:
        assert _expect_close(ws).reason == "INGEST_DISABLED"
    assert ingest_module.active_connection_count() == 0


def test_unknown_session_is_rejected(env):
    client, _ = env
    with client.websocket_connect(_url("ward_doesnotexist0001"), headers=HEADERS) as ws:
        assert _expect_close(ws).reason == "SESSION_NOT_FOUND"


def test_first_message_must_be_config(env):
    client, _ = env
    sid = _new_session(client)
    with client.websocket_connect(_url(sid), headers=HEADERS) as ws:
        ws.send_bytes(frame_message(True, _UNITS[0][1]))
        assert _expect_close(ws).reason == "CONFIG_REQUIRED"


# ---- 정상 경로 ----

def test_decodes_h264_into_jpeg_frames_and_updates_session(env):
    client, _ = env
    sid = _new_session(client)
    _, seq0 = frame_store.get_frame_with_seq(sid)
    with client.websocket_connect(_url(sid), headers=HEADERS) as ws:
        ws.send_text(_config())
        for i, (key, au) in enumerate(_UNITS[:30]):
            ws.send_bytes(frame_message(key, au, i * 66_666))
            time.sleep(0.04)
        assert _wait(lambda: frame_store.get_frame_with_seq(sid)[1] - seq0 >= 25)
        frame, _ = frame_store.get_frame_with_seq(sid)
        assert frame[:2] == b"\xff\xd8"
        assert parse_jpeg_size(frame) == (640, 360)  # SOF0 baseline - 클립 인코더가 읽는 형식
    # 연결을 닫으면 레지스트리가 비고 ffmpeg 도 정리된다.
    assert _wait(lambda: ingest_module.active_connection_count() == 0)


def test_delta_before_keyframe_is_dropped_and_keyframe_requested(env):
    client, _ = env
    sid = _new_session(client)
    _, seq0 = frame_store.get_frame_with_seq(sid)
    with client.websocket_connect(_url(sid), headers=HEADERS) as ws:
        ws.send_text(_config())
        ws.send_bytes(frame_message(False, _UNITS[1][1]))
        msg = ws.receive_text()
        assert json.loads(msg) == {"type": "keyframe_request"}
        assert frame_store.get_frame_with_seq(sid)[1] == seq0


def test_oversized_message_closes_with_1009(env):
    client, configure = env
    configure(ingest_ws_max_message_bytes=4096)
    sid = _new_session(client)
    with client.websocket_connect(_url(sid), headers=HEADERS) as ws:
        ws.send_text(_config())
        ws.send_bytes(b"\x00" * 8 + b"\x01" + b"\x00\x00\x01" + b"\x00" * 5000)
        err = _expect_close(ws)
        assert (err.code, err.reason) == (1009, "MESSAGE_TOO_LARGE")


def test_repeated_garbage_closes_with_bad_payload(env):
    client, configure = env
    configure(ingest_ws_max_consecutive_failures=5)
    sid = _new_session(client)
    with client.websocket_connect(_url(sid), headers=HEADERS) as ws:
        ws.send_text(_config())
        for _ in range(6):
            ws.send_bytes(b"\x00" * 8 + b"\x01" + b"garbage-not-annexb")
        assert _expect_close(ws).reason == "BAD_PAYLOAD"


def test_idle_connection_is_closed(env):
    client, configure = env
    configure(ingest_ws_idle_timeout_sec=3.0)
    sid = _new_session(client)
    with client.websocket_connect(_url(sid), headers=HEADERS) as ws:
        ws.send_text(_config())
        assert _expect_close(ws).reason == "IDLE_TIMEOUT"


# ---- 교체·상한·프로세스 정리 ----

def test_second_connection_for_same_session_replaces_first(env):
    client, _ = env
    sid = _new_session(client)
    with client.websocket_connect(_url(sid), headers=HEADERS) as first:
        first.send_text(_config())
        assert _wait(lambda: ingest_module.active_connection_count() == 1)
        with client.websocket_connect(_url(sid), headers=HEADERS) as second:
            second.send_text(_config())
            err = _expect_close(first)
            assert (err.code, err.reason) == (1013, "REPLACED")
            assert ingest_module.active_connection_count() == 1


def test_total_connection_limit_is_enforced(env):
    client, configure = env
    configure(ingest_ws_max_connections=1)
    first_id, second_id = _new_session(client), _new_session(client)
    with client.websocket_connect(_url(first_id), headers=HEADERS) as first:
        first.send_text(_config())
        assert _wait(lambda: ingest_module.active_connection_count() == 1)
        with client.websocket_connect(_url(second_id), headers=HEADERS) as second:
            err = _expect_close(second)
            assert (err.code, err.reason) == (1013, "BUSY")


def test_ffmpeg_process_is_reaped_on_close(env, monkeypatch):
    client, _ = env
    pids: list[int] = []
    original = h264_decoder.H264ToJpegDecoder.start

    async def recording_start(self):
        await original(self)
        pids.append(self.pid)

    monkeypatch.setattr(h264_decoder.H264ToJpegDecoder, "start", recording_start)
    sid = _new_session(client)
    with client.websocket_connect(_url(sid), headers=HEADERS) as ws:
        ws.send_text(_config())
        ws.send_bytes(frame_message(True, _UNITS[0][1]))
        assert _wait(lambda: len(pids) == 1)

    def gone() -> bool:
        try:
            os.kill(pids[0], 0)
        except ProcessLookupError:
            return True
        # 좀비(회수 전)도 살아 있는 것으로 센다 - wait() 로 회수했는지 본다.
        return False

    assert _wait(gone, 6.0)


def test_reconfig_restarts_decoder_without_dropping_connection(env):
    client, _ = env
    sid = _new_session(client)
    _, seq0 = frame_store.get_frame_with_seq(sid)
    small = make_access_units(width=320, height=180, seconds=1)
    with client.websocket_connect(_url(sid), headers=HEADERS) as ws:
        ws.send_text(_config())
        for i, (key, au) in enumerate(_UNITS[:8]):
            ws.send_bytes(frame_message(key, au))
            time.sleep(0.04)
        assert _wait(lambda: frame_store.get_frame_with_seq(sid)[1] - seq0 >= 5)
        ws.send_text(_config(width=320, height=180))
        for key, au in small[:8]:
            ws.send_bytes(frame_message(key, au))
            time.sleep(0.04)
        assert _wait(lambda: parse_jpeg_size(frame_store.get_frame_with_seq(sid)[0] or b"") == (320, 180))


def test_rate_limit_drops_delta_and_waits_for_next_keyframe(env):
    client, _ = env
    sid = _new_session(client)
    with client.websocket_connect(_url(sid), headers=HEADERS) as ws:
        ws.send_text(_config(framerate=1))  # 초당 1.5장, 버스트 2장
        ws.send_bytes(frame_message(True, _UNITS[0][1]))
        for key, au in _UNITS[1:12]:
            ws.send_bytes(frame_message(key, au))
        msg = ws.receive_text()
        assert json.loads(msg) == {"type": "keyframe_request"}
        # 키프레임이 오면 다시 받아들인다(연결은 유지).
        ws.send_bytes(frame_message(True, _UNITS[15][1]))
        assert ingest_module.active_connection_count() == 1


def test_undecodable_stream_closes_with_decode_failed(env):
    client, configure = env
    configure(ingest_ws_max_consecutive_failures=5)
    sid = _new_session(client)
    junk = b"\x00\x00\x00\x01" + bytes(range(1, 60))
    with client.websocket_connect(_url(sid), headers=HEADERS) as ws:
        ws.send_text(_config())
        for _ in range(8):
            ws.send_bytes(frame_message(True, junk))
        assert _expect_close(ws).reason == "DECODE_FAILED"
