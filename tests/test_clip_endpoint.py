from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timedelta

import pytest
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from tests.clip_helpers import WEBM_MAGIC, fake_webm, make_jpeg
# 라우터가 끌고 오는 AnalysisResult 는 cameras·ai_models 를 FK 로 참조한다. app.main 에선 다른 라우터가
# 같이 임포트해 주지만 여기선 빠지므로, 공유 metadata 가 깨지지 않게(다른 테스트의 create_all) 함께 등록한다.
import app.models.ai_model  # noqa: F401
import app.models.camera  # noqa: F401
from app.core.config import get_settings
from app.core.response import error_response
from app.core.security import require_api_key
from app.routers import live_stream_router as router_module
from app.schemas.clip_schema import ClipRequest
from app.services.clip_buffer import ClipFrameBuffer, clip_frame_buffer
from app.services.clip_service import ClipEncodeError, ClipError, ClipService, ffmpeg_available

API_KEY = "test-clip-key"
HEADERS = {"X-API-Key": API_KEY}


def _settings(**overrides):
    return get_settings().model_copy(update={"api_key": API_KEY, **overrides})


def _make_app() -> FastAPI:
    # app.main 은 모델 로딩까지 끌고 와서, 라우터와 예외 핸들러(main.py 와 같은 규칙)만 붙인다.
    app = FastAPI()

    @app.exception_handler(HTTPException)
    async def _handler(_: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "success" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(status_code=exc.status_code, content=error_response(str(exc.detail), "HTTP_ERROR", None))

    app.include_router(router_module.router, dependencies=[Depends(require_api_key)])
    app.dependency_overrides[get_settings] = lambda: _settings()
    return app


@pytest.fixture
def client():
    from app.database.base import Base
    from app.database.session import engine

    Base.metadata.create_all(bind=engine)  # 세션 생성 응답이 analysis_results 를 조회한다(main.py lifespan 과 동일)
    with TestClient(_make_app()) as c:
        yield c


def _start_session(client: TestClient, session_id: str) -> None:
    res = client.post(
        "/api/v1/stream-sessions",
        json={"sessionId": session_id, "cameraIdentifier": "cam-test"},
        headers=HEADERS,
    )
    assert res.status_code == 200, res.text


def _fill(session_id: str, start: datetime, seconds: float, fps: int, size=(641, 361)) -> None:
    for i in range(int(seconds * fps) + 1):
        clip_frame_buffer.add(session_id, start + timedelta(seconds=i / fps), make_jpeg(size[0], size[1], i))


def _past_detection(session_id: str) -> str:
    """6초 전 감지 — 앞 3초·뒤 2초 프레임이 모두 버퍼에 있는 상태로 만든다."""
    detected = (datetime.utcnow() - timedelta(seconds=6)).replace(microsecond=0)  # 직렬화(ms)와 경계 일치
    _fill(session_id, detected - timedelta(seconds=4), 7, 5)
    return detected.isoformat(timespec="milliseconds") + "Z"


def _use_service(monkeypatch: pytest.MonkeyPatch, service: ClipService) -> None:
    monkeypatch.setattr(router_module, "clip_service", service)


@pytest.mark.skipif(not ffmpeg_available(), reason="imageio-ffmpeg 미설치")
def test_clip_200_returns_webm_with_headers(client: TestClient) -> None:
    _start_session(client, "clip_ok")
    detected = _past_detection("clip_ok")

    res = client.post("/api/v1/live-streams/clip_ok/clips", json={"detectedAt": detected}, headers=HEADERS)

    assert res.status_code == 200, res.text
    assert res.headers["content-type"] == "video/webm"
    assert res.content[:4] == WEBM_MAGIC
    assert res.headers["x-clip-frames"] == "26"  # 5fps × 5초, 양 끝 포함
    assert res.headers["x-clip-duration-ms"] == "5000"
    assert (res.headers["x-clip-width"], res.headers["x-clip-height"]) == ("640", "360")
    assert res.headers["x-clip-started-at"].endswith("Z")
    assert res.headers["cache-control"] == "no-store"


def test_ingest_endpoint_feeds_clip_buffer(client: TestClient) -> None:
    _start_session(client, "clip_ingest")
    for i in range(3):
        res = client.post(
            "/api/v1/stream-sessions/clip_ingest/frame",
            files={"frame": ("f.jpg", make_jpeg(64, 48, i), "image/jpeg")},
            headers=HEADERS,
        )
        assert res.status_code == 200, res.text
    assert clip_frame_buffer.latest_time("clip_ingest") is not None

    client.post("/api/v1/stream-sessions/clip_ingest/stop", headers=HEADERS)
    assert clip_frame_buffer.latest_time("clip_ingest") is None  # stop 시 버퍼 삭제


def test_clip_401_without_api_key(client: TestClient) -> None:
    res = client.post("/api/v1/live-streams/any/clips", json={})
    assert res.status_code == 401
    assert res.json()["errorCode"] == "AUTH_INVALID_KEY"


def test_clip_404_unknown_session(client: TestClient) -> None:
    res = client.post("/api/v1/live-streams/no_such_session/clips", json={}, headers=HEADERS)
    assert res.status_code == 404
    assert res.json()["errorCode"] == "STREAM_SESSION_NOT_FOUND"


def test_clip_409_when_window_has_no_frames(client: TestClient) -> None:
    _start_session(client, "clip_empty")
    old = (datetime.utcnow() - timedelta(minutes=5)).isoformat() + "Z"
    res = client.post("/api/v1/live-streams/clip_empty/clips", json={"detectedAt": old}, headers=HEADERS)
    assert res.status_code == 409
    body = res.json()
    assert body["errorCode"] == "CLIP_NOT_ENOUGH_FRAMES" and body["success"] is False


def test_clip_409_after_session_stopped(client: TestClient) -> None:
    _start_session(client, "clip_stopped")
    detected = _past_detection("clip_stopped")
    client.post("/api/v1/stream-sessions/clip_stopped/stop", headers=HEADERS)
    res = client.post("/api/v1/live-streams/clip_stopped/clips", json={"detectedAt": detected}, headers=HEADERS)
    assert res.status_code == 409


@pytest.mark.parametrize(
    "body",
    [b"{not json", b"[1,2]", b'{"preSeconds": 9}', b'{"postSeconds": 4}',
     b'{"preSeconds": 8, "postSeconds": 3}', b'{"preSeconds": 7.5, "postSeconds": 2.6}', b'{"preSeconds":0,"postSeconds":0}',
     b'{"detectedAt": "nope"}'],
)
def test_clip_422_invalid_params(client: TestClient, body: bytes) -> None:
    _start_session(client, "clip_bad")
    res = client.post(
        "/api/v1/live-streams/clip_bad/clips",
        content=body,
        headers={**HEADERS, "Content-Type": "application/json"},
    )
    assert res.status_code == 422
    assert res.json()["errorCode"] == "CLIP_INVALID_PARAMS"


def test_clip_503_when_disabled(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(router_module, "get_settings", lambda: _settings(clip_enabled=False))
    res = client.post("/api/v1/live-streams/any/clips", json={}, headers=HEADERS)
    assert res.status_code == 503
    assert res.json()["errorCode"] == "CLIP_DISABLED"


def test_clip_500_on_encode_failure(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(*_: object) -> bytes:
        raise ClipEncodeError("timeout")

    service = ClipService(_settings(), clip_frame_buffer, encoder=failing)
    _use_service(monkeypatch, service)
    _start_session(client, "clip_fail")
    detected = _past_detection("clip_fail")

    res = client.post("/api/v1/live-streams/clip_fail/clips", json={"detectedAt": detected}, headers=HEADERS)

    assert res.status_code == 500
    assert res.json()["errorCode"] == "CLIP_ENCODE_FAILED"
    assert service.inflight == 0


def test_clip_429_when_admission_full(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    gate = threading.Event()

    def blocking(*_: object) -> bytes:
        gate.wait(5)
        return fake_webm()

    service = ClipService(_settings(clip_max_concurrency=1, clip_queue_max=0), clip_frame_buffer, encoder=blocking)
    _use_service(monkeypatch, service)
    _start_session(client, "clip_busy")
    detected = _past_detection("clip_busy")
    url = "/api/v1/live-streams/clip_busy/clips"

    results: list[int] = []
    first = threading.Thread(
        target=lambda: results.append(client.post(url, json={"detectedAt": detected}, headers=HEADERS).status_code),
    )
    first.start()
    deadline = time.monotonic() + 5
    while service.inflight == 0 and time.monotonic() < deadline:
        time.sleep(0.01)

    res = client.post(url, json={"detectedAt": detected}, headers=HEADERS)
    assert res.status_code == 429
    assert res.json()["errorCode"] == "CLIP_BUSY"

    gate.set()
    first.join(5)
    assert results == [200]
    assert service.inflight == 0


def test_ingest_is_not_delayed_while_encoding(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def slow(*_: object) -> bytes:
        time.sleep(1.5)  # 인코딩은 전용 스레드에서 — 이벤트 루프를 막지 않아야 한다
        return fake_webm()

    service = ClipService(_settings(), clip_frame_buffer, encoder=slow)
    _use_service(monkeypatch, service)
    _start_session(client, "clip_slow")
    detected = _past_detection("clip_slow")

    worker = threading.Thread(
        target=lambda: client.post("/api/v1/live-streams/clip_slow/clips", json={"detectedAt": detected}, headers=HEADERS),
    )
    worker.start()
    while service.inflight == 0:
        time.sleep(0.01)
    time.sleep(0.1)

    started = time.monotonic()
    res = client.post(
        "/api/v1/stream-sessions/clip_slow/frame",
        files={"frame": ("f.jpg", make_jpeg(64, 48), "image/jpeg")},
        headers=HEADERS,
    )
    elapsed = time.monotonic() - started
    worker.join(5)

    assert res.status_code == 200
    assert elapsed < 0.5, elapsed


# ---- 서비스 단위(동시성·대기·스냅샷 순서) ----


def _service_buffer() -> ClipFrameBuffer:
    return ClipFrameBuffer(buffer_seconds=10, max_frames=100)


def _fill_buffer(buf: ClipFrameBuffer, sid: str, start: datetime, seconds: float, fps: int) -> None:
    for i in range(int(seconds * fps) + 1):
        buf.add(sid, start + timedelta(seconds=i / fps), make_jpeg(32, 32, i))


def test_service_caps_concurrent_encodes() -> None:
    lock = threading.Lock()
    active = {"now": 0, "max": 0}

    def tracking(*_: object) -> bytes:
        with lock:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        time.sleep(0.2)
        with lock:
            active["now"] -= 1
        return fake_webm()

    buf = _service_buffer()
    detected = datetime.utcnow() - timedelta(seconds=6)
    _fill_buffer(buf, "s", detected - timedelta(seconds=4), 7, 5)
    service = ClipService(_settings(clip_max_concurrency=2, clip_queue_max=10), buf, encoder=tracking)
    req = ClipRequest(detectedAt=detected)

    async def run(n: int) -> list[object]:
        return await asyncio.gather(*(service.create_clip("s", req) for _ in range(n)), return_exceptions=True)

    results = asyncio.run(run(14))
    busy = [r for r in results if isinstance(r, ClipError)]
    ok = [r for r in results if not isinstance(r, Exception)]

    assert active["max"] == 2
    assert len(ok) == 12  # 동시 2 + 대기열 10
    assert len(busy) == 2 and all(e.status_code == 429 for e in busy)
    assert service.inflight == 0


def test_service_waits_for_post_window_frames() -> None:
    buf = _service_buffer()
    captured: dict[str, int] = {}

    def counting(frames, *_: object) -> bytes:
        captured["n"] = len(frames)
        return fake_webm()

    service = ClipService(_settings(), buf, encoder=counting)
    detected = datetime.utcnow()
    _fill_buffer(buf, "s", detected - timedelta(seconds=3), 3, 5)  # 앞 3초만 있음

    async def run() -> None:
        loop = asyncio.get_running_loop()

        def feed_post() -> None:
            for i in range(1, 6):  # 뒤 1초(0.2초 간격) — 구간 끝 이후 프레임까지
                buf.add("s", detected + timedelta(seconds=i * 0.2), make_jpeg(32, 32, 100 + i))

        loop.call_later(0.3, feed_post)
        await service.create_clip("s", ClipRequest(detectedAt=detected, preSeconds=3, postSeconds=1))

    started = time.monotonic()
    asyncio.run(run())
    assert captured["n"] == 16 + 5  # 앞 3초 16장 + 뒤 1초 5장
    assert time.monotonic() - started < 2.0  # post+1초 안에 끝남


def test_service_wait_is_bounded_when_stream_stalls() -> None:
    buf = _service_buffer()
    service = ClipService(_settings(), buf, encoder=fake_webm)
    detected = datetime.utcnow()
    _fill_buffer(buf, "s", detected - timedelta(seconds=3), 3, 5)  # 이후 프레임이 끊김

    started = time.monotonic()
    result = asyncio.run(service.create_clip("s", ClipRequest(detectedAt=detected, preSeconds=3, postSeconds=1)))
    elapsed = time.monotonic() - started

    assert 1.9 <= elapsed < 2.6  # 최대 post + 1초
    assert result.frames == 16


def test_service_snapshots_before_queueing() -> None:
    buf = _service_buffer()
    gate = threading.Event()
    counts: list[int] = []

    def gated(frames, *_: object) -> bytes:
        gate.wait(5)
        counts.append(len(frames))
        return fake_webm()

    service = ClipService(_settings(clip_max_concurrency=1, clip_queue_max=5), buf, encoder=gated)
    detected = datetime.utcnow() - timedelta(seconds=6)
    _fill_buffer(buf, "s", detected - timedelta(seconds=4), 7, 5)
    req = ClipRequest(detectedAt=detected)

    async def run() -> None:
        first = asyncio.create_task(service.create_clip("s", req))
        second = asyncio.create_task(service.create_clip("s", req))
        await asyncio.sleep(0.2)  # 둘 다 스냅샷을 뜨고 둘째는 큐에서 대기
        buf.clear("s")  # 큐 대기 중 링버퍼가 비어도
        gate.set()
        await asyncio.gather(first, second)

    asyncio.run(run())
    assert counts == [26, 26]


def test_service_queue_timeout_fails_without_encoding() -> None:
    buf = _service_buffer()
    calls: list[int] = []

    def slow(*_: object) -> bytes:
        calls.append(1)
        time.sleep(0.6)
        return fake_webm()

    service = ClipService(_settings(clip_max_concurrency=1, clip_queue_max=5, clip_timeout_sec=0.4), buf, encoder=slow)
    detected = datetime.utcnow() - timedelta(seconds=6)
    _fill_buffer(buf, "s", detected - timedelta(seconds=4), 7, 5)
    req = ClipRequest(detectedAt=detected)

    async def run() -> list[object]:
        return await asyncio.gather(service.create_clip("s", req), service.create_clip("s", req), return_exceptions=True)

    results = asyncio.run(run())
    assert len(calls) == 1  # 둘째는 큐에서 상한을 다 써서 인코딩하지 않음
    errors = [r for r in results if isinstance(r, ClipError)]
    assert len(errors) == 1 and errors[0].error_code == "CLIP_ENCODE_FAILED"
    assert service.inflight == 0


def test_service_wraps_unexpected_encoder_errors() -> None:
    def broken(*_: object) -> bytes:
        raise FileNotFoundError("tmp dir missing")

    buf = _service_buffer()
    service = ClipService(_settings(), buf, encoder=broken)
    detected = datetime.utcnow() - timedelta(seconds=6)
    _fill_buffer(buf, "s", detected - timedelta(seconds=4), 7, 5)

    with pytest.raises(ClipError) as exc_info:
        asyncio.run(service.create_clip("s", ClipRequest(detectedAt=detected)))
    assert (exc_info.value.status_code, exc_info.value.error_code) == (500, "CLIP_ENCODE_FAILED")
    assert service.inflight == 0


def test_clip_pre_8_seconds_allowed(client: TestClient) -> None:
    _start_session(client, "clip_pre8")
    detected = (datetime.utcnow() - timedelta(seconds=5)).replace(microsecond=0).isoformat() + "Z"
    res = client.post(
        "/api/v1/live-streams/clip_pre8/clips",
        json={"detectedAt": detected, "preSeconds": 8, "postSeconds": 1},
        headers=HEADERS,
    )
    assert res.status_code != 422  # 구간 검증 통과(프레임이 없으면 409)


def test_clip_total_over_buffer_422(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _start_session(client, "clip_buf")
    monkeypatch.setitem(client.app.dependency_overrides, get_settings, lambda: _settings(clip_buffer_seconds=6.0))
    monkeypatch.setattr(router_module, "get_settings", lambda: _settings(clip_buffer_seconds=6.0))
    res = client.post(
        "/api/v1/live-streams/clip_buf/clips", json={"preSeconds": 5, "postSeconds": 2}, headers=HEADERS,
    )
    assert res.status_code == 422
    assert res.json()["errorCode"] == "CLIP_INVALID_PARAMS"


def test_clip_request_defaults_unchanged() -> None:
    req = ClipRequest()
    assert (req.preSeconds, req.postSeconds) == (3, 2)
    assert ClipRequest(preSeconds=8, postSeconds=2).preSeconds == 8
