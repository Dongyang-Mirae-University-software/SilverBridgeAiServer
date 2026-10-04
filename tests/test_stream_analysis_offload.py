from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from tests.clip_helpers import make_jpeg
import app.models.ai_model  # noqa: F401  (AnalysisResult FK 대상 — test_clip_endpoint 와 같은 이유)
import app.models.camera  # noqa: F401
from app.core.config import get_settings
from app.core.security import require_api_key
from app.routers import live_stream_router as router_module
from app.services import stream_session_service as sss
from app.services.session_analysis_store import session_analysis_store

API_KEY = "test-offload-key"


class _SlowDetector:
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def detect_from_jpeg(self, frame_bytes: bytes) -> dict:
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        with self._lock:
            self.active -= 1
        return {
            "detectedType": "fire",
            "confidence": 0.9,
            "danger": True,
            "detections": [],
            "detectedAt": datetime.utcnow().isoformat(),
        }


@pytest.fixture
def detector(monkeypatch: pytest.MonkeyPatch) -> _SlowDetector:
    fake = _SlowDetector(delay=0.4)
    monkeypatch.setattr(sss, "get_fire_smoke_detector", lambda: fake)
    settings = get_settings().model_copy(
        update={"api_key": API_KEY, "fire_smoke_enabled": True, "stream_sample_every_n_frames": 1},
    )
    monkeypatch.setattr(sss, "get_settings", lambda: settings)
    return fake


def _service() -> sss.StreamSessionService:
    return sss.StreamSessionService(db=None, frame_store=sss.frame_store)  # type: ignore[arg-type]


def _fresh(session_id: str) -> str:
    session_analysis_store.clear_session(session_id)
    return session_id


def test_inference_does_not_block_event_loop(detector: _SlowDetector) -> None:
    sid = _fresh("offload_loop")

    async def run() -> tuple[dict | None, float]:
        lags: list[float] = []

        async def ticker() -> None:
            while True:
                t = time.monotonic()
                await asyncio.sleep(0.01)
                lags.append(time.monotonic() - t - 0.01)

        tick = asyncio.create_task(ticker())
        result = await _service().analyze_stream_frame_async(sid, b"jpeg")
        tick.cancel()
        return result, max(lags)

    result, max_lag = asyncio.run(run())

    assert result is not None and result["danger"] is True
    assert max_lag < 0.1, max_lag  # 추론 0.4초 동안에도 루프가 돈다
    assert session_analysis_store.get_result(sid)["detectedType"] == "fire"
    assert not sss._analysis_in_flight.contains(sid)


def test_same_session_frame_is_skipped_while_inference_in_flight(detector: _SlowDetector) -> None:
    sid = _fresh("offload_skip")
    session_analysis_store.set_result(sid, {"detectedType": "normal", "danger": False})

    async def run() -> list:
        service = _service()
        first = asyncio.create_task(service.analyze_stream_frame_async(sid, b"1"))
        await asyncio.sleep(0.05)
        started = time.monotonic()
        second = await service.analyze_stream_frame_async(sid, b"2")
        skipped_in = time.monotonic() - started
        return [await first, second, skipped_in]

    first, second, skipped_in = asyncio.run(run())

    assert detector.calls == 1
    assert second == {"detectedType": "normal", "danger": False}  # 직전 결과
    assert skipped_in < 0.05  # 기다리지 않고 바로 돌아온다
    assert first["detectedType"] == "fire"


def test_inference_runs_one_at_a_time_across_sessions(detector: _SlowDetector) -> None:
    detector.delay = 0.1
    sids = [_fresh(f"offload_multi_{i}") for i in range(3)]

    async def run() -> list:
        service = _service()
        return await asyncio.gather(*(service.analyze_stream_frame_async(s, b"x") for s in sids))

    results = asyncio.run(run())

    assert detector.calls == 3  # 다른 세션은 건너뛰지 않는다
    assert detector.max_active == 1
    assert all(r["danger"] for r in results)


def test_cancelled_request_keeps_in_flight_until_inference_ends(detector: _SlowDetector) -> None:
    sid = _fresh("offload_cancel")

    async def run() -> tuple[bool, bool]:
        task = asyncio.create_task(_service().analyze_stream_frame_async(sid, b"x"))
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.sleep(0.05)
        during = sss._analysis_in_flight.contains(sid)
        await asyncio.sleep(0.5)
        return during, sss._analysis_in_flight.contains(sid)

    during, after = asyncio.run(run())

    assert during is True  # 추론이 아직 돌고 있으니 다음 프레임은 건너뛴다
    assert after is False
    assert session_analysis_store.get_result(sid)["detectedType"] == "fire"


def test_ingest_endpoint_uses_offloaded_inference(detector: _SlowDetector) -> None:
    from app.database.base import Base
    from app.database.session import engine

    Base.metadata.create_all(bind=engine)
    app = FastAPI()
    app.include_router(router_module.router, dependencies=[Depends(require_api_key)])
    app.dependency_overrides[get_settings] = lambda: get_settings().model_copy(update={"api_key": API_KEY})
    headers = {"X-API-Key": API_KEY}
    sid = _fresh("offload_ingest")

    with TestClient(app) as client:
        res = client.post("/api/v1/stream-sessions", json={"sessionId": sid, "cameraIdentifier": "cam"}, headers=headers)
        assert res.status_code == 200, res.text

        results: list[int] = []
        first = threading.Thread(
            target=lambda: results.append(
                client.post(
                    f"/api/v1/stream-sessions/{sid}/frame",
                    files={"frame": ("f.jpg", make_jpeg(64, 48, 0), "image/jpeg")},
                    headers=headers,
                ).status_code,
            ),
        )
        first.start()
        deadline = time.monotonic() + 3
        while not sss._analysis_in_flight.contains(sid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert sss._analysis_in_flight.contains(sid), "첫 프레임 추론이 전용 executor 에서 돌고 있어야 한다"

        started = time.monotonic()
        res = client.post(
            f"/api/v1/stream-sessions/{sid}/frame",
            files={"frame": ("f.jpg", make_jpeg(64, 48, 1), "image/jpeg")},
            headers=headers,
        )
        elapsed = time.monotonic() - started
        first.join(5)

    assert res.status_code == 200 and results == [200]
    assert elapsed < 0.3, elapsed  # 같은 세션 두 번째 프레임은 추론을 기다리지 않는다
    assert detector.calls == 1
    assert session_analysis_store.get_result(sid)["danger"] is True
