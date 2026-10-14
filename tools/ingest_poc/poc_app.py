"""H.264 수신 PoC 시험용 최소 앱 (운영 앱이 아니다).

운영 `app.main` 은 기동 때 의료 LLM·GPU·감지 모델을 올린다. 변환·수신 경로(ingest-ws)와 기존 /frame 경로의 비용·지연·프레임
간격만 보려면 그런 것이 필요 없어, 같은 라우터를 그대로 붙인 가벼운 앱을 따로 둔다. 감지 모델은 올리지 않는다
(FIRE_SMOKE_ENABLED·KNIFE_ENABLED·FALL_ENABLED=false) - 감지 일치는 모델이 있는 서버에서 따로 본다.

추가로 /poc/* 계측 경로를 둔다(시험 전용, API 키 필요):
  POST /poc/record/{sessionId}   이후 set_frame 으로 저장되는 프레임(JPEG)을 디스크에 기록 시작
  POST /poc/stop/{sessionId}     기록 종료 + 요약(JSON)
요약: 프레임 수, 바이트, 도착 간격(ms) 분포(p50/p99/max), 첫~마지막 시각.
가상 장면만 쓴다(실제 가정 영상 금지).
"""

from __future__ import annotations

import os
import statistics
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI

import app.models.ai_model  # noqa: F401
import app.models.camera  # noqa: F401
from app.core.security import require_api_key
from app.database.base import Base
from app.database.session import engine
from app.routers.live_ingest_ws_router import router as ingest_router
from app.routers.live_stream_router import router as stream_router
from app.services import stream_session_service as sss
from app.services.live_ws_manager import live_ws_manager

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")  # [INGEST-WS] 집계 로그를 보려고

RECORD_DIR = Path(os.getenv("POC_RECORD_DIR", "/tmp/poc_frames"))
_records: dict[str, dict] = {}
_lock = threading.Lock()

_original_set_frame = sss.StreamFrameStore.set_frame


def _recording_set_frame(self, session_id: str, frame_bytes: bytes):
    result = _original_set_frame(self, session_id, frame_bytes)
    with _lock:
        rec = _records.get(session_id)
        if rec is not None:
            index = rec["count"]
            rec["count"] += 1
            rec["bytes"] += len(frame_bytes)
            rec["times"].append(time.monotonic())
            (rec["dir"] / f"{index:06d}.jpg").write_bytes(frame_bytes)
    return result


sss.StreamFrameStore.set_frame = _recording_set_frame  # type: ignore[method-assign]

# 감지 결과 기록(감지 일치 시험): 분석이 끝날 때마다 (그때까지 저장된 프레임 수, 대표 판정, 종류별 판정)을 남긴다.
# 프레임 번호로 두 경로(JPEG·H.264)의 같은 원본 구간을 맞춰 비교한다.
_original_detect = sss.StreamSessionService._detect_and_store


def _recording_detect(self, session_id: str, frame_bytes: bytes):
    payload = _original_detect(self, session_id, frame_bytes)
    with _lock:
        rec = _records.get(session_id)
        if rec is not None:
            kinds = {}
            for item in payload.get("results") or []:
                if isinstance(item, dict):
                    kinds[str(item.get("kind") or item.get("detectedType"))] = {
                        "type": item.get("detectedType"),
                        "conf": item.get("confidence"),
                        "danger": item.get("danger"),
                    }
            rec["analysis"].append({
                "idx": rec["count"],
                "t": round(time.monotonic() - rec["t0"], 3),
                "type": payload.get("detectedType"),
                "conf": payload.get("confidence"),
                "danger": bool(payload.get("danger")),
                "boxes": len(payload.get("detections") or []),
                "kinds": kinds,
                "quality": (payload.get("frameQuality") or {}).get("ok"),
            })
    return payload


sss.StreamSessionService._detect_and_store = _recording_detect  # type: ignore[method-assign]


@asynccontextmanager
async def lifespan(_: FastAPI):
    import asyncio

    live_ws_manager.bind_loop(asyncio.get_running_loop())
    Base.metadata.create_all(bind=engine)
    # 감지 모델은 켜진 것만 올린다(운영 main.py 와 같은 try_load). 세 종류가 모두 꺼져 있으면 아무것도 올리지 않는다.
    from app.core.config import get_settings

    cfg = get_settings()
    if cfg.fire_smoke_enabled:
        from app.services.fire_smoke_detection_service import get_fire_smoke_detector

        get_fire_smoke_detector().try_load()
    if cfg.knife_enabled:
        from app.services.knife_detection_service import get_knife_detector

        get_knife_detector().try_load()
    if cfg.fall_enabled:
        from app.services.fall_detection_service import get_fall_detector

        get_fall_detector().try_load()
    yield


app = FastAPI(title="ingest-ws PoC", lifespan=lifespan)
app.include_router(stream_router, dependencies=[Depends(require_api_key)])
app.include_router(ingest_router)


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.post("/poc/record/{session_id}", dependencies=[Depends(require_api_key)])
def record_start(session_id: str) -> dict:
    target = RECORD_DIR / session_id
    target.mkdir(parents=True, exist_ok=True)
    for old in target.glob("*.jpg"):
        old.unlink()
    with _lock:
        _records[session_id] = {"dir": target, "count": 0, "bytes": 0, "times": [], "analysis": [], "t0": time.monotonic()}
    return {"recording": session_id, "dir": str(target)}


@app.post("/poc/stop/{session_id}", dependencies=[Depends(require_api_key)])
def record_stop(session_id: str) -> dict:
    with _lock:
        rec = _records.pop(session_id, None)
    if rec is None:
        return {"error": "not recording"}
    import json

    (rec["dir"] / "analysis.json").write_text(json.dumps(rec["analysis"]))
    times = rec["times"]
    gaps = [(b - a) * 1000.0 for a, b in zip(times, times[1:])]
    gaps_sorted = sorted(gaps)

    def pct(p: float) -> float | None:
        return round(gaps_sorted[min(len(gaps_sorted) - 1, int(len(gaps_sorted) * p))], 1) if gaps_sorted else None

    return {
        "frames": rec["count"],
        "bytes": rec["bytes"],
        "durationSec": round(times[-1] - times[0], 2) if len(times) > 1 else 0,
        "gapMs": {
            "p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99),
            "max": round(max(gaps), 1) if gaps else None,
            "mean": round(statistics.fmean(gaps), 1) if gaps else None,
        },
        "analysisCount": len(rec["analysis"]),
        "dir": str(rec["dir"]),
    }
