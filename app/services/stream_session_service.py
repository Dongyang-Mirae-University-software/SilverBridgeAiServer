from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Lock
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.response import error_response
from app.services.bbox_overlay import bbox_overlay
from app.services.clip_buffer import clip_frame_buffer
from app.services.fall_detection_service import get_fall_detector
from app.services.fall_hold_tracker import fall_hold_tracker
from app.services.fire_smoke_detection_service import get_fire_smoke_detector
from app.services.knife_detection_service import get_knife_detector
from app.services.live_detection_merge import merge_kind_results
from app.services.session_analysis_store import session_analysis_store
from app.models.analysis_result import AnalysisResult
from app.models.stream_session import StreamSession

_LOG = logging.getLogger(__name__)
# 낙상 유지 판정의 시계(분석 시각). 테스트가 바꿔 끼운다.
_analysis_clock = time.monotonic


class StreamFrameStore:
    def __init__(self) -> None:
        self._frames: dict[str, bytes] = {}
        self._frame_times: dict[str, datetime] = {}
        self._last_epoch: dict[str, float] = {}
        self._fps_map: dict[str, float] = {}
        self._viewers: dict[str, int] = {}
        self._seq: dict[str, int] = {}
        self._lock = Lock()

    def set_frame(self, session_id: str, frame_bytes: bytes) -> tuple[datetime, float]:
        now = datetime.utcnow()
        now_epoch = time.time()
        with self._lock:
            self._frames[session_id] = frame_bytes
            self._seq[session_id] = self._seq.get(session_id, 0) + 1
            self._frame_times[session_id] = now
            prev = self._last_epoch.get(session_id)
            if prev is None or now_epoch <= prev:
                fps = self._fps_map.get(session_id, 0.0)
            else:
                fps = round(1.0 / (now_epoch - prev), 2)
            self._last_epoch[session_id] = now_epoch
            self._fps_map[session_id] = fps
            self._viewers.setdefault(session_id, 0)
        # 클립용 링버퍼 — 같은 수신 시각을 써야 analyzedAt 으로 구간을 자를 수 있다.
        if get_settings().clip_enabled:
            clip_frame_buffer.add(session_id, now, frame_bytes)
        return now, fps

    def get_frame(self, session_id: str) -> bytes | None:
        with self._lock:
            return self._frames.get(session_id)

    def get_frame_with_seq(self, session_id: str) -> tuple[bytes | None, int]:
        """원본 JPEG 와 프레임 순번(같은 프레임이면 같은 값) — 박스 그리기 캐시의 프레임 식별자."""
        with self._lock:
            return self._frames.get(session_id), self._seq.get(session_id, 0)

    def get_fps(self, session_id: str) -> float:
        with self._lock:
            return self._fps_map.get(session_id, 0.0)

    def increment_viewer(self, session_id: str) -> None:
        with self._lock:
            self._viewers[session_id] = self._viewers.get(session_id, 0) + 1

    def decrement_viewer(self, session_id: str) -> None:
        with self._lock:
            if session_id not in self._viewers:
                return
            self._viewers[session_id] = max(0, self._viewers[session_id] - 1)

    def get_viewer_count(self, session_id: str) -> int:
        with self._lock:
            return self._viewers.get(session_id, 0)


class StreamSessionService:
    def __init__(self, db: Session, frame_store: StreamFrameStore) -> None:
        self.db = db
        self.frame_store = frame_store
        self.settings = get_settings()
        self._state_store = stream_session_state_store

    @property
    def use_memory_state(self) -> bool:
        return self.settings.stream_state_backend.lower() == "memory"

    def _viewer_url(self, session_id: str) -> str:
        if self.settings.mediamtx_enabled and self.settings.mediamtx_webrtc_view_base:
            return f"{self.settings.mediamtx_webrtc_view_base.rstrip('/')}/{session_id}"
        return f"/api/v1/live-streams/{session_id}/mjpeg"

    def _hls_url(self, session_id: str) -> str | None:
        if self.settings.mediamtx_enabled and self.settings.mediamtx_hls_view_base:
            return f"{self.settings.mediamtx_hls_view_base.rstrip('/')}/{session_id}/index.m3u8"
        return None

    def _ingest_url(self, session_id: str) -> str | None:
        if self.settings.mediamtx_enabled and self.settings.mediamtx_webrtc_ingest_base:
            return f"{self.settings.mediamtx_webrtc_ingest_base.rstrip('/')}/{session_id}"
        return None

    def create_or_restart(self, session_id: str, camera_identifier: str, device_type: str) -> StreamSessionState:
        # 같은 sessionId 재시작 시 이전 송출의 프레임이 클립에 섞이지 않게 비운다.
        clip_frame_buffer.clear(session_id)
        fall_hold_tracker.clear(session_id)
        if self.use_memory_state:
            return self._state_store.create_or_restart(session_id, camera_identifier, device_type)

        session = self.db.query(StreamSession).filter(StreamSession.session_id == session_id).first()
        now = datetime.utcnow()
        if session:
            session.camera_identifier = camera_identifier
            session.device_type = device_type
            session.status = "running"
            session.started_at = now
            session.last_frame_at = None
            session.stopped_at = None
            session.fps = 0.0
            session.viewer_count = 0
        else:
            session = StreamSession(
                session_id=session_id,
                camera_identifier=camera_identifier,
                device_type=device_type,
                status="running",
                started_at=now,
                is_analyzing=1,
            )
            self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def get_by_session_id(self, session_id: str) -> StreamSessionState | None:
        if self.use_memory_state:
            return self._state_store.get(session_id)

        session = self.db.query(StreamSession).filter(StreamSession.session_id == session_id).first()
        if not session:
            return None
        return StreamSessionState(
            session_id=session.session_id,
            camera_identifier=session.camera_identifier,
            device_type=session.device_type,
            status=session.status,
            started_at=session.started_at,
            last_frame_at=session.last_frame_at,
            stopped_at=session.stopped_at,
            fps=session.fps,
            viewer_count=session.viewer_count,
            is_analyzing=bool(session.is_analyzing),
        )

    def ingest_frame(self, session: StreamSessionState, frame_bytes: bytes) -> StreamSessionState:
        frame_time, fps = self.frame_store.set_frame(session.session_id, frame_bytes)
        if self.use_memory_state:
            return self._state_store.ingest_frame(session.session_id, frame_time, fps)

        row = self.db.query(StreamSession).filter(StreamSession.session_id == session.session_id).first()
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_response("스트림 세션을 찾을 수 없습니다.", "STREAM_SESSION_NOT_FOUND", None),
            )
        row.last_frame_at = frame_time
        row.fps = fps
        row.status = "running"
        row.stopped_at = None
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        session.last_frame_at = frame_time
        session.fps = fps
        session.status = "running"
        session.stopped_at = None
        return session

    def analyze_stream_frame(self, session_id: str, frame_bytes: bytes) -> dict[str, Any] | None:
        if not self._any_detector_enabled():
            return session_analysis_store.get_result(session_id)
        if not session_analysis_store.should_analyze(session_id, self.settings.stream_sample_every_n_frames):
            return session_analysis_store.get_result(session_id)
        return self._detect_and_store(session_id, frame_bytes)

    async def analyze_stream_frame_async(self, session_id: str, frame_bytes: bytes) -> dict[str, Any] | None:
        """analyze_stream_frame 의 이벤트 루프 비차단판 — 프레임 수신(async) 경로용.

        YOLO 추론을 루프 안에서 돌리면 추론하는 동안 서버 전체(다른 카메라 수신·WS·클립 대기)가 멈춘다.
        추론은 전용 1스레드에서 돌린다(감지기가 잠금으로 직렬화하므로 스레드를 늘려도 빨라지지 않고, 공용
        스레드풀에 태우면 잠금 대기 스레드가 동기 엔드포인트 몫까지 차지한다).
        같은 세션의 추론이 아직 진행 중이면 이 프레임은 분석하지 않고 직전 결과를 돌려준다 — 루프가 막혀
        저절로 걸리던 속도 조절을 대신해, 대기열이 세션 수를 넘지 않게 한다.
        """
        if not self._any_detector_enabled():
            return session_analysis_store.get_result(session_id)
        if not session_analysis_store.should_analyze(session_id, self.settings.stream_sample_every_n_frames):
            return session_analysis_store.get_result(session_id)
        if not _analysis_in_flight.try_enter(session_id):
            return session_analysis_store.get_result(session_id)
        try:
            future = _analysis_executor.submit(self._detect_and_store, session_id, frame_bytes)
        except BaseException:
            _analysis_in_flight.leave(session_id)
            raise
        # 요청이 취소돼도 추론은 끝까지 가므로, 표시는 추론이 실제로 끝날 때 푼다.
        future.add_done_callback(lambda _: _analysis_in_flight.leave(session_id))
        return await asyncio.wrap_future(future)

    def _any_detector_enabled(self) -> bool:
        s = self.settings
        return s.fire_smoke_enabled or s.knife_enabled or s.fall_enabled

    def _enabled_detectors(self) -> list[tuple[str, Any]]:
        detectors: list[tuple[str, Any]] = []
        if self.settings.fire_smoke_enabled:
            detectors.append(("fire", get_fire_smoke_detector()))
        if self.settings.knife_enabled:
            detectors.append(("knife", get_knife_detector()))
        if self.settings.fall_enabled:
            detectors.append(("fall", get_fall_detector()))
        return detectors

    def _detect_and_store(self, session_id: str, frame_bytes: bytes) -> dict[str, Any]:
        """켜진 감지기를 같은 프레임에 차례로 돌려 대표 결과 하나로 합친다(전용 1스레드 안).

        응답의 detectedType/confidence/danger/detections/analyzedAt 은 백엔드 계약 그대로다 — 대표 선택
        규칙은 live_detection_merge.merge_kind_results. 종류별 결과는 results 에 따로 싣는다.
        """
        kind_results: list[tuple[str, dict[str, Any]]] = []
        timings_ms: dict[str, float] = {}
        for kind, detector in self._enabled_detectors():
            started = time.perf_counter()
            try:
                result = detector.detect_from_jpeg(frame_bytes)
            except Exception:  # noqa: BLE001 - 한 종류의 오류가 다른 종류(특히 화재) 판정을 막지 않게
                _LOG.exception("[LIVE-DETECT-FAILED] kind=%s sessionId=%s", kind, session_id)
                result = {"detectedType": "unknown", "confidence": 0.0, "danger": False, "detections": []}
            timings_ms[kind] = (time.perf_counter() - started) * 1000
            if kind == "fall":
                result = self._apply_fall_hold(session_id, result)
            kind_results.append((kind, result))
        _inference_timing.record(timings_ms)

        payload = merge_kind_results(kind_results)
        if payload.get("analyzedAt") is None:
            payload["analyzedAt"] = datetime.utcnow().isoformat()
        session_analysis_store.set_result(session_id, payload)
        return payload

    def _apply_fall_hold(self, session_id: str, result: dict[str, Any]) -> dict[str, Any]:
        """낙상은 한 장으로 판정하지 않는다 — 세션별 유지 조건(FallHoldTracker)으로 danger 를 다시 정한다."""
        if result.get("detectedType") == "unknown":
            return result  # 판정 불가 장면은 유지 이력에 넣지 않는다(danger=False 그대로)
        decision = fall_hold_tracker.update(
            session_id,
            float(result.get("confidence") or 0.0),
            _analysis_clock(),
            threshold=self.settings.fall_danger_threshold,
            hold_sec=self.settings.fall_hold_sec,
            hold_ratio=self.settings.fall_hold_ratio,
        )
        held = dict(result)
        held["danger"] = decision.danger
        held["confidence"] = round(decision.confidence, 4)
        if decision.danger:
            # 유지 구간 안의 한 장이 놓쳐 이번 장면이 normal 이어도 낙상 판정은 낙상으로 올린다.
            held["detectedType"] = "fall"
        return held

    def latest_analysis_for_session(self, session_id: str, camera_identifier: str) -> dict[str, Any] | None:
        cached = session_analysis_store.get_result(session_id)
        if cached is not None:
            return cached
        return self.latest_analysis(camera_identifier)

    def stop(self, session: StreamSessionState) -> StreamSessionState:
        session_analysis_store.clear_session(session.session_id)
        bbox_overlay.clear_session(session.session_id)
        clip_frame_buffer.clear(session.session_id)
        fall_hold_tracker.clear(session.session_id)
        if self.use_memory_state:
            return self._state_store.stop(session.session_id)

        row = self.db.query(StreamSession).filter(StreamSession.session_id == session.session_id).first()
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_response("스트림 세션을 찾을 수 없습니다.", "STREAM_SESSION_NOT_FOUND", None),
            )
        row.status = "stopped"
        row.stopped_at = datetime.utcnow()
        row.fps = 0.0
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        session.status = "stopped"
        session.stopped_at = row.stopped_at
        session.fps = 0.0
        return session

    def refresh_disconnect_status(self, session: StreamSessionState) -> StreamSessionState:
        timeout_sec = self.settings.live_stream_disconnect_timeout_sec
        if session.status == "running" and session.last_frame_at is not None:
            if datetime.utcnow() - session.last_frame_at > timedelta(seconds=timeout_sec):
                if self.use_memory_state:
                    return self._state_store.mark_disconnected(session.session_id)
                session.status = "disconnected"
                session.fps = 0.0
                row = self.db.query(StreamSession).filter(StreamSession.session_id == session.session_id).first()
                if row:
                    row.status = "disconnected"
                    row.fps = 0.0
                    self.db.add(row)
                self.db.commit()
        return session

    def latest_analysis(self, camera_identifier: str) -> dict[str, Any] | None:
        result = (
            self.db.query(AnalysisResult)
            .filter(AnalysisResult.camera_identifier == camera_identifier)
            .order_by(AnalysisResult.id.desc())
            .first()
        )
        if not result:
            return None
        return {
            "detectedType": result.detected_type,
            "confidence": result.confidence,
            "danger": result.danger,
        }

    def get_status_payload(self, session_id: str) -> dict[str, Any]:
        session = self.require_session(session_id)
        return {
            "sessionId": session.session_id,
            "status": session.status,
            "lastFrameAt": session.last_frame_at.isoformat() if session.last_frame_at else None,
            "fps": self.frame_store.get_fps(session_id),
            "viewerCount": self.frame_store.get_viewer_count(session_id),
            "isAnalyzing": bool(session.is_analyzing),
        }

    def list_live(self) -> list[dict[str, Any]]:
        if self.use_memory_state:
            sessions = self._state_store.list_all()
        else:
            sessions = [
                StreamSessionState(
                    session_id=s.session_id,
                    camera_identifier=s.camera_identifier,
                    device_type=s.device_type,
                    status=s.status,
                    started_at=s.started_at,
                    last_frame_at=s.last_frame_at,
                    stopped_at=s.stopped_at,
                    fps=s.fps,
                    viewer_count=s.viewer_count,
                    is_analyzing=bool(s.is_analyzing),
                )
                for s in self.db.query(StreamSession).order_by(StreamSession.started_at.desc()).all()
            ]
        items: list[dict[str, Any]] = []
        for session in sessions:
            refreshed = self.refresh_disconnect_status(session)
            if refreshed.status not in {"running", "disconnected"}:
                continue
            viewer_count = self.frame_store.get_viewer_count(refreshed.session_id)
            refreshed.viewer_count = viewer_count
            items.append(
                {
                    "sessionId": refreshed.session_id,
                    "cameraIdentifier": refreshed.camera_identifier,
                    "deviceType": refreshed.device_type,
                    "status": refreshed.status,
                    "lastFrameAt": refreshed.last_frame_at.isoformat() if refreshed.last_frame_at else None,
                    "viewerUrl": self._viewer_url(refreshed.session_id),
                    "hlsUrl": self._hls_url(refreshed.session_id),
                    "ingestUrl": self._ingest_url(refreshed.session_id),
                    "latestAnalysis": self.latest_analysis_for_session(
                        refreshed.session_id,
                        refreshed.camera_identifier,
                    ),
                },
            )
        return items

    def require_session(self, session_id: str) -> StreamSessionState:
        session = self.get_by_session_id(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_response("스트림 세션을 찾을 수 없습니다.", "STREAM_SESSION_NOT_FOUND", None),
            )
        return self.refresh_disconnect_status(session)


frame_store = StreamFrameStore()


class _InFlightSessions:
    """추론이 진행 중인 세션 집합."""

    def __init__(self) -> None:
        self._sessions: set[str] = set()
        self._lock = Lock()

    def try_enter(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                return False
            self._sessions.add(session_id)
            return True

    def leave(self, session_id: str) -> None:
        with self._lock:
            self._sessions.discard(session_id)

    def contains(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions


class _InferenceTimingLog:
    """프레임당 추론 시간 요약 — STREAM_INFER_LOG_INTERVAL_SEC 마다 INFO 한 줄(감지기 여러 개를 켤 때 속도 확인용)."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._reset(time.monotonic())

    def _reset(self, now: float) -> None:
        self._window_start = now
        self._frames = 0
        self._total_sum = 0.0
        self._total_max = 0.0
        self._kind_sums: dict[str, float] = {}

    def record(self, timings_ms: dict[str, float]) -> None:
        if not timings_ms:
            return
        total = sum(timings_ms.values())
        _LOG.debug("[LIVE-INFER] total=%.1fms %s", total, timings_ms)
        interval = get_settings().stream_infer_log_interval_sec
        if interval <= 0:
            return
        now = time.monotonic()
        with self._lock:
            self._frames += 1
            self._total_sum += total
            self._total_max = max(self._total_max, total)
            for kind, ms in timings_ms.items():
                self._kind_sums[kind] = self._kind_sums.get(kind, 0.0) + ms
            if now - self._window_start < interval:
                return
            frames = self._frames
            avg = self._total_sum / frames
            peak = self._total_max
            per_kind = ", ".join(f"{k} {v / frames:.1f}ms" for k, v in self._kind_sums.items())
            elapsed = now - self._window_start
            self._reset(now)
        _LOG.info(
            "[LIVE-INFER] %.0f초 %d프레임 프레임당 평균 %.1fms (최대 %.1fms) - %s",
            elapsed,
            frames,
            avg,
            peak,
            per_kind,
        )


_inference_timing = _InferenceTimingLog()
_analysis_in_flight = _InFlightSessions()
_analysis_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stream-analyze")


@dataclass
class StreamSessionState:
    session_id: str
    camera_identifier: str
    device_type: str
    status: str
    started_at: datetime
    last_frame_at: datetime | None
    stopped_at: datetime | None
    fps: float
    viewer_count: int
    is_analyzing: bool


class StreamSessionStateStore:
    def __init__(self) -> None:
        self._sessions: dict[str, StreamSessionState] = {}
        self._lock = Lock()

    def create_or_restart(self, session_id: str, camera_identifier: str, device_type: str) -> StreamSessionState:
        now = datetime.utcnow()
        with self._lock:
            state = StreamSessionState(
                session_id=session_id,
                camera_identifier=camera_identifier,
                device_type=device_type,
                status="running",
                started_at=now,
                last_frame_at=None,
                stopped_at=None,
                fps=0.0,
                viewer_count=0,
                is_analyzing=True,
            )
            self._sessions[session_id] = state
            return state

    def get(self, session_id: str) -> StreamSessionState | None:
        with self._lock:
            return self._sessions.get(session_id)

    def ingest_frame(self, session_id: str, frame_time: datetime, fps: float) -> StreamSessionState:
        with self._lock:
            state = self._sessions[session_id]
            state.last_frame_at = frame_time
            state.fps = fps
            state.status = "running"
            state.stopped_at = None
            return state

    def stop(self, session_id: str) -> StreamSessionState:
        with self._lock:
            state = self._sessions[session_id]
            state.status = "stopped"
            state.stopped_at = datetime.utcnow()
            state.fps = 0.0
            return state

    def mark_disconnected(self, session_id: str) -> StreamSessionState:
        with self._lock:
            state = self._sessions[session_id]
            state.status = "disconnected"
            state.fps = 0.0
            return state

    def list_all(self) -> list[StreamSessionState]:
        with self._lock:
            return sorted(self._sessions.values(), key=lambda s: s.started_at, reverse=True)


stream_session_state_store = StreamSessionStateStore()
