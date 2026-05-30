from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Lock
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.response import error_response
from app.services.fire_smoke_detection_service import get_fire_smoke_detector
from app.services.session_analysis_store import session_analysis_store
from app.models.analysis_result import AnalysisResult
from app.models.stream_session import StreamSession


class StreamFrameStore:
    def __init__(self) -> None:
        self._frames: dict[str, bytes] = {}
        self._frame_times: dict[str, datetime] = {}
        self._last_epoch: dict[str, float] = {}
        self._fps_map: dict[str, float] = {}
        self._viewers: dict[str, int] = {}
        self._lock = Lock()

    def set_frame(self, session_id: str, frame_bytes: bytes) -> tuple[datetime, float]:
        now = datetime.utcnow()
        now_epoch = time.time()
        with self._lock:
            self._frames[session_id] = frame_bytes
            self._frame_times[session_id] = now
            prev = self._last_epoch.get(session_id)
            if prev is None or now_epoch <= prev:
                fps = self._fps_map.get(session_id, 0.0)
            else:
                fps = round(1.0 / (now_epoch - prev), 2)
            self._last_epoch[session_id] = now_epoch
            self._fps_map[session_id] = fps
            self._viewers.setdefault(session_id, 0)
            return now, fps

    def get_frame(self, session_id: str) -> bytes | None:
        with self._lock:
            return self._frames.get(session_id)

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
        if not self.settings.fire_smoke_enabled:
            return session_analysis_store.get_result(session_id)
        if not session_analysis_store.should_analyze(session_id, self.settings.stream_sample_every_n_frames):
            return session_analysis_store.get_result(session_id)

        detector = get_fire_smoke_detector()
        result = detector.detect_from_jpeg(frame_bytes)
        payload = {
            "detectedType": result.get("detectedType", "normal"),
            "confidence": result.get("confidence", 0.0),
            "danger": False,
            "detections": result.get("detections") or [],
            "analyzedAt": result.get("detectedAt"),
        }
        session_analysis_store.set_result(session_id, payload)
        return payload

    def latest_analysis_for_session(self, session_id: str, camera_identifier: str) -> dict[str, Any] | None:
        cached = session_analysis_store.get_result(session_id)
        if cached is not None:
            return cached
        return self.latest_analysis(camera_identifier)

    def stop(self, session: StreamSessionState) -> StreamSessionState:
        session_analysis_store.clear_session(session.session_id)
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
