from __future__ import annotations

import time
from collections import deque
from datetime import datetime, timedelta
from threading import Lock


class ClipFrameBuffer:
    """세션별 최근 프레임 링버퍼 — 클립 생성용 (in-memory).

    JPEG 바이트를 디코딩하지 않고 그대로 보관한다(FHD 원본은 장당 약 6.5MB).
    시각은 StreamFrameStore.set_frame 의 수신 시각(naive UTC)과 같은 값을 쓴다 —
    분석 결과의 analyzedAt 도 같은 시계라 감지 시각으로 구간을 바로 자를 수 있다.
    """

    _SWEEP_INTERVAL_SEC = 5.0

    def __init__(self, buffer_seconds: float, max_frames: int) -> None:
        self._buffer_seconds = buffer_seconds
        self._max_frames = max(1, max_frames)
        self._frames: dict[str, deque[tuple[datetime, bytes]]] = {}
        self._lock = Lock()
        self._last_sweep = 0.0

    def add(self, session_id: str, frame_time: datetime, frame_bytes: bytes) -> None:
        with self._lock:
            frames = self._frames.get(session_id)
            if frames is None:
                frames = deque(maxlen=self._max_frames)
                self._frames[session_id] = frames
            frames.append((frame_time, frame_bytes))
            cutoff = frame_time - timedelta(seconds=self._buffer_seconds)
            while frames and frames[0][0] < cutoff:
                frames.popleft()
            self._sweep_locked(frame_time)

    def _sweep_locked(self, now: datetime) -> None:
        # stop 없이 사라진 세션(브라우저 종료 등)의 버퍼는 새 프레임이 오지 않아 시간 제거가 돌지 않는다.
        # 프레임이 들어올 때 가끔 전체를 훑어, 마지막 프레임이 보관 시간보다 오래된 세션을 지운다.
        mono = time.monotonic()
        if mono - self._last_sweep < self._SWEEP_INTERVAL_SEC:
            return
        self._last_sweep = mono
        cutoff = now - timedelta(seconds=self._buffer_seconds)
        stale = [sid for sid, frames in self._frames.items() if not frames or frames[-1][0] < cutoff]
        for sid in stale:
            self._frames.pop(sid, None)

    def snapshot(self, session_id: str, start: datetime, end: datetime) -> list[tuple[datetime, bytes]]:
        """[start, end] 구간 프레임 목록. 바이트는 복사하지 않고 참조만 넘긴다(bytes 는 불변)."""
        with self._lock:
            frames = self._frames.get(session_id)
            if not frames:
                return []
            return [item for item in frames if start <= item[0] <= end]

    def latest_time(self, session_id: str) -> datetime | None:
        with self._lock:
            frames = self._frames.get(session_id)
            return frames[-1][0] if frames else None

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._frames.pop(session_id, None)

    def session_count(self) -> int:
        with self._lock:
            return len(self._frames)


def _build_default() -> ClipFrameBuffer:
    from app.core.config import get_settings

    settings = get_settings()
    return ClipFrameBuffer(settings.clip_buffer_seconds, settings.clip_max_frames)


clip_frame_buffer = _build_default()
