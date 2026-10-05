from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime, timedelta
from threading import Lock

_LOG = logging.getLogger(__name__)


class ClipFrameBuffer:
    """세션별 최근 프레임 링버퍼 — 클립 생성용 (in-memory).

    JPEG 바이트를 디코딩하지 않고 그대로 보관한다(FHD JPEG 는 보통 장당 0.2~0.5MB지만 수신 크기 제한은 없다).
    시각은 StreamFrameStore.set_frame 의 수신 시각(naive UTC)과 같은 값을 쓴다 —
    분석 결과의 analyzedAt 도 같은 시계라 감지 시각으로 구간을 바로 자를 수 있다.

    메모리 상한은 시간(buffer_seconds)·장수(max_frames)·바이트(세션별·전체) 세 겹이다.
    바이트 상한을 넘으면 오래된 프레임부터 지운다 — 새 프레임을 거부하면 감지 시점 프레임이 빠진다.
    전체 상한은 가장 많이 쓰는 세션부터 줄여, 큰 세션 하나가 다른 세션의 클립을 비우지 않게 한다.
    0 이하의 바이트 상한은 그 상한을 끈다.
    """

    _SWEEP_INTERVAL_SEC = 5.0
    _WARN_INTERVAL_SEC = 30.0

    def __init__(
        self,
        buffer_seconds: float,
        max_frames: int,
        session_max_bytes: int = 0,
        total_max_bytes: int = 0,
    ) -> None:
        self._buffer_seconds = buffer_seconds
        self._max_frames = max(1, max_frames)
        self._session_max_bytes = max(0, session_max_bytes)
        self._total_max_bytes = max(0, total_max_bytes)
        caps = [cap for cap in (self._session_max_bytes, self._total_max_bytes) if cap]
        self._frame_max_bytes = min(caps) if caps else 0
        self._frames: dict[str, deque[tuple[datetime, bytes]]] = {}
        self._session_bytes: dict[str, int] = {}
        self._total_bytes = 0
        self._lock = Lock()
        self._last_sweep = 0.0
        self._last_warn: dict[tuple[str, str], float] = {}

    def add(self, session_id: str, frame_time: datetime, frame_bytes: bytes) -> None:
        size = len(frame_bytes)
        with self._lock:
            if self._frame_max_bytes and size > self._frame_max_bytes:
                # 한 장이 상한보다 크면 버퍼에만 넣지 않는다(분석·MJPEG 송출은 호출부에서 그대로 진행된다).
                self._warn_locked(
                    "too_large", session_id,
                    "clip buffer frame skipped session=%r bytes=%d limit=%d",
                    session_id, size, self._frame_max_bytes,
                )
                self._sweep_locked(frame_time)
                return
            frames = self._frames.get(session_id)
            if frames is None:
                frames = deque()
                self._frames[session_id] = frames
                self._session_bytes[session_id] = 0
            frames.append((frame_time, frame_bytes))
            self._session_bytes[session_id] += size
            self._total_bytes += size

            cutoff = frame_time - timedelta(seconds=self._buffer_seconds)
            while frames and frames[0][0] < cutoff:
                self._pop_oldest_locked(session_id)
            while len(frames) > self._max_frames:
                self._pop_oldest_locked(session_id)
            self._enforce_session_cap_locked(session_id)
            self._enforce_total_cap_locked(session_id)
            self._sweep_locked(frame_time)

    def _pop_oldest_locked(self, session_id: str) -> None:
        frames = self._frames.get(session_id)
        if not frames:
            return
        _, data = frames.popleft()
        self._session_bytes[session_id] -= len(data)
        self._total_bytes -= len(data)
        if not frames:
            self._drop_session_locked(session_id)

    def _drop_session_locked(self, session_id: str) -> None:
        self._frames.pop(session_id, None)
        self._total_bytes -= self._session_bytes.pop(session_id, 0)
        for kind in ("too_large", "session_cap", "total_cap"):
            self._last_warn.pop((kind, session_id), None)

    def _enforce_session_cap_locked(self, session_id: str) -> None:
        if not self._session_max_bytes or self._session_bytes.get(session_id, 0) <= self._session_max_bytes:
            return
        while self._session_bytes.get(session_id, 0) > self._session_max_bytes:
            self._pop_oldest_locked(session_id)
        self._warn_locked(
            "session_cap", session_id,
            "clip buffer session cap reached session=%r bytes=%d limit=%d",
            session_id, self._session_bytes.get(session_id, 0), self._session_max_bytes,
        )

    def _enforce_total_cap_locked(self, current_session_id: str) -> None:
        while self._total_max_bytes and self._total_bytes > self._total_max_bytes and self._frames:
            # 가장 많이 쓰는 세션의 가장 오래된 프레임부터. 방금 넣은 한 장뿐인 세션은 후순위다.
            victim = max(
                self._frames,
                key=lambda sid: (
                    not (sid == current_session_id and len(self._frames[sid]) == 1),
                    self._session_bytes.get(sid, 0),
                ),
            )
            self._pop_oldest_locked(victim)
            self._warn_locked(
                "total_cap", victim,
                "clip buffer total cap reached session=%r total_bytes=%d limit=%d",
                victim, self._total_bytes, self._total_max_bytes,
            )

    def _warn_locked(self, kind: str, session_id: str, msg: str, *args: object) -> None:
        # 세션 ID 와 바이트 수만 남긴다(프레임 내용 금지). 같은 세션·사유는 30초에 한 번.
        mono = time.monotonic()
        key = (kind, session_id)
        last = self._last_warn.get(key)
        if last is not None and mono - last < self._WARN_INTERVAL_SEC:
            return
        self._last_warn[key] = mono
        _LOG.warning(msg, *args)

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
            self._drop_session_locked(sid)
        # 버퍼에 한 장도 못 들어간 세션(전부 상한 초과)의 로그 제한 기록도 오래되면 지운다.
        expired = [key for key, at in self._last_warn.items()
                   if key[1] not in self._frames and mono - at >= self._WARN_INTERVAL_SEC]
        for key in expired:
            self._last_warn.pop(key, None)

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
            self._drop_session_locked(session_id)

    def session_count(self) -> int:
        with self._lock:
            return len(self._frames)

    def session_bytes(self, session_id: str) -> int:
        with self._lock:
            return self._session_bytes.get(session_id, 0)

    def total_bytes(self) -> int:
        with self._lock:
            return self._total_bytes


def _build_default() -> ClipFrameBuffer:
    from app.core.config import get_settings

    settings = get_settings()
    return ClipFrameBuffer(
        settings.clip_buffer_seconds,
        settings.clip_max_frames,
        settings.clip_session_max_bytes,
        settings.clip_total_max_bytes,
    )


clip_frame_buffer = _build_default()
