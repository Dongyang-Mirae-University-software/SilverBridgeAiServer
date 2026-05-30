from __future__ import annotations

from threading import Lock
from typing import Any


class SessionAnalysisStore:
    """세션별 최신 fire/smoke 분석 결과 (in-memory)."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._results: dict[str, dict[str, Any]] = {}
        self._frame_counters: dict[str, int] = {}

    def increment_frame(self, session_id: str) -> int:
        with self._lock:
            count = self._frame_counters.get(session_id, 0) + 1
            self._frame_counters[session_id] = count
            return count

    def should_analyze(self, session_id: str, every_n_frames: int) -> bool:
        if every_n_frames <= 1:
            return True
        count = self.increment_frame(session_id)
        return count == 1 or count % every_n_frames == 0

    def set_result(self, session_id: str, result: dict[str, Any]) -> None:
        with self._lock:
            self._results[session_id] = result

    def get_result(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._results.get(session_id)
            return dict(item) if item else None

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._results.pop(session_id, None)
            self._frame_counters.pop(session_id, None)


session_analysis_store = SessionAnalysisStore()
