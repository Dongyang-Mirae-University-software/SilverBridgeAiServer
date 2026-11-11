from __future__ import annotations

from threading import Lock
from typing import Any


class SessionAnalysisStore:
    """세션별 최신 fire/smoke 분석 결과 (in-memory)."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._results: dict[str, dict[str, Any]] = {}
        self._frame_counters: dict[str, int] = {}
        # 세션 세대 번호 - stop·재시작 때 올린다. 분석을 시작할 때 잡은 세대와 저장 때 세대가 다르면 결과를 버린다(A-6).
        self._generations: dict[str, int] = {}

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

    def generation(self, session_id: str) -> int:
        with self._lock:
            return self._generations.get(session_id, 0)

    def is_current(self, session_id: str, generation: int) -> bool:
        return self.generation(session_id) == generation

    def set_result(self, session_id: str, result: dict[str, Any], generation: int | None = None) -> bool:
        """결과를 저장한다. generation 을 주면 그 사이 세션이 멈추거나 재시작됐을 때 저장하지 않고 False."""
        with self._lock:
            if generation is not None and self._generations.get(session_id, 0) != generation:
                return False
            self._results[session_id] = result
            return True

    def get_result(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._results.get(session_id)
            return dict(item) if item else None

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._results.pop(session_id, None)
            self._frame_counters.pop(session_id, None)
            self._generations[session_id] = self._generations.get(session_id, 0) + 1


session_analysis_store = SessionAnalysisStore()
