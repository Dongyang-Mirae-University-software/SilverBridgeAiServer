from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock

_LOG = logging.getLogger(__name__)

_PENDING_LOG_INTERVAL_SEC = 60.0


@dataclass(frozen=True)
class FireHoldDecision:
    danger: bool
    # danger 면 최근 양성 장면의 종류·평균 점수, 아니면 None(호출자가 이번 장면 값을 그대로 쓴다).
    detected_type: str | None = None
    confidence: float | None = None
    positives: int = 0
    scenes: int = 0


@dataclass
class _SessionHistory:
    # (분석 시각, 양성 여부, 점수, 종류)
    scenes: deque[tuple[float, bool, float, str]] = field(default_factory=deque)
    last_pending_log: float | None = None


class FireHoldTracker:
    """세션별 화재·연기 유지 판정.

    한 장짜리 양성(흔들린 화면·렌즈 가림 등 순간적인 오탐)으로는 알리지 않는다. 최근 window 번의 분석 중
    min_positive 번 이상이 양성(한 장 기준 danger)일 때만 danger 다.

    - 시간 창이 아니라 분석 횟수로 센다. 분석은 카메라 수만큼 직렬이라 한 카메라의 간격이 부하에 따라 달라지는데,
      시간 창이면 간격이 창보다 길어질 때 화재 알림이 영영 나지 않는다(낙상 트래커와 다른 이유).
    - 마지막 분석과의 간격이 max_gap_sec 보다 길면 연속 관찰이 끊긴 것으로 보고 다시 센다(옛 장면과 섞지 않는다).
    - 품질 게이트가 막은 장면은 음성으로 센다.
    """

    def __init__(self, clock=time.monotonic) -> None:
        self._lock = Lock()
        self._sessions: dict[str, _SessionHistory] = {}
        self._clock = clock

    def update(
        self,
        session_id: str,
        positive: bool,
        score: float,
        detected_type: str,
        now: float,
        *,
        window: int,
        min_positive: int,
        max_gap_sec: float,
    ) -> FireHoldDecision:
        min_positive = max(1, min(min_positive, window))
        with self._lock:
            history = self._sessions.setdefault(session_id, _SessionHistory())
            scenes = history.scenes
            if scenes and now - scenes[-1][0] > max_gap_sec:
                scenes.clear()
            scenes.append((now, positive, score, detected_type))
            while len(scenes) > window:
                scenes.popleft()

            positives = [s for s in scenes if s[1]]
            if len(positives) >= min_positive:
                avg = sum(s[2] for s in positives) / len(positives)
                return FireHoldDecision(True, positives[-1][3], avg, len(positives), len(scenes))
            if positive and self._pending_log_due(history):
                _LOG.info(
                    "[FIRE-HOLD-PENDING] sessionId=%s 양성 %d/%d회 - 유지 조건(%d회) 전이라 알리지 않는다",
                    session_id,
                    len(positives),
                    len(scenes),
                    min_positive,
                )
            return FireHoldDecision(False, None, None, len(positives), len(scenes))

    def _pending_log_due(self, history: _SessionHistory) -> bool:
        now = self._clock()
        last = history.last_pending_log
        if last is not None and now - last < _PENDING_LOG_INTERVAL_SEC:
            return False
        history.last_pending_log = now
        return True

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def tracked_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)


fire_hold_tracker = FireHoldTracker()
