from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock

_LOG = logging.getLogger(__name__)

_PENDING_LOG_INTERVAL_SEC = 60.0
_MAX_SCENES = 300  # 분석이 아주 빠를 때 메모리 상한(2초 창 x 150회/초)


@dataclass(frozen=True)
class FireHoldDecision:
    danger: bool
    # danger 면 유지 구간 양성 장면의 마지막 종류·평균 점수, 아니면 None(호출자가 이번 장면 값을 그대로 쓴다).
    detected_type: str | None = None
    confidence: float | None = None
    positives: int = 0
    scenes: int = 0


@dataclass
class _SessionHistory:
    # (분석 시각, 양성 여부, 점수, 종류)
    scenes: deque[tuple[float, bool, float, str]] = field(default_factory=deque)
    observed_since: float | None = None
    last_pending_log: float | None = None


class FireHoldTracker:
    """세션별 화재·연기 유지 판정.

    한 장짜리 양성(흔들린 화면·철판 같은 순간적 오탐)으로는 알리지 않는다. 최근 window_sec 초 동안의 분석 중
    ratio 이상이 양성(한 장 기준 danger)이고, 그 구간에 분석이 min_scenes 회 이상 있으며, 연속 관찰이
    window_sec 이상 이어졌을 때만 danger 다.

    - 시간 기준이다. 서버 분석 속도는 정상 상태에서 초당 약 10회라 횟수 기준(4회 중 3회)은 0.3초에 불과했다
      (2026-10-13 실물 검증: 단발 화재 오탐이 0.13초 만에 알림이 됨).
    - 분석이 느려져(카메라 증가·부하) 창 안에 min_scenes 회가 안 들어오면 최근 min_scenes 회를 대신 본다 -
      시간 창만 쓰면 간격이 창보다 길 때 화재 알림이 영영 나지 않는다.
    - 마지막 분석과의 간격이 max_gap_sec 보다 길면 연속 관찰이 끊긴 것으로 보고 다시 센다.
    - 품질 게이트가 막은 장면은 음성으로 센다.
    """

    def __init__(self, clock=time.monotonic, label: str = "FIRE") -> None:
        self._label = label
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
        window_sec: float,
        ratio: float,
        min_scenes: int,
        max_gap_sec: float,
    ) -> FireHoldDecision:
        min_scenes = max(1, min_scenes)
        with self._lock:
            history = self._sessions.setdefault(session_id, _SessionHistory())
            scenes = history.scenes
            if scenes and now - scenes[-1][0] > max_gap_sec:
                scenes.clear()
                history.observed_since = None
            if history.observed_since is None:
                history.observed_since = now
            scenes.append((now, positive, score, detected_type))
            # 창 밖 장면은 버리되, 최근 min_scenes 회는 느린 분석에서도 남긴다.
            window_start = now - window_sec
            while len(scenes) > min_scenes and scenes[0][0] < window_start:
                scenes.popleft()
            while len(scenes) > _MAX_SCENES:
                scenes.popleft()

            total = len(scenes)
            positives = [s for s in scenes if s[1]]
            observed = now - history.observed_since
            if total >= min_scenes and observed >= window_sec and len(positives) / total >= ratio:
                avg = sum(s[2] for s in positives) / len(positives)
                return FireHoldDecision(True, positives[-1][3], avg, len(positives), total)
            if positive and self._pending_log_due(history):
                _LOG.info(
                    "[%s-HOLD-PENDING] sessionId=%s 양성 %d/%d회 관찰 %.1f/%.1f초 - 유지 조건 전이라 알리지 않는다",
                    self._label,
                    session_id,
                    len(positives),
                    total,
                    observed,
                    window_sec,
                )
            return FireHoldDecision(False, None, None, len(positives), total)

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

# 흉기도 같은 구조(시간 기준 창·최소 장면 수·끊김 리셋)다. 세션 이력은 종류별로 분리된다.
knife_hold_tracker = FireHoldTracker(label="KNIFE")
