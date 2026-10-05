from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from threading import Lock

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class FallHoldDecision:
    danger: bool
    # danger 면 유지 구간에서 임계를 넘은 장면들의 평균 점수, 아니면 이번 장면 점수.
    confidence: float


@dataclass
class _SessionHistory:
    scenes: deque[tuple[float, float]] = field(default_factory=deque)  # (분석 시각, fallen 최고 점수)
    observed_since: float | None = None
    gap_warned: bool = False


class FallHoldTracker:
    """세션별 낙상 유지 판정.

    낙상은 한 장으로 알리지 않는다 — 앉거나 몸을 숙인 순간이 fallen 으로 한 번 튀는 것을 거르기 위해,
    최근 hold_sec 초 동안 분석된 장면 중 hold_ratio 이상에서 점수가 threshold 이상일 때만 danger 다.

    - 시각은 분석 시각이다(샘플링 간격·추론 건너뛰기로 간격이 일정하지 않아 장면 수로 세지 않는다).
    - 관찰이 hold_sec 이상 이어져야 판정한다. 첫 장면 하나로 100% 가 되지 않게 하기 위함이다.
    - 직전 분석과의 간격이 hold_sec 보다 길면 연속 관찰이 끊긴 것으로 보고 다시 센다. 분석 간격이 늘
      hold_sec 보다 길면 낙상은 절대 danger 가 되지 않으므로 세션당 한 번 WARN 을 남긴다.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: dict[str, _SessionHistory] = {}

    def update(
        self,
        session_id: str,
        score: float,
        now: float,
        *,
        threshold: float,
        hold_sec: float,
        hold_ratio: float,
    ) -> FallHoldDecision:
        with self._lock:
            history = self._sessions.setdefault(session_id, _SessionHistory())
            scenes = history.scenes
            if scenes and now - scenes[-1][0] > hold_sec:
                if not history.gap_warned:
                    history.gap_warned = True
                    _LOG.warning(
                        "[FALL-HOLD-GAP] sessionId=%s 분석 간격 %.2fs > FALL_HOLD_SEC %.2fs - 간격이 계속 이보다 길면 "
                        "낙상 판정이 나지 않는다(STREAM_SAMPLE_EVERY_N_FRAMES·추론 시간 확인)",
                        session_id,
                        now - scenes[-1][0],
                        hold_sec,
                    )
                scenes.clear()
                history.observed_since = None
            if history.observed_since is None:
                history.observed_since = now
            scenes.append((now, score))
            window_start = now - hold_sec
            while scenes and scenes[0][0] < window_start:
                scenes.popleft()

            if now - history.observed_since < hold_sec or len(scenes) < 2:
                return FallHoldDecision(danger=False, confidence=score)
            positive = [s for _, s in scenes if s >= threshold]
            if len(positive) / len(scenes) >= hold_ratio:
                return FallHoldDecision(danger=True, confidence=sum(positive) / len(positive))
            return FallHoldDecision(danger=False, confidence=score)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def tracked_sessions(self) -> int:
        with self._lock:
            return len(self._sessions)


fall_hold_tracker = FallHoldTracker()
