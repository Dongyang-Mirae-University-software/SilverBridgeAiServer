from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # opencv 가 없는 환경(일부 단위 테스트)에서도 import 는 되어야 한다 - 그때는 평가하지 않는다.
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None  # type: ignore[assignment]

_LOG = logging.getLogger(__name__)

# 사유 코드(응답·로그에 그대로 나간다).
REASON_WHITE = "white"
REASON_BLACK = "black"
REASON_FLAT = "flat"
REASON_BLUR_FLAT = "blur_flat"

_TARGET_WIDTH = 640
_WHITE_LEVEL = 240
_BLACK_LEVEL = 25


@dataclass(frozen=True)
class QualityThresholds:
    white_ratio: float
    black_ratio: float
    std_min: float
    lap_max: float
    blur_std_max: float


@dataclass(frozen=True)
class QualityVerdict:
    ok: bool
    reason: str | None = None

    def as_payload(self) -> dict[str, Any]:
        return {"ok": self.ok, "reason": self.reason}


def thresholds_from(settings: Any) -> QualityThresholds:
    return QualityThresholds(
        white_ratio=settings.frame_quality_white_ratio,
        black_ratio=settings.frame_quality_black_ratio,
        std_min=settings.frame_quality_std_min,
        lap_max=settings.frame_quality_lap_max,
        blur_std_max=settings.frame_quality_blur_std_max,
    )


def assess_gray(gray: np.ndarray, t: QualityThresholds) -> QualityVerdict:
    """흑백 화면 1장의 품질 판정. 가로 640 으로 줄인 뒤 4지표(흰색·검은색 비율, 밝기 표준편차, Laplacian 분산)를 본다.

    규칙: white > W 또는 black > B 또는 std < S 또는 (lap < L 그리고 std < T) → 불량.
    흐린 화면(blur·defocus)은 진짜 연기와 구분되지 않아 막지 않는다 - lap 은 "단색에 가까운 흐림"일 때만 쓴다.
    """
    h, w = gray.shape[:2]
    if w > _TARGET_WIDTH:
        gray = cv2.resize(gray, (_TARGET_WIDTH, max(1, int(h * _TARGET_WIDTH / w))), interpolation=cv2.INTER_AREA)
    total = gray.shape[0] * gray.shape[1]
    # 임계 2회 + meanStdDev 는 numpy 의 비교·평균·표준편차보다 빠르다(프레임당 비용을 1ms 안팎으로 유지).
    white = cv2.countNonZero(cv2.threshold(gray, _WHITE_LEVEL - 1, 255, cv2.THRESH_BINARY)[1]) / total
    black = cv2.countNonZero(cv2.threshold(gray, _BLACK_LEVEL, 255, cv2.THRESH_BINARY_INV)[1]) / total
    std = float(cv2.meanStdDev(gray)[1][0][0])
    if white > t.white_ratio:
        return QualityVerdict(False, REASON_WHITE)
    if black > t.black_ratio:
        return QualityVerdict(False, REASON_BLACK)
    if std < t.std_min:
        return QualityVerdict(False, REASON_FLAT)
    if std < t.blur_std_max:
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap < t.lap_max:
            return QualityVerdict(False, REASON_BLUR_FLAT)
    return QualityVerdict(True)


def assess_jpeg(frame_bytes: bytes, t: QualityThresholds) -> QualityVerdict | None:
    """JPEG 바이트의 품질 판정. 디코딩할 수 없으면 None(판정 불가 - 게이트는 막지 않는다)."""
    if cv2 is None or not frame_bytes:
        return None
    gray = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None or gray.size == 0:
        return None
    return assess_gray(gray, t)


class _BlockLogLimiter:
    """같은 세션의 차단 로그는 1분에 한 번만 남긴다(프레임마다 쏟아지지 않게)."""

    def __init__(self, interval_sec: float = 60.0, clock=time.monotonic) -> None:
        self._interval = interval_sec
        self._clock = clock
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def log_blocked(self, session_id: str, reason: str) -> None:
        now = self._clock()
        with self._lock:
            last = self._last.get(session_id)
            if last is not None and now - last < self._interval:
                return
            self._last[session_id] = now
        _LOG.info("[FRAME-QUALITY-BLOCKED] sessionId=%s reason=%s", session_id, reason)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._last.pop(session_id, None)


block_log_limiter = _BlockLogLimiter()
