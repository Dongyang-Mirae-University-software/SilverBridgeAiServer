from __future__ import annotations

import asyncio
import logging
import math
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.core.config import Settings, get_settings
from app.services.session_analysis_store import session_analysis_store

_LOG = logging.getLogger(__name__)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

# 감지 박스는 "보여주는 순간"에만 복사본에 그린다. frame_store·분석 입력·클립 링버퍼의 원본 JPEG 는 건드리지 않는다.
#
# 박스 표시 기준 = 종류별 위험(danger) 기준(LIVE_BBOX_MIN_CONFIDENCE=danger, 기본)이다. 추론 후보 기준
# (*_CONF_THRESHOLD)보다 높고, 박스가 보이면 알림 기준을 넘은 것이다 - 단 낙상은 한 장 점수만 보며
# 알림은 FallHoldTracker 의 유지 조건까지 통과해야 나간다. 박스는 표시 필터일 뿐 판정·알림에 관여하지 않는다.

_BLUE_BGR = (255, 0, 0)
_TEXT_BGR = (255, 255, 255)
_JPEG_QUALITY = 85
# detections[].detectedType → 위험 기준을 고르는 종류. 이 밖의 값은 계약에 없는 것이라 그리지 않는다.
_KIND_BY_TYPE = {"fire": "fire", "smoke": "fire", "knife": "knife", "fall": "fall"}
_DANGER_MODE = "danger"


@dataclass(frozen=True)
class Box:
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


def _utcnow() -> datetime:
    # 분석 시각(analyzedAt)이 naive UTC 라 같은 기준으로 맞춘다. 테스트가 바꿔 끼운다.
    return datetime.now(timezone.utc).replace(tzinfo=None)


_warned_min_conf: set[str] = set()


def _override_threshold(settings: Settings) -> float | None:
    """LIVE_BBOX_MIN_CONFIDENCE 가 숫자면 모든 종류에 쓸 재정의 값, 'danger'(기본)·잘못된 값이면 None."""
    raw = (settings.live_bbox_min_confidence or "").strip().lower()
    if raw in ("", _DANGER_MODE):
        return None
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    if not 0.0 <= value <= 1.0:
        if raw not in _warned_min_conf:
            _warned_min_conf.add(raw)
            _LOG.warning("[BBOX-CONFIG] LIVE_BBOX_MIN_CONFIDENCE 값이 올바르지 않아 danger 기준을 쓴다")
        return None
    return value


def _danger_threshold(kind: str, settings: Settings) -> float:
    # 값을 복사해 두지 않고 그때그때 settings 에서 읽는다 - .env 로 위험 기준을 바꾸면 박스 경계도 따라간다.
    if kind == "fire":
        return settings.fire_smoke_danger_threshold
    if kind == "knife":
        return settings.knife_danger_threshold
    return settings.fall_danger_threshold


def select_boxes(detections: Any, settings: Settings) -> tuple[Box, ...]:
    """표시 기준을 넘는 감지만 박스로 고른다. 형식이 틀린 항목은 건너뛴다."""
    if not isinstance(detections, (list, tuple)):
        return ()
    override = _override_threshold(settings)
    boxes: list[Box] = []
    for item in detections:
        if not isinstance(item, dict):
            continue
        kind = _KIND_BY_TYPE.get(item.get("detectedType"))
        bbox = item.get("bbox")
        if kind is None or not isinstance(bbox, dict):
            continue
        try:
            score = float(item.get("confidence"))
            coords = [float(bbox[k]) for k in ("x1", "y1", "x2", "y2")]
        except (TypeError, ValueError, KeyError):
            continue
        if not math.isfinite(score) or not all(math.isfinite(c) for c in coords):
            continue
        threshold = override if override is not None else _danger_threshold(kind, settings)
        if score < threshold:
            continue
        x1, y1, x2, y2 = (int(round(c)) for c in coords)
        if x1 >= x2 or y1 >= y2:
            continue
        boxes.append(Box(item["detectedType"], score, x1, y1, x2, y2))
    return tuple(boxes)


def active_boxes(session_id: str, settings: Settings) -> tuple[Box, ...]:
    """지금 화면에 그릴 박스. 마지막 분석 결과를 LIVE_BBOX_HOLD_SECONDS 동안 유지한다.

    분석은 N프레임마다 한 번이라 그 사이 프레임에도 직전 결과를 보여 준다. 감지 0건 결과가 오면 결과가
    바뀌므로 박스는 즉시 사라지고, 새 결과 없이 유지 시간이 지나도 사라진다.
    """
    result = session_analysis_store.get_result(session_id)
    if not result:
        return ()
    detections = result.get("detections")
    if not detections:
        return ()
    analyzed_at = result.get("analyzedAt")
    try:
        at = datetime.fromisoformat(str(analyzed_at))
    except ValueError:
        return ()  # 시각을 모르면 낡은 결과인지 판단할 수 없다 - 그리지 않는다
    if at.tzinfo is not None:
        at = at.astimezone(timezone.utc).replace(tzinfo=None)
    age = (_utcnow() - at).total_seconds()
    if age > settings.live_bbox_hold_seconds:
        return ()
    return select_boxes(detections, settings)


def draw_boxes(frame_bytes: bytes, boxes: tuple[Box, ...]) -> bytes:
    """JPEG 를 복사해 파란 박스와 '클래스 0.00' 라벨을 그려 다시 인코딩한다. 그릴 수 없으면 원본 그대로."""
    if cv2 is None or not boxes:
        return frame_bytes
    frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        return frame_bytes
    h, w = frame.shape[:2]
    thickness = min(3, max(2, round(min(h, w) / 300)))
    font_scale = max(0.5, min(h, w) / 800)
    text_thickness = 1 if font_scale < 1.0 else 2
    drawn = 0
    for box in boxes:
        x1, y1 = max(0, min(box.x1, w - 1)), max(0, min(box.y1, h - 1))
        x2, y2 = max(0, min(box.x2, w - 1)), max(0, min(box.y2, h - 1))
        if x1 >= x2 or y1 >= y2:
            continue
        cv2.rectangle(frame, (x1, y1), (x2, y2), _BLUE_BGR, thickness)
        label = f"{box.label} {box.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        lh = th + baseline + 2
        lx = max(0, min(x1, w - tw - 4))
        # 위쪽 가장자리에 붙은 박스는 라벨을 박스 안쪽 위에 둔다(밖에 두면 프레임 밖으로 잘린다).
        ly_top = y1 - lh if y1 - lh >= 0 else y1
        cv2.rectangle(frame, (lx, ly_top), (lx + tw + 4, ly_top + lh), _BLUE_BGR, -1)
        cv2.putText(
            frame, label, (lx + 2, ly_top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, _TEXT_BGR,
            text_thickness, cv2.LINE_AA,
        )
        drawn += 1
    if not drawn:
        return frame_bytes
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
    return buf.tobytes() if ok else frame_bytes


class BboxOverlay:
    """송출·최신 프레임 응답용 박스 그리기. 같은 프레임을 여러 시청자가 봐도 그리기는 한 번만 한다."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[Any, bytes]] = {}  # session → (프레임 순번+박스, 그린 JPEG) 마지막 1개
        self._inflight: dict[tuple[str, Any], Future[bytes]] = {}
        self._timeouts: dict[str, int] = {}
        # 전용 풀 - 이벤트 루프·공용 스레드풀을 그리기가 막지 않게 한다. 2개면 FHD 몇 세션은 충분하다.
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bbox-draw")

    def _draw_safely(self, session_id: str, key: Any, frame_bytes: bytes, boxes: tuple[Box, ...]) -> bytes:
        try:
            out = draw_boxes(frame_bytes, boxes)
        except Exception as exc:  # noqa: BLE001 - 어떤 오류도 영상을 끊지 않는다
            _LOG.warning("[BBOX-DRAW-FAILED] sessionId=%s error=%s", session_id, type(exc).__name__)
            out = frame_bytes
        with self._lock:
            self._cache[session_id] = (key, out)  # 실패해도 원본을 캐시해 같은 프레임에서 되풀이하지 않는다
            self._inflight.pop((session_id, key), None)
        return out

    def _prepare(self, session_id: str, frame_bytes: bytes, seq: int) -> bytes | Future[bytes]:
        settings = get_settings()
        if not settings.live_draw_bbox:
            return frame_bytes
        boxes = active_boxes(session_id, settings)
        if not boxes:
            return frame_bytes  # 감지가 없으면 디코딩·재인코딩 없이 원본 그대로 - 평소 추가 비용 0
        key = (seq, boxes)
        with self._lock:
            cached = self._cache.get(session_id)
            if cached is not None and cached[0] == key:
                return cached[1]
            pending = self._inflight.get((session_id, key))
            if pending is not None:
                return pending
            try:
                future = self._executor.submit(self._draw_safely, session_id, key, frame_bytes, boxes)
            except RuntimeError:  # 풀이 닫힌 종료 중
                return frame_bytes
            self._inflight[(session_id, key)] = future
            return future

    def _on_timeout(self, session_id: str) -> None:
        with self._lock:
            count = self._timeouts.get(session_id, 0) + 1
            self._timeouts[session_id] = count
        if count == 1 or count % 25 == 0:
            _LOG.warning("[BBOX-DRAW-TIMEOUT] sessionId=%s count=%d", session_id, count)

    def render(self, session_id: str, frame_bytes: bytes, seq: int) -> bytes:
        """동기판(latest-frame). 상한을 넘으면 박스 없이 원본을 돌려준다."""
        prepared = self._prepare(session_id, frame_bytes, seq)
        if isinstance(prepared, bytes):
            return prepared
        try:
            return prepared.result(timeout=get_settings().live_bbox_draw_timeout_ms / 1000)
        except FutureTimeout:
            self._on_timeout(session_id)
            return frame_bytes

    async def render_async(self, session_id: str, frame_bytes: bytes, seq: int) -> bytes:
        """MJPEG 용. 그리기는 전용 스레드에서 하고, 상한을 넘으면 박스 없이 원본을 내보낸다(영상 지연 < 박스 누락)."""
        prepared = self._prepare(session_id, frame_bytes, seq)
        if isinstance(prepared, bytes):
            return prepared
        try:
            # shield: 시간 초과로 이 시청자만 포기해도 다른 시청자가 같이 기다리는 그리기는 취소하지 않는다.
            return await asyncio.wait_for(
                asyncio.shield(asyncio.wrap_future(prepared)),
                timeout=get_settings().live_bbox_draw_timeout_ms / 1000,
            )
        except asyncio.TimeoutError:
            self._on_timeout(session_id)
            return frame_bytes

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._cache.pop(session_id, None)
            self._timeouts.pop(session_id, None)


bbox_overlay = BboxOverlay()
