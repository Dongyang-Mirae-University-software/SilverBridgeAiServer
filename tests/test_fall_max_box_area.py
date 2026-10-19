from __future__ import annotations

import logging
import types

import cv2
import numpy as np
import pytest

from app.core.config import get_settings
from app.services import yolo_class_detector as ycd
from app.services.fall_detection_service import FallDetectionService
from app.services.fall_hold_tracker import _IgnoredBoxLogLimiter, fall_hold_tracker
from app.services.knife_detection_service import KnifeDetectionService

W, H = 100, 100  # 면적 10000 - 상자 면적 비율을 계산하기 쉽게


class _Scalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class _Row:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


class _Boxes:
    def __init__(self, items: list[tuple[float, list[float]]]) -> None:
        self.cls = [_Scalar(0) for _ in items]
        self.conf = [_Scalar(score) for score, _ in items]
        self.xyxy = [_Row(xyxy) for _, xyxy in items]

    def __len__(self) -> int:
        return len(self.cls)


class _Result:
    def __init__(self, items: list[tuple[float, list[float]]]) -> None:
        self.boxes = _Boxes(items)


def _box(area_percent: int) -> list[float]:
    """100x100 화면에서 면적이 area_percent% 인 상자(가로 100, 세로 area_percent)."""
    return [0.0, 0.0, 100.0, float(area_percent)]


def _detector(max_area: float) -> FallDetectionService:
    settings = get_settings().model_copy(update={"fall_enabled": True, "fall_max_box_area": max_area})
    detector = FallDetectionService(settings)
    detector._class_names = {0: "fallen"}
    return detector


def _parse(detector: FallDetectionService, items: list[tuple[float, list[float]]]):  # noqa: ANN202
    return detector._parse_boxes(
        [_Result(items)], frame_w=W, frame_h=H, max_area_ratio=detector._config().max_box_area_ratio
    )


def test_only_big_box_gives_zero_score() -> None:
    detections, best_type, best_conf, ignored = _parse(_detector(0.35), [(0.3, _box(60))])
    assert (detections, best_type, best_conf) == ([], "normal", 0.0)
    assert ignored == [{"area": 0.6, "confidence": 0.3}]


def test_small_box_is_unchanged() -> None:
    detections, best_type, best_conf, ignored = _parse(_detector(0.35), [(0.5, _box(20))])
    assert (best_type, best_conf, ignored) == ("fall", 0.5, [])
    assert len(detections) == 1


def test_big_and_small_boxes_use_the_small_box_score() -> None:
    detections, best_type, best_conf, ignored = _parse(_detector(0.35), [(0.9, _box(70)), (0.25, _box(10))])
    assert (best_type, best_conf) == ("fall", 0.25)  # 점수가 더 높아도 큰 상자는 후보가 아니다
    assert len(detections) == 1 and len(ignored) == 1


def test_zero_turns_the_filter_off() -> None:
    detections, best_type, best_conf, ignored = _parse(_detector(0), [(0.3, _box(90))])
    assert (best_type, best_conf, ignored) == ("fall", 0.3, [])
    assert len(detections) == 1


def test_boundary_is_excluded_at_exactly_the_limit_only() -> None:
    _, below, _, ignored_below = _parse(_detector(0.35), [(0.5, _box(34))])
    _, at, _, ignored_at = _parse(_detector(0.35), [(0.5, _box(35))])
    assert below == "fall" and ignored_below == []
    assert at == "normal" and len(ignored_at) == 1


def test_knife_detector_never_filters_by_area() -> None:
    settings = get_settings().model_copy(update={"knife_enabled": True, "fall_max_box_area": 0.35})
    assert KnifeDetectionService(settings)._config().max_box_area_ratio == 0.0


def test_detect_from_jpeg_reports_ignored_boxes_and_danger_false(monkeypatch: pytest.MonkeyPatch) -> None:
    detector = _detector(0.35)
    detector._loaded = True

    class _Model:
        def predict(self, **_: object):  # noqa: ANN202
            return [_Result([(0.9, [0.0, 0.0, 80.0, 80.0])])]  # 화면 64%

    detector._model = _Model()
    monkeypatch.setattr(ycd, "torch", types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)))
    ok, buf = cv2.imencode(".jpg", np.zeros((H, W, 3), np.uint8))

    out = detector.detect_from_jpeg(buf.tobytes())

    assert out["detectedType"] == "normal" and out["confidence"] == 0.0 and out["danger"] is False
    assert out["detections"] == []
    assert out["ignoredBoxes"][0]["area"] > 0.6


def test_only_big_boxes_never_reach_hold_danger() -> None:
    # 큰 상자만 계속 들어오면 점수 0 이 쌓여 유지 조건(6초·50%)을 채우지 못한다.
    fall_hold_tracker.clear("fm_big")
    decisions = [
        fall_hold_tracker.update("fm_big", 0.0, float(t), threshold=0.2, hold_sec=6, hold_ratio=0.5) for t in range(20)
    ]
    assert not any(d.danger for d in decisions)


def test_ignored_log_is_once_per_minute_per_session_and_has_no_full_session_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    now = [0.0]
    limiter = _IgnoredBoxLogLimiter(clock=lambda: now[0])
    with caplog.at_level(logging.INFO):
        limiter.log_ignored("ward_abcdef123456", 0.61, 0.3)
        now[0] = 30.0
        limiter.log_ignored("ward_abcdef123456", 0.61, 0.3)  # 1분 안 - 생략
        limiter.log_ignored("ward_other000000", 0.5, 0.2)  # 다른 세션은 따로 센다
        now[0] = 61.0
        limiter.log_ignored("ward_abcdef123456", 0.62, 0.3)
    messages = [r.getMessage() for r in caplog.records if "FALL-BOX-IGNORED" in r.getMessage()]
    assert len(messages) == 3
    assert all("ward_abcdef123456" not in m for m in messages)
    assert "session=123456" in messages[0]
