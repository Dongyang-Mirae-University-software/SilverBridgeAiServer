from __future__ import annotations

from app.core.config import get_settings
from app.services.yolo_class_detector import DetectorConfig, YoloClassDetector


class FallDetectionService(YoloClassDetector):
    """fall.pt YOLO — 바닥에 쓰러져 있는 사람(fallen) 감지.

    모델 클래스는 fallen 이지만 응답 detectedType 은 반드시 "fall" 이다 — 백엔드는 fall 만 낙상으로
    받고 fallen 은 모르는 값으로 버린다.

    여기서의 danger 는 한 장 판정일 뿐이다. 낙상은 한 장으로 알리지 않는다 — 세션별 유지 조건은
    FallHoldTracker 가 판정한다(stream_session_service._detect_and_store).
    """

    NAME = "fall"
    ENABLED_ENV = "FALL_ENABLED"
    DEFAULT_MODEL_FILE = "fall.pt"
    CLASS_TO_TYPE = {"fallen": "fall"}

    def _config(self) -> DetectorConfig:
        s = self._settings
        return DetectorConfig(
            enabled=s.fall_enabled,
            model_path=s.fall_model_path,
            conf_threshold=s.fall_conf_threshold,
            iou_threshold=s.fall_iou_threshold,
            danger_threshold=s.fall_danger_threshold,
        )


_detector: FallDetectionService | None = None


def get_fall_detector() -> FallDetectionService:
    global _detector
    if _detector is None:
        _detector = FallDetectionService(get_settings())
    return _detector
