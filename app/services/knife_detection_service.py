from __future__ import annotations

from app.core.config import get_settings
from app.services.yolo_class_detector import DetectorConfig, YoloClassDetector


class KnifeDetectionService(YoloClassDetector):
    """knife.pt YOLO — 흉기 감지.

    백엔드는 detectedType "knife" 를 흉기(WEAPON)로 받는다. 6월 모델은 {Knife, Knife_Handle} 두
    클래스라 이름을 소문자로 비교하고 knife 만 인정한다(손잡이만 보인 박스는 흉기로 세지 않는다).
    """

    NAME = "knife"
    ENABLED_ENV = "KNIFE_ENABLED"
    DEFAULT_MODEL_FILE = "knife.pt"
    CLASS_TO_TYPE = {"knife": "knife"}

    def _config(self) -> DetectorConfig:
        s = self._settings
        return DetectorConfig(
            enabled=s.knife_enabled,
            model_path=s.knife_model_path,
            conf_threshold=s.knife_conf_threshold,
            iou_threshold=s.knife_iou_threshold,
            danger_threshold=s.knife_danger_threshold,
        )


_detector: KnifeDetectionService | None = None


def get_knife_detector() -> KnifeDetectionService:
    global _detector
    if _detector is None:
        _detector = KnifeDetectionService(get_settings())
    return _detector
