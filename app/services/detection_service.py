from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from app.models.ai_model import AIModel


class DetectionService:
    """
    PRD 초기 MVP용 감지 서비스.
    실제 모델 추론 대신 확장 가능한 인터페이스를 우선 제공한다.
    """

    @staticmethod
    def detect_from_frame(model: AIModel, frame_available: bool = True) -> dict:
        confidence = 0.8 if frame_available else 0.0
        detected_type = "fall" if model.type == "fall_detection" else "normal"
        if model.type == "fire_detection":
            detected_type = "fire"
        elif model.type == "weapon_detection":
            detected_type = "weapon"

        if not frame_available:
            detected_type = "unknown"

        danger = confidence >= model.threshold and detected_type not in {"normal", "unknown"}
        return {
            "analysisNo": f"ANL-{uuid4().hex[:12]}",
            "detectedType": detected_type,
            "confidence": confidence,
            "danger": danger,
            "rawResultJson": {
                "modelType": model.type,
                "framework": model.framework,
                "frameAvailable": frame_available,
                "detectedAt": datetime.utcnow().isoformat(),
            },
        }
