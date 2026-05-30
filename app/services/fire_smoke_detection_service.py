from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings, get_settings

_LOG = logging.getLogger(__name__)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[misc, assignment]

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


class FireSmokeDetectionService:
    """fire_smoke.pt YOLO — 화재·연기 감지 (보호자 모니터링 표시 전용)."""

    TARGET_CLASSES = {"fire", "smoke"}

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._model: Any = None
        self._loaded = False
        self._load_error: str | None = None
        self._class_names: dict[int, str] = {}

    def _candidate_model_paths(self) -> list[Path]:
        raw = (self._settings.fire_smoke_model_path or "").strip()
        if raw:
            path = Path(raw)
            if not path.is_absolute():
                candidates = [
                    Path(self._settings.model_base_path) / path,
                    Path(__file__).resolve().parents[2] / "models" / path,
                    Path("/app/models") / path,
                    Path("/models") / path,
                ]
                unique_candidates: list[Path] = []
                for candidate in candidates:
                    if candidate not in unique_candidates:
                        unique_candidates.append(candidate)
                return unique_candidates
            return [path]

        return [
            Path(self._settings.model_base_path) / "fire_smoke.pt",
            Path(__file__).resolve().parents[2] / "models" / "fire_smoke.pt",
            Path("/app/models/fire_smoke.pt"),
            Path("/models/fire_smoke.pt"),
        ]

    def resolved_model_path(self) -> Path:
        candidates = self._candidate_model_paths()
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return candidates[0]

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def try_load(self) -> None:
        if not self._settings.fire_smoke_enabled:
            self._load_error = "FIRE_SMOKE_ENABLED=false"
            return
        try:
            from ultralytics import YOLO
        except ImportError:
            self._load_error = "ultralytics 미설치"
            _LOG.error(self._load_error)
            return

        candidates = self._candidate_model_paths()
        path = self.resolved_model_path()
        if not path.is_file():
            tried = ", ".join(str(candidate) for candidate in candidates)
            self._load_error = f"모델 파일 없음: {path} (tried: {tried})"
            _LOG.warning(self._load_error)
            return

        try:
            model = YOLO(str(path))
            names = getattr(model, "names", None) or {}
            class_names: dict[int, str] = {}
            if isinstance(names, dict):
                for key, value in names.items():
                    class_names[int(key)] = str(value).lower()
            self._model = model
            self._class_names = class_names
            self._loaded = True
            self._load_error = None
            _LOG.info("fire_smoke 모델 로드 완료: %s classes=%s", path, class_names)
        except Exception as exc:  # noqa: BLE001
            self._model = None
            self._loaded = False
            self._load_error = str(exc)
            _LOG.exception("fire_smoke 모델 로드 실패: %s", exc)

    def detect_from_jpeg(self, frame_bytes: bytes) -> dict[str, Any]:
        """JPEG 프레임 1장 분석. danger 는 항상 False (표시 전용)."""
        base = {
            "detectedType": "normal",
            "confidence": 0.0,
            "danger": False,
            "detectedAt": datetime.utcnow().isoformat(),
            "detections": [],
        }
        if not frame_bytes:
            base["detectedType"] = "unknown"
            return base
        if not self._settings.fire_smoke_enabled:
            return base
        if not self._loaded or self._model is None:
            base["detectedType"] = "unknown"
            base["loadError"] = self._load_error
            return base
        if cv2 is None:
            base["detectedType"] = "unknown"
            base["loadError"] = "opencv 미설치"
            return base
        if torch is None:
            base["detectedType"] = "unknown"
            base["loadError"] = "torch 미설치"
            return base

        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            base["detectedType"] = "unknown"
            return base

        try:
            with self._lock:
                results = self._model.predict(
                    source=frame,
                    conf=self._settings.fire_smoke_conf_threshold,
                    iou=self._settings.fire_smoke_iou_threshold,
                    device=0 if torch.cuda.is_available() else "cpu",
                    verbose=False,
                )
        except Exception as exc:  # noqa: BLE001
            _LOG.exception("fire_smoke 추론 실패: %s", exc)
            base["detectedType"] = "unknown"
            base["loadError"] = str(exc)
            return base

        detections: list[dict[str, Any]] = []
        best_type = "normal"
        best_conf = 0.0

        if results:
            r0 = results[0]
            boxes = getattr(r0, "boxes", None)
            if boxes is not None and len(boxes):
                xyxy = getattr(boxes, "xyxy", None)
                conf = getattr(boxes, "conf", None)
                cls = getattr(boxes, "cls", None)
                for i in range(len(boxes)):
                    class_id = int(cls[i].item()) if cls is not None else -1
                    raw_name = self._class_names.get(class_id, str(class_id)).lower()
                    score = float(conf[i].item()) if conf is not None else 0.0
                    if raw_name not in self.TARGET_CLASSES:
                        continue
                    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy[i].tolist()]
                    detections.append(
                        {
                            "detectedType": raw_name,
                            "confidence": round(score, 4),
                            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        },
                    )
                    if score > best_conf:
                        best_conf = score
                        best_type = raw_name

        base["detectedType"] = best_type
        base["confidence"] = round(best_conf, 4)
        base["detections"] = detections
        return base


_detector: FireSmokeDetectionService | None = None


def get_fire_smoke_detector() -> FireSmokeDetectionService:
    global _detector
    if _detector is None:
        _detector = FireSmokeDetectionService(get_settings())
    return _detector
