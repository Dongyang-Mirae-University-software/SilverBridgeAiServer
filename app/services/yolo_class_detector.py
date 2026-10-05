from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings

_LOG = logging.getLogger(__name__)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[misc, assignment]

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class DetectorConfig:
    enabled: bool
    model_path: str
    conf_threshold: float
    iou_threshold: float
    danger_threshold: float


class YoloClassDetector:
    """라이브 이상감지용 YOLO 감지기 공통부 — FireSmokeDetectionService 와 같은 모양.

    서브클래스는 종류 이름·설정·인정할 클래스만 정한다. 모델이 없거나 로드에 실패해도 예외를 밖으로
    내보내지 않고 그 종류만 꺼진다(detectedType=unknown, danger=False).
    """

    NAME: str = ""
    ENABLED_ENV: str = ""
    DEFAULT_MODEL_FILE: str = ""
    # 모델 클래스명(소문자) → 응답 detectedType. 여기 없는 클래스는 박스도 버린다.
    CLASS_TO_TYPE: dict[str, str] = {}

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._model: Any = None
        self._loaded = False
        self._load_error: str | None = None
        self._class_names: dict[int, str] = {}

    def _config(self) -> DetectorConfig:  # pragma: no cover - 서브클래스 구현
        raise NotImplementedError

    def _candidate_model_paths(self) -> list[Path]:
        raw = (self._config().model_path or "").strip() or self.DEFAULT_MODEL_FILE
        path = Path(raw)
        if path.is_absolute():
            return [path]
        candidates = [
            Path(self._settings.model_base_path) / path,
            Path(__file__).resolve().parents[2] / "models" / path,
            Path("/app/models") / path,
            Path("/models") / path,
        ]
        unique: list[Path] = []
        for candidate in candidates:
            if candidate not in unique:
                unique.append(candidate)
        return unique

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
        if not self._config().enabled:
            self._load_error = f"{self.ENABLED_ENV}=false"
            return
        try:
            from ultralytics import YOLO
        except ImportError:
            self._load_error = "ultralytics 미설치"
            _LOG.error("%s: %s", self.NAME, self._load_error)
            return

        candidates = self._candidate_model_paths()
        path = self.resolved_model_path()
        if not path.is_file():
            tried = ", ".join(str(candidate) for candidate in candidates)
            self._load_error = f"모델 파일 없음: {path} (tried: {tried})"
            _LOG.warning("%s: %s", self.NAME, self._load_error)
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
            accepted = sorted(set(class_names.values()) & set(self.CLASS_TO_TYPE))
            _LOG.info("%s 모델 로드 완료: %s classes=%s accepted=%s", self.NAME, path, class_names, accepted)
            if not accepted:
                # 클래스명이 바뀐 모델을 넣으면 에러 없이 영원히 normal 만 낸다 — 로드 시점에 드러낸다.
                _LOG.warning(
                    "%s 모델에 인정할 클래스(%s)가 없다 — 이 종류는 감지하지 못한다",
                    self.NAME,
                    sorted(self.CLASS_TO_TYPE),
                )
        except Exception as exc:  # noqa: BLE001
            self._model = None
            self._loaded = False
            self._load_error = str(exc)
            _LOG.exception("%s 모델 로드 실패: %s", self.NAME, exc)

    def detect_from_jpeg(self, frame_bytes: bytes) -> dict[str, Any]:
        """JPEG 프레임 1장 분석. 반환 형태는 FireSmokeDetectionService.detect_from_jpeg 와 같다.

        danger 는 이 한 장만 본 판정이다(점수 >= danger 임계). 추론 실패·모델 미로드 등 판정 불가
        상황에서는 False 를 유지한다(알람은 확신할 때만).
        """
        config = self._config()
        base: dict[str, Any] = {
            "detectedType": "normal",
            "confidence": 0.0,
            "danger": False,
            "detectedAt": datetime.utcnow().isoformat(),
            "detections": [],
        }
        if not frame_bytes:
            base["detectedType"] = "unknown"
            return base
        if not config.enabled:
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
                    conf=config.conf_threshold,
                    iou=config.iou_threshold,
                    device=0 if torch.cuda.is_available() else "cpu",
                    verbose=False,
                )
        except Exception as exc:  # noqa: BLE001
            _LOG.exception("%s 추론 실패: %s", self.NAME, exc)
            base["detectedType"] = "unknown"
            base["loadError"] = str(exc)
            return base

        detections, best_type, best_conf = self._parse_boxes(results)
        base["detectedType"] = best_type
        base["confidence"] = round(best_conf, 4)
        base["detections"] = detections
        base["danger"] = best_type != "normal" and best_conf >= config.danger_threshold
        return base

    def _parse_boxes(self, results: Any) -> tuple[list[dict[str, Any]], str, float]:
        detections: list[dict[str, Any]] = []
        best_type = "normal"
        best_conf = 0.0
        if not results:
            return detections, best_type, best_conf
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or not len(boxes):
            return detections, best_type, best_conf
        xyxy = getattr(boxes, "xyxy", None)
        conf = getattr(boxes, "conf", None)
        cls = getattr(boxes, "cls", None)
        for i in range(len(boxes)):
            class_id = int(cls[i].item()) if cls is not None else -1
            raw_name = self._class_names.get(class_id, str(class_id)).lower()
            detected_type = self.CLASS_TO_TYPE.get(raw_name)
            if detected_type is None:
                continue
            score = float(conf[i].item()) if conf is not None else 0.0
            x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy[i].tolist()]
            detections.append(
                {
                    "detectedType": detected_type,
                    "confidence": round(score, 4),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                },
            )
            if score > best_conf:
                best_conf = score
                best_type = detected_type
        return detections, best_type, best_conf
