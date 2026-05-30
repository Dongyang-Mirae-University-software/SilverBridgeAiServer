from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from typing import Any, Callable

from app.core.config import get_settings
from app.models.ai_model import AIModel
from app.models.analysis_result import AnalysisResult
from app.models.camera import Camera
from app.services.detection_service import DetectionService
from app.utils.file_utils import make_snapshot_path

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


class StreamAnalysisManager:
    def __init__(self) -> None:
        self._workers: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._settings = get_settings()

    def is_running(self, camera_identifier: str) -> bool:
        with self._lock:
            worker = self._workers.get(camera_identifier)
            return bool(worker and worker["thread"].is_alive())

    def start(self, db_factory: Callable[[], Any], camera: Camera, model: AIModel) -> bool:
        with self._lock:
            existing = self._workers.get(camera.identifier)
            if existing and existing["thread"].is_alive():
                return False

            stop_event = threading.Event()
            thread = threading.Thread(
                target=self._run_worker,
                args=(db_factory, camera, model, stop_event),
                daemon=True,
                name=f"stream-{camera.identifier}",
            )
            self._workers[camera.identifier] = {"thread": thread, "stop_event": stop_event}
            thread.start()
            return True

    def stop(self, camera_identifier: str) -> bool:
        with self._lock:
            worker = self._workers.get(camera_identifier)
            if not worker:
                return False
            worker["stop_event"].set()
            return True

    def _run_worker(
        self,
        db_factory: Callable[[], Any],
        camera: Camera,
        model: AIModel,
        stop_event: threading.Event,
    ) -> None:
        frame_count = 0
        cap = None
        if cv2 is not None:
            cap = cv2.VideoCapture(camera.stream_url)

        while not stop_event.is_set():
            frame_available = False
            frame = None

            if cap is not None and cap.isOpened():
                ok, frame = cap.read()
                frame_available = bool(ok and frame is not None)
            frame_count += 1

            if frame_count % self._settings.stream_sample_every_n_frames == 0:
                result = DetectionService.detect_from_frame(model=model, frame_available=frame_available)
                snapshot_path = ""
                if frame_available and cv2 is not None:
                    snapshot_path = make_snapshot_path(self._settings.snapshot_base_path)
                    cv2.imwrite(snapshot_path, frame)

                session = db_factory()
                try:
                    entity = AnalysisResult(
                        analysis_no=result["analysisNo"],
                        camera_id=camera.id,
                        camera_identifier=camera.identifier,
                        model_id=model.id,
                        model_identifier=model.identifier,
                        detected_type=result["detectedType"],
                        confidence=result["confidence"],
                        danger=result["danger"],
                        snapshot_path=snapshot_path,
                        raw_result_json=json.dumps(result["rawResultJson"], ensure_ascii=False),
                        analyzed_at=datetime.utcnow(),
                    )
                    session.add(entity)
                    session.commit()
                finally:
                    session.close()

            time.sleep(self._settings.stream_fallback_interval_sec)

        if cap is not None:
            cap.release()

        with self._lock:
            self._workers.pop(camera.identifier, None)


stream_analysis_manager = StreamAnalysisManager()
