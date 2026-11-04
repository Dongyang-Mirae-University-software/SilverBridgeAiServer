from __future__ import annotations

import os
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.core.config import get_settings

DETECT_FILE = "face_detection_yunet_2023mar.onnx"  # YuNet (OpenCV Zoo). 결정 2026-11-01: 검출기 YuNet
EMOTION_MODEL = "enet_b0_8_best_afew"  # D24 결정: HSEmotion
EMOTION_FILE = EMOTION_MODEL + ".onnx"


def _opencv_path(path: Path) -> str:
    """현재 폴더 아래에 있으면 상대 경로, 아니면 절대 경로를 돌려준다."""
    try:
        return os.path.relpath(path, os.getcwd())
    except ValueError:  # 드라이브가 다르면 상대 경로를 만들 수 없다
        return str(path)


class FaceUnavailable(Exception):
    """표정 모델이나 검출기를 쓸 수 없음 (파일 없음, 불러오기 실패). 대화는 계속되고 표정만 빠진다."""


@dataclass
class FaceResult:
    label: str | None  # 최빈 표정. 얼굴이 없으면 None
    confidence: float  # 최빈 표정 프레임들의 평균 확률
    frameCount: int  # 최빈 표정이 나온 프레임 수
    usable: bool  # D6: 얼굴 검출 비율이 companion_face_min_rate 이상이면 True
    totalFrames: int
    facesFound: int


def summarize(detections: list[tuple[str, float]], total: int, min_rate: float) -> FaceResult:
    """프레임별 (표정, 확률) 목록을 턴 표정 하나로 줄인다 (S4-4).

    detections: 얼굴이 검출된 프레임만 담는다. total: 보낸 프레임 전체 수.
    """
    if not detections or total <= 0:
        return FaceResult(None, 0.0, 0, False, total, 0)
    counts = Counter(label for label, _ in detections)
    mode, mode_count = counts.most_common(1)[0]
    mode_conf = float(np.mean([p for label, p in detections if label == mode]))
    usable = len(detections) / total >= min_rate
    return FaceResult(mode, round(mode_conf, 4), mode_count, usable, total, len(detections))


class FaceService:
    """YuNet 검출 + HSEmotion 분류 (S4). 모델은 처음 쓸 때 한 번만 올린다."""

    def __init__(self) -> None:
        self._detector = None
        self._recognizer = None
        self._lock = threading.Lock()
        self._load_error: str | None = None

    def _paths(self) -> tuple[Path, Path]:
        s = get_settings()
        base = Path(s.model_base_path)
        return (base / s.companion_face_detect_subdir / DETECT_FILE,
                base / s.companion_face_model_subdir / EMOTION_FILE)

    def _load(self):
        if self._detector is not None and self._recognizer is not None:
            return self._detector, self._recognizer
        if self._load_error is not None:
            raise FaceUnavailable(self._load_error)
        with self._lock:
            if self._detector is not None and self._recognizer is not None:
                return self._detector, self._recognizer
            detect_path, emotion_path = self._paths()
            if not detect_path.is_file() or not emotion_path.is_file():
                self._load_error = "표정 모델 또는 얼굴 검출 모델 파일이 없습니다"
                raise FaceUnavailable(self._load_error)
            try:
                import cv2
                from emotiefflib import facial_analysis as fa

                # OpenCV 의 ONNX 읽기는 한글 경로에서 실패할 수 있어 현재 폴더 기준 상대 경로로 넘긴다 (Windows 개발 PC)
                detector = cv2.FaceDetectorYN.create(_opencv_path(detect_path), "", (320, 320), 0.6, 0.3, 10)
                # emotiefflib 는 모델을 자기 캐시에 받는다. 우리 폴더의 파일을 쓰도록 경로 함수만 바꾼다 (D19).
                original = fa.get_model_path_onnx
                fa.get_model_path_onnx = lambda _name: str(emotion_path)
                try:
                    recognizer = fa.EmotiEffLibRecognizerOnnx(EMOTION_MODEL)
                finally:
                    fa.get_model_path_onnx = original
                self._detector, self._recognizer = detector, recognizer
            except Exception as exc:
                self._load_error = f"표정 모델을 불러오지 못했습니다 ({type(exc).__name__})"
                raise FaceUnavailable(self._load_error) from exc
            return self._detector, self._recognizer

    def warm_up(self) -> None:
        """검출·표정 모델을 미리 올린다 (S9-1 시간 제한 때문에 첫 분석 전에 한다)."""
        self._load()

    def analyze(self, frames: list[bytes]) -> FaceResult:
        """프레임 목록을 턴 표정으로 줄인다. 프레임 원본은 함수 밖으로 남기지 않는다 (S4-5)."""
        import cv2

        s = get_settings()
        detector, recognizer = self._load()
        detections: list[tuple[str, float]] = []
        for raw in frames:
            img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue  # 이미지로 읽을 수 없는 프레임은 얼굴 없음으로 센다
            h, w = img.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(img)
            if faces is None or len(faces) == 0:
                continue
            x, y, fw, fh = (int(v) for v in faces[0][:4])  # 가장 먼저 찾은 얼굴 (score 순)
            x, y = max(0, x), max(0, y)
            crop = img[y:y + fh, x:x + fw]
            if crop.size == 0:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            labels, scores = recognizer.predict_emotions(rgb, logits=False)
            detections.append((str(labels[0]), float(np.max(scores))))
        return summarize(detections, len(frames), s.companion_face_min_rate)


face_service = FaceService()
