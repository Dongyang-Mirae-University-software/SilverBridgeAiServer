from __future__ import annotations

import io
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings


class SttUnavailable(Exception):
    """음성 인식 모델을 쓸 수 없음 (모델 파일 없음, 불러오기 실패)."""


@dataclass
class SttResult:
    text: str
    confidence: float  # 0~1. D13: 세그먼트 평균 로그확률을 지수 변환해 얻는다
    elapsed_ms: int
    # turbo 토큰 평균 로그확률 (되묻기 기준에 쓴다). 확률을 내지 않는 엔진·무음은 None
    avg_logprob: float | None = None


def confidence_from_logprobs(avg_logprobs: list[float]) -> float:
    """D13 결정: 세그먼트별 평균 로그확률의 평균을 exp()로 바꿔 0~1 값을 만든다. 세그먼트가 없으면 0."""
    if not avg_logprobs:
        return 0.0
    mean = sum(avg_logprobs) / len(avg_logprobs)
    return max(0.0, min(1.0, math.exp(mean)))


class SttService:
    """faster-whisper 래퍼 (S3). 모델은 처음 쓸 때 한 번만 올리고, 동시 요청은 lock 으로 막는다 (S3-4)."""

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._load_error: str | None = None

    def model_dir(self) -> Path:
        s = get_settings()
        # D19 결정: MODEL_BASE_PATH 아래 모델별 폴더. 예: models/companion/stt/large-v3
        return Path(s.model_base_path) / s.companion_stt_subdir / s.companion_stt_model

    def _load(self):
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            raise SttUnavailable(self._load_error)
        with self._lock:
            if self._model is not None:
                return self._model
            path = self.model_dir()
            if not path.is_dir():
                self._load_error = "음성 인식 모델을 사용할 수 없습니다."
                raise SttUnavailable(self._load_error)
            try:
                from faster_whisper import WhisperModel

                s = get_settings()
                self._model = WhisperModel(str(path), device=s.companion_stt_device,
                                           compute_type=s.companion_stt_compute)
            except Exception as exc:  # 로드 실패는 상태로 남기고, 다음 요청이 같은 실패를 반복하지 않게 한다
                self._load_error = f"음성 인식 모델을 불러오지 못했습니다 ({type(exc).__name__})"
                raise SttUnavailable(self._load_error) from exc
            return self._model

    def is_loaded(self) -> bool:
        return self._model is not None

    def warm_up(self) -> None:
        """음성 인식 모델을 미리 올린다 (S9-1 시간 제한 때문에 첫 인식 전에 한다)."""
        self._load()

    def transcribe(self, audio: bytes) -> SttResult:
        """음성 바이트를 글자로 바꾼다. 파일로 쓰지 않고 메모리에서만 읽는다 (faster-whisper 가 파일 객체를 받는다)."""
        model = self._load()
        started = time.monotonic()
        segments, _info = model.transcribe(
            io.BytesIO(audio),
            language="ko",
            beam_size=5,
            vad_filter=True,  # 무음 구간은 버린다 (무음이면 글자가 비어 무응답으로 처리)
        )
        texts: list[str] = []
        logprobs: list[float] = []
        for seg in segments:
            if seg.text.strip():
                texts.append(seg.text.strip())
                logprobs.append(seg.avg_logprob)
        return SttResult(
            text=" ".join(texts),
            confidence=confidence_from_logprobs(logprobs),
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )


stt_service = SttService()
