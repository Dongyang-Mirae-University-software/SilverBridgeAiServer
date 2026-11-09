"""말소리 감지: 받아쓰기 전에 사람 말이 실제로 있는지 본다. 없으면 받아쓰기를 하지 않는다 (무응답 턴).

silero (11/09 기본): Silero VAD 로 찾은 말소리 구간 합계가 기준(0.3초) 이상일 때만 말로 본다.
  근거: DGX jh_vad.py (11/09) - 어르신 말소리 200문장 x 깨끗함·잡음 10dB·5dB 모두 100% 통과,
  생활 소음(방 소리) 20개는 RMS 기준 0% → Silero 100% 걸러냄.
rms (재희 버전): 전체 소리 크기(RMS)가 0.01 미만이면 무음. 생활 소음은 크기가 커서 통과한다.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from app.core.config import get_settings
from app.services.companion.audio_decode import TARGET_RATE

SILENCE_RMS = 0.01  # 재희 기준 (sensevoice_service.SILENCE_RMS 와 같은 값)

logger = logging.getLogger("companion.vad")


def rms(pcm: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(pcm)))) if pcm.size else 0.0


class SileroVad:
    """Silero VAD (pip silero-vad, 모델 파일이 패키지 안에 들어 있다). 모델은 상태를 가져 한 번에 하나씩 돌린다."""

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._load_failed = False

    def _load(self):
        if self._model is None and not self._load_failed:
            try:
                from silero_vad import load_silero_vad

                self._model = load_silero_vad()
            except Exception as exc:  # noqa: BLE001 - 못 쓰면 재희 기준으로 대신 판단한다 (대화 우선)
                self._load_failed = True
                logger.warning("[VAD] Silero 를 불러오지 못해 소리 크기 기준을 씁니다 (%s)", type(exc).__name__)
        return self._model

    def speech_seconds(self, pcm: np.ndarray) -> float | None:
        """말소리 구간 합계(초). Silero 를 쓸 수 없으면 None."""
        with self._lock:
            model = self._load()
            if model is None:
                return None
            import torch
            from silero_vad import get_speech_timestamps

            stamps = get_speech_timestamps(torch.from_numpy(np.ascontiguousarray(pcm, dtype=np.float32)), model,
                                           sampling_rate=TARGET_RATE)
            return sum(t["end"] - t["start"] for t in stamps) / TARGET_RATE


silero_vad = SileroVad()


def has_speech(pcm: np.ndarray) -> bool:
    """받아쓰기를 할 만큼 말소리가 있으면 True."""
    if pcm.size == 0:
        return False
    s = get_settings()
    if s.companion_vad == "silero":
        seconds = silero_vad.speech_seconds(pcm)
        if seconds is not None:
            return seconds >= s.companion_vad_min_speech_sec
    return rms(pcm) >= SILENCE_RMS
