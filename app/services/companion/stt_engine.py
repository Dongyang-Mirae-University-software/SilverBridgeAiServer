from __future__ import annotations

from typing import Callable

import numpy as np

from app.core.config import get_settings
from app.services.companion.audio_decode import decode_to_pcm16k
from app.services.companion.stt_service import SttResult, SttUnavailable
from app.services.companion.vad import has_speech


def _engine(name: str):
    if name == "whisper":
        from app.services.companion.stt_service import stt_service

        return stt_service
    if name == "turbo":
        from app.services.companion.turbo_service import turbo_service

        return turbo_service
    from app.services.companion.sensevoice_service import sensevoice_service

    return sensevoice_service


def get_stt_engine():
    """설정(COMPANION_STT_ENGINE)에 맞는 받아쓰기 엔진을 돌려준다.

    turbo: 11/09 기본 (Whisper large-v3-turbo GPU). sensevoice: 재희 버전 기본. whisper: faster-whisper (비교·대체용).
    모두 transcribe(bytes), warm_up() 을 가진다.
    """
    return _engine(get_settings().companion_stt_engine)


def gated_pcm_transcriber(engine) -> Callable[[np.ndarray], SttResult]:
    """말소리 감지(COMPANION_VAD)를 먼저 하고, 말소리가 없으면 엔진을 부르지 않고 빈 결과를 돌려준다."""

    def run(pcm: np.ndarray) -> SttResult:
        if not has_speech(pcm):
            return SttResult(text="", confidence=0.0, elapsed_ms=0)
        return engine.transcribe_pcm(pcm)

    return run


def transcribe_turn_audio(audio: bytes) -> SttResult:
    """말벗 턴 음성: 디코딩 → 말소리 감지 → 받아쓰기.

    whisper(faster-whisper)는 자체 무음 거르기를 쓰므로 그대로 넘긴다 (재희 버전과 같음).
    sensevoice + rms 이면 재희 버전과 같은 결과다 (SenseVoice 안에도 같은 RMS 기준이 있다).
    """
    engine = get_stt_engine()
    if get_settings().companion_stt_engine == "whisper":
        return engine.transcribe(audio)
    try:
        pcm = decode_to_pcm16k(audio)
    except ValueError as exc:
        raise SttUnavailable(str(exc)) from exc
    return gated_pcm_transcriber(engine)(pcm)


def get_stt_api_transcriber() -> Callable[[np.ndarray], SttResult]:
    """챗봇 받아쓰기 API(/companion/stt) 엔진 (COMPANION_STT_API_ENGINE).

    sensevoice(기본): 지금과 같은 SenseVoice 그대로. turbo: 말소리 감지(COMPANION_VAD) + turbo + 반복 정리.
    """
    if get_settings().companion_stt_api_engine == "turbo":
        return gated_pcm_transcriber(_engine("turbo"))
    return _engine("sensevoice").transcribe_pcm
