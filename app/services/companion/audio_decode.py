from __future__ import annotations

import io

import numpy as np

TARGET_RATE = 16000  # 음성 인식 모델이 받는 표본 비율


def decode_to_pcm16k(data: bytes) -> np.ndarray:
    """WAV·webm·mp3 바이트를 16kHz 모노 float32 배열로 바꾼다 (av 로 디코딩).

    SenseVoice 는 파일 경로 대신 이 배열을 받는다. 디코딩에 실패하면 ValueError.
    """
    import av

    try:
        container = av.open(io.BytesIO(data))
    except av.error.InvalidDataError as exc:
        raise ValueError("음성을 읽을 수 없습니다") from exc
    resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_RATE)
    chunks: list[np.ndarray] = []
    with container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None) or []:
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    pcm = np.concatenate(chunks).astype(np.float32) / 32768.0
    return pcm
