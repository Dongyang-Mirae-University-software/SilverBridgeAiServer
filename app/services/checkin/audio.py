"""어르신 대답 음성 처리: 소리 파일 -> 16kHz 모노 배열, 그리고 감정이 아닌 '측정값'.

목소리로 감정을 맞히는 모델은 쓰지 않는다(2026-10-25 DGX 시험: 한국어 일상 대화에서 평온을 부정으로 55~85% 판정).
대신 대답 길이·말한 시간·쉬는 비율·말 속도·크기처럼 정확히 잴 수 있는 값만 내고, 평소 대비 판단은 백엔드가 한다.
음성은 메모리에서만 다루고 저장하지 않는다.
"""

from __future__ import annotations

import io

import numpy as np

SR = 16000
FRAME = 480  # 30ms


class AudioError(ValueError):
    """code 는 API 오류 코드 그대로 쓴다."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def decode(raw: bytes) -> np.ndarray:
    if not raw:
        raise AudioError("AUDIO_EMPTY")
    if raw[:4] == b"RIFF":
        import soundfile as sf

        try:
            arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        except Exception:
            raise AudioError("AUDIO_UNREADABLE") from None
        arr = arr.mean(axis=1)
        if sr != SR:
            n = int(len(arr) * SR / sr)
            arr = np.interp(np.linspace(0, len(arr) - 1, n), np.arange(len(arr)), arr).astype("float32")
        return arr
    import av  # webm·ogg·m4a·mp3 (브라우저 MediaRecorder 기본은 webm/opus)

    try:
        container = av.open(io.BytesIO(raw))
        if not container.streams.audio:
            raise ValueError
        resampler = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=SR)
        chunks = []
        for frame in container.decode(audio=0):
            chunks.extend(out.to_ndarray().reshape(-1) for out in resampler.resample(frame))
        chunks.extend(out.to_ndarray().reshape(-1) for out in resampler.resample(None))
        container.close()
    except Exception:
        raise AudioError("AUDIO_UNREADABLE") from None
    if not chunks:
        raise AudioError("AUDIO_UNREADABLE")
    return np.concatenate(chunks).astype("float32")


def voice_metrics(arr: np.ndarray, transcript: str) -> dict:
    """에너지 기준으로 말한 구간을 나눠 잰다. 값은 같은 사람의 평소와 비교할 때만 의미가 있다."""
    answer_sec = round(len(arr) / SR, 2)
    n = len(arr) // FRAME
    if n == 0:
        return {"answerSec": answer_sec, "speechSec": 0.0, "pauseRatio": None, "charsPerSec": None, "loudnessDb": None}
    frames = arr[: n * FRAME].reshape(n, FRAME)
    rms = np.sqrt((frames.astype("float64") ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    # 잡음 바닥(하위 10%) + 15dB 를 말소리 기준으로 본다. 너무 조용한 녹음은 최소 -50dBFS
    threshold = max(float(np.percentile(db, 10)) + 15.0, -50.0)
    voiced = db > threshold
    speech_sec = round(float(voiced.sum()) * FRAME / SR, 2)
    if speech_sec <= 0.2:
        return {"answerSec": answer_sec, "speechSec": speech_sec, "pauseRatio": None, "charsPerSec": None, "loudnessDb": None}
    idx = np.flatnonzero(voiced)
    span = (idx[-1] - idx[0] + 1)  # 첫 말소리 ~ 마지막 말소리 사이에서 쉰 비율 (앞뒤 침묵 제외)
    pause_ratio = round(1 - float(voiced[idx[0]: idx[-1] + 1].sum()) / span, 3)
    chars = sum(1 for ch in transcript if "가" <= ch <= "힣")
    return {
        "answerSec": answer_sec,
        "speechSec": speech_sec,
        "pauseRatio": pause_ratio,
        "charsPerSec": round(chars / speech_sec, 2) if chars else None,
        "loudnessDb": round(float(db[voiced].mean()), 1),
    }
