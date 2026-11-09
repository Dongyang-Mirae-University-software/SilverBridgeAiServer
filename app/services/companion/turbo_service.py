"""Whisper large-v3-turbo 받아쓰기 (transformers, GPU fp16). 11/09 말벗 기본 엔진.

근거 (DGX cmp_stt.py hf_turbo · cmp_far.py, 노션 "AI 안부말벗·정서분석 모델 학습 기록"):
- 어르신 200문장 글자 오류 깨끗한 소리 2.4% (SenseVoice 4.7%), 받아쓰기 p50 0.18초, GPU 메모리 약 1.5GB.
- 토큰 평균 로그확률(마지막 토큰 제외)이 -0.3 미만이면 되묻는다: 맞게 받아쓴 문장 되묻기 0/139, 틀린 문장(잡음 5dB) 71/103 잡음.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import numpy as np

from app.core.config import get_settings
from app.services.companion.audio_decode import TARGET_RATE, decode_to_pcm16k
from app.services.companion.repetition import clean_transcript
from app.services.companion.stt_service import SttResult, SttUnavailable

WINDOW_SEC = 30  # Whisper 한 번에 듣는 길이. 더 긴 음성은 30초씩 나눠 받아쓴다
_EMPTY_LOGPROB = -9.0  # 토큰이 하나도 없을 때 (근거 코드와 같은 값)


def window_logprobs(token_logprobs: list[float]) -> list[float]:
    """창 하나의 토큰 로그확률에서 유한한 값만 남기고 마지막 토큰(끝 표시)을 뺀다 (근거 cmp_far.py 와 같은 계산)."""
    vals = [v for v in token_logprobs if math.isfinite(v)]
    return vals[:-1] if len(vals) > 1 else vals


def mean_logprob(values: list[float]) -> float:
    return sum(values) / len(values) if values else _EMPTY_LOGPROB


class TurboService:
    name = "whisper-large-v3-turbo"

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._lock = threading.Lock()  # 모델 올리기·GPU 추론을 한 번에 하나씩
        self._load_error: str | None = None

    def model_dir(self) -> Path:
        s = get_settings()
        return Path(s.model_base_path) / s.companion_turbo_subdir

    def _load(self):
        if self._model is not None:
            return self._model, self._processor
        if self._load_error is not None:
            raise SttUnavailable(self._load_error)
        with self._lock:
            if self._model is not None:
                return self._model, self._processor
            folder = self.model_dir()
            if not (folder / "config.json").is_file():
                self._load_error = "받아쓰기 모델 파일이 없습니다 (whisper-large-v3-turbo)"
                raise SttUnavailable(self._load_error)
            try:
                import torch
                from transformers import WhisperForConditionalGeneration, WhisperProcessor

                device = get_settings().companion_turbo_device
                dtype = torch.float16 if device.startswith("cuda") else torch.float32
                self._processor = WhisperProcessor.from_pretrained(str(folder))
                self._model = WhisperForConditionalGeneration.from_pretrained(str(folder), dtype=dtype).to(device).eval()
            except Exception as exc:  # noqa: BLE001 - 실패는 상태로 남겨 같은 실패를 반복하지 않는다
                self._model = None
                self._load_error = f"받아쓰기 모델을 불러오지 못했습니다 ({type(exc).__name__})"
                raise SttUnavailable(self._load_error) from exc
            return self._model, self._processor

    def is_loaded(self) -> bool:
        return self._model is not None

    def warm_up(self) -> None:
        self._load()

    def _generate(self, pcm: np.ndarray) -> tuple[str, list[float]]:
        """30초 이하 소리 하나를 받아쓴다. (글자, 토큰별 로그확률). 테스트에서 바꿔 끼운다."""
        import torch

        model, proc = self._load()
        s = get_settings()
        with self._lock:
            feats = proc(pcm, sampling_rate=TARGET_RATE, return_tensors="pt").input_features
            feats = feats.to(model.device, model.dtype)
            with torch.no_grad():
                out = model.generate(feats, language="ko", task="transcribe", return_dict_in_generate=True,
                                     output_scores=True, max_new_tokens=s.companion_turbo_max_new_tokens)
                scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)[0]
            text = proc.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
            return text, [float(v) for v in scores.float().cpu().numpy()]

    def transcribe(self, audio: bytes) -> SttResult:
        try:
            pcm = decode_to_pcm16k(audio)
        except ValueError as exc:
            raise SttUnavailable(str(exc)) from exc
        return self.transcribe_pcm(pcm)

    def transcribe_pcm(self, pcm: np.ndarray) -> SttResult:
        """16kHz 모노 소리를 글자로 바꾼다. 말소리 감지는 부르는 쪽(stt_engine)이 먼저 한다."""
        started = time.monotonic()
        if pcm.size == 0:
            return SttResult(text="", confidence=0.0, elapsed_ms=0)
        step = WINDOW_SEC * TARGET_RATE
        texts: list[str] = []
        logprobs: list[float] = []
        for i in range(0, pcm.size, step):
            text, lps = self._generate(pcm[i:i + step])
            if text:
                texts.append(text)
            logprobs.extend(window_logprobs(lps))
        text = clean_transcript(" ".join(texts), get_settings().companion_stt_collapse_repeats)
        elapsed = int((time.monotonic() - started) * 1000)
        if not text:
            return SttResult(text="", confidence=0.0, elapsed_ms=elapsed)
        lp = mean_logprob(logprobs)
        return SttResult(text=text, confidence=max(0.0, min(1.0, math.exp(lp))), elapsed_ms=elapsed, avg_logprob=lp)


turbo_service = TurboService()
