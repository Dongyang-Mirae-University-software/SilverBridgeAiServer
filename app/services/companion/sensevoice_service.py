from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from app.core.config import get_settings
from app.services.companion.audio_decode import decode_to_pcm16k
from app.services.companion.stt_service import SttResult, SttUnavailable

# 팀 합의(10/26): 받아쓰기는 SenseVoiceSmall. 어르신 음성 글자 오류 5.5% (팀 측정).
SENSEVOICE_FILE = "model.pt"


class SenseVoiceService:
    """SenseVoiceSmall 받아쓰기 (FunASR). 모델은 처음 쓸 때 한 번만 올린다."""

    name = "SenseVoiceSmall"

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._load_error: str | None = None

    def model_dir(self) -> Path:
        s = get_settings()
        return Path(s.model_base_path) / s.companion_sensevoice_subdir

    def _load(self):
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            raise SttUnavailable(self._load_error)
        with self._lock:
            if self._model is not None:
                return self._model
            folder = self.model_dir()
            if not (folder / SENSEVOICE_FILE).is_file():
                self._load_error = "받아쓰기 모델 파일이 없습니다 (SenseVoiceSmall)"
                raise SttUnavailable(self._load_error)
            try:
                from funasr import AutoModel

                # SentencePiece 는 한글이 든 절대 경로를 열지 못한다 (Windows 개발 PC). 현재 폴더 기준 상대 경로로 넘긴다.
                rel = os.path.relpath(folder, Path.cwd())
                self._model = AutoModel(model=rel, device="cpu", disable_update=True, disable_pbar=True)
            except Exception as exc:  # noqa: BLE001
                self._load_error = f"받아쓰기 모델을 불러오지 못했습니다 ({type(exc).__name__})"
                raise SttUnavailable(self._load_error) from exc
            return self._model

    def is_loaded(self) -> bool:
        return self._model is not None

    def warm_up(self) -> None:
        self._load()

    def transcribe(self, audio: bytes) -> SttResult:
        try:
            pcm = decode_to_pcm16k(audio)
        except ValueError as exc:
            raise SttUnavailable(str(exc)) from exc
        return self.transcribe_pcm(pcm)

    def transcribe_pcm(self, pcm) -> SttResult:
        """16kHz 모노 소리 데이터를 글자로 바꾼다 (음성 인식 API가 이 함수를 쓴다)."""
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        model = self._load()
        started = time.monotonic()
        if pcm.size == 0:
            return SttResult(text="", confidence=0.0, elapsed_ms=0)
        out = model.generate(input=pcm, language="ko", use_itn=True, batch_size_s=60)
        elapsed = int((time.monotonic() - started) * 1000)
        raw = out[0]["text"] if out else ""
        # <|ko|><|NEUTRAL|>... 같은 표시를 떼고 글자만 남긴다
        text = rich_transcription_postprocess(raw).strip()
        # SenseVoice 는 확신도를 내지 않는다. 확신도 기준은 faster-whisper 에서만 쓴다.
        return SttResult(text=text, confidence=1.0 if text else 0.0, elapsed_ms=elapsed)


sensevoice_service = SenseVoiceService()
