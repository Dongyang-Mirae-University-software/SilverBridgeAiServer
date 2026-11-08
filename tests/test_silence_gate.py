"""무음 처리: 조용한 녹음은 받아쓰기를 하지 않고, 조용한 소리가 만든 한 글자 환각("그.")은 대답이 아니다."""

import re
import sys
import types

import numpy as np

try:
    import funasr.utils.postprocess_utils  # noqa: F401
except ImportError:  # 개발 PC에 funasr 가 없어도 돌도록 같은 이름의 가짜 모듈을 넣는다
    _post = types.ModuleType("funasr.utils.postprocess_utils")
    _post.rich_transcription_postprocess = lambda s: re.sub(r"<\|[^|]*\|>", "", s)
    sys.modules.setdefault("funasr", types.ModuleType("funasr"))
    sys.modules.setdefault("funasr.utils", types.ModuleType("funasr.utils"))
    sys.modules["funasr.utils.postprocess_utils"] = _post

import pytest  # noqa: E402

from app.services.companion import sensevoice_service as sv  # noqa: E402


class FakeModel:
    def __init__(self, text: str):
        self.text = text
        self.calls = 0

    def generate(self, **_kwargs):
        self.calls += 1
        return [{"text": self.text}]


def make_service(text: str):
    svc = sv.SenseVoiceService()
    fake = FakeModel(text)
    svc._load = lambda: fake  # 실제 모델을 올리지 않는다
    return svc, fake


def speech_like(seconds: float = 1.0, amp: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(-amp, amp, int(16000 * seconds)).astype(np.float32)


def test_silence_returns_empty_without_calling_model():
    svc, fake = make_service("안녕하세요")
    result = svc.transcribe_pcm(np.zeros(16000, dtype=np.float32))
    assert result.text == ""
    assert fake.calls == 0


def test_very_quiet_noise_below_threshold_is_silence():
    svc, fake = make_service("안녕하세요")
    result = svc.transcribe_pcm(speech_like(amp=0.002))  # RMS 약 0.001 < 0.01
    assert result.text == ""
    assert fake.calls == 0


def test_speech_level_audio_reaches_model():
    svc, fake = make_service("잘 잤어요")
    result = svc.transcribe_pcm(speech_like(amp=0.1))  # RMS 약 0.058 > 0.01
    assert result.text == "잘 잤어요"
    assert fake.calls == 1


def test_filler_hallucination_becomes_empty():
    svc, _ = make_service("그.")
    assert svc.transcribe_pcm(speech_like(amp=0.1)).text == ""


@pytest.mark.parametrize("answer", ["네", "예", "응"])
def test_single_char_real_answers_are_kept(answer):
    svc, _ = make_service(answer)
    assert svc.transcribe_pcm(speech_like(amp=0.1)).text == answer
