"""11/09 말벗 음성 단계: Silero 말소리 감지 · turbo 받아쓰기 · 반복 정리 · 확신도 되묻기.

모델은 모두 가짜로 바꿔 끼운다 (실제 모델·API 키를 쓰지 않는다). 스위치를 끄면 재희 동작과 같아야 한다.
"""

from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from app.core.config import get_settings
from app.services.companion import pipeline, stt_engine, vad
from app.services.companion.pipeline import LOW_CONFIDENCE_REPLY
from app.services.companion.reply_service import SAFETY_REPLY
from app.services.companion.repetition import clean_transcript, collapse_repeats, strip_hallucination
from app.services.companion.sensevoice_service import sensevoice_service
from app.services.companion.stt_service import SttResult
from app.services.companion.text_emotion_service import TextEmotionResult
from app.services.companion.turbo_service import TurboService, mean_logprob, window_logprobs
from app.services.companion.llm_client import LlmUnavailable

SR = 16000


@pytest.fixture
def settings(monkeypatch):
    s = get_settings()

    def set_(**kw):
        for k, v in kw.items():
            monkeypatch.setattr(s, k, v)
    return set_


def _noise(seconds: float, amp: float, seed: int = 0) -> np.ndarray:
    return (np.random.default_rng(seed).standard_normal(int(SR * seconds)) * amp).astype(np.float32)


def _wav(pcm: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(pcm, -1, 1) * 32767).astype(np.int16).tobytes())
    return buf.getvalue()


class FakeEngine:
    """transcribe_pcm 만 가진 가짜 엔진. 불린 횟수를 센다."""

    def __init__(self, text="네 잘 먹었어요", logprob=-0.1):
        self.calls = 0
        self.text, self.logprob = text, logprob

    def transcribe_pcm(self, pcm):
        self.calls += 1
        return SttResult(self.text, 0.9, 1, avg_logprob=self.logprob)


# ---------- 말소리 감지 ----------

def test_silero_needs_point_three_seconds_of_speech(settings, monkeypatch):
    settings(companion_vad="silero", companion_vad_min_speech_sec=0.3)
    loud = _noise(2.0, 0.1)
    monkeypatch.setattr(vad.silero_vad, "speech_seconds", lambda _p: 0.29)
    assert vad.has_speech(loud) is False
    monkeypatch.setattr(vad.silero_vad, "speech_seconds", lambda _p: 0.3)
    assert vad.has_speech(loud) is True


def test_room_noise_is_rejected_by_silero_but_passed_by_rms(settings, monkeypatch):
    room = _noise(3.0, 0.05)  # 생활 소음: 크기는 커서 재희 RMS 기준(0.01)은 통과한다
    monkeypatch.setattr(vad.silero_vad, "speech_seconds", lambda _p: 0.0)  # Silero 는 말소리를 못 찾는다
    settings(companion_vad="silero")
    assert vad.has_speech(room) is False
    settings(companion_vad="rms")
    assert vad.has_speech(room) is True  # 재희 동작 그대로


def test_silence_is_rejected_in_both_modes(settings, monkeypatch):
    monkeypatch.setattr(vad.silero_vad, "speech_seconds", lambda _p: 0.0)
    for mode in ("silero", "rms"):
        settings(companion_vad=mode)
        assert vad.has_speech(np.zeros(SR, dtype=np.float32)) is False
        assert vad.has_speech(np.zeros(0, dtype=np.float32)) is False


def test_silero_unavailable_falls_back_to_rms(settings, monkeypatch):
    settings(companion_vad="silero")
    monkeypatch.setattr(vad.silero_vad, "speech_seconds", lambda _p: None)
    assert vad.has_speech(_noise(1.0, 0.05)) is True
    assert vad.has_speech(_noise(1.0, 0.001)) is False


def test_turn_audio_without_speech_skips_the_engine(settings, monkeypatch):
    settings(companion_vad="silero", companion_stt_engine="turbo")
    engine = FakeEngine()
    monkeypatch.setattr(stt_engine, "_engine", lambda _n: engine)
    monkeypatch.setattr(vad.silero_vad, "speech_seconds", lambda _p: 0.0)
    r = stt_engine.transcribe_turn_audio(_wav(_noise(2.0, 0.05)))
    assert r.text == "" and r.avg_logprob is None and engine.calls == 0
    monkeypatch.setattr(vad.silero_vad, "speech_seconds", lambda _p: 1.5)
    r = stt_engine.transcribe_turn_audio(_wav(_noise(2.0, 0.05)))
    assert r.text == "네 잘 먹었어요" and r.avg_logprob == -0.1 and engine.calls == 1


def test_sensevoice_with_rms_matches_jaehee(settings, monkeypatch):
    """스위치를 모두 끄면(sensevoice + rms) 예전과 같이 SenseVoice 의 transcribe_pcm 결과 그대로다."""
    settings(companion_vad="rms", companion_stt_engine="sensevoice")
    monkeypatch.setattr(sensevoice_service, "transcribe_pcm", lambda pcm: SttResult("예 먹었어요", 1.0, 3))
    r = stt_engine.transcribe_turn_audio(_wav(_noise(1.0, 0.05)))
    assert (r.text, r.confidence, r.avg_logprob) == ("예 먹었어요", 1.0, None)
    r = stt_engine.transcribe_turn_audio(_wav(np.zeros(SR, dtype=np.float32)))
    assert r.text == ""  # 재희 무음 기준 (RMS 0.01)


# ---------- 반복 정리 ----------

@pytest.mark.parametrize("raw, expected", [
    ("네 네 네 네 먹었어요", "네 먹었어요"),
    ("밥 먹었어 밥 먹었어 밥 먹었어", "밥 먹었어"),
    ("오늘 아침에 오늘 아침에 오늘 아침에 산책했어요", "오늘 아침에 산책했어요"),
    ("음 그러니까 밥 먹었어 밥 먹었어 밥 먹었어 요", "음 그러니까 밥 먹었어 요"),  # 홀수 위치에서 시작하는 반복
    ("네 네 먹었어요", "네 네 먹었어요"),  # 두 번은 실제 말일 수 있어 그대로 둔다
    ("고마워요고마워요고마워요고마워요 네", "고마워요 네"),  # 띄어쓰기 없는 2~10글자 묶음 4번 이상
    ("오늘은 날씨가 좋아요", "오늘은 날씨가 좋아요"),
])
def test_collapse_repeats(raw, expected):
    assert collapse_repeats(raw) == expected


def test_hallucinated_caption_is_removed():
    assert strip_hallucination("시청해 주셔서 감사합니다.") == ""
    assert strip_hallucination("밥 먹었어요 자막 제공 배달의민족") == "밥 먹었어요"


def test_collapse_switch_off_keeps_repeats():
    assert clean_transcript("네 네 네 네", collapse=False) == "네 네 네 네"
    assert clean_transcript("네 네 네 네", collapse=True) == "네"


# ---------- turbo 받아쓰기 (모델은 가짜) ----------

def test_logprob_mean_drops_end_token_and_non_finite():
    assert window_logprobs([-0.1, float("-inf"), -0.3, -5.0]) == [-0.1, -0.3]
    assert window_logprobs([-0.4]) == [-0.4]
    assert mean_logprob([]) == -9.0


def test_turbo_splits_long_audio_and_cleans(settings, monkeypatch):
    settings(companion_stt_collapse_repeats=True)
    svc = TurboService()
    seen = []

    def gen(pcm):
        seen.append(pcm.size)
        return ("네 네 네 네" if len(seen) == 1 else "밥 먹었어요"), [-0.2, -0.4, -9.0]

    monkeypatch.setattr(svc, "_generate", gen)
    pcm = _noise(45.0, 0.1)
    pcm[SR * 27:SR * 27 + 800] = 0.0  # 27초 근처 조용한 틈에서 나뉜다
    r = svc.transcribe_pcm(pcm)
    assert len(seen) == 2 and SR * 27 <= seen[0] <= SR * 27 + 800 and sum(seen) == SR * 45
    assert r.text == "네 밥 먹었어요"
    assert r.avg_logprob == pytest.approx(-0.3)  # 창마다 끝 토큰(-9.0)을 빼고 평균


def test_turbo_hallucination_only_is_no_answer(monkeypatch):
    svc = TurboService()
    monkeypatch.setattr(svc, "_generate", lambda pcm: ("시청해 주셔서 감사합니다.", [-0.1, -0.1]))
    r = svc.transcribe_pcm(np.zeros(SR, dtype=np.float32))
    assert r.text == "" and r.avg_logprob is None


def test_turbo_missing_model_is_unavailable(settings, tmp_path):
    from app.services.companion.stt_service import SttUnavailable

    settings(model_base_path=str(tmp_path))
    with pytest.raises(SttUnavailable):
        TurboService().warm_up()


# ---------- 확신도 되묻기 ----------

class ProbLlm:
    """위험 확률만 준다. 대답·신호 생성은 연결 실패."""

    def __init__(self, p_yes: float = 0.0):
        self.p_yes = p_yes
        self.prob_calls = 0

    def generate(self, *_a, **_k):
        raise LlmUnavailable("off")

    def next_token_probs(self, prompt, prefix, candidates, timeout):
        self.prob_calls += 1
        return {"있": self.p_yes, "없": 1 - self.p_yes}


def _turn(transcript, lp, llm=None):
    return pipeline.run_turn(transcript=transcript, face={},
                             classify_text=lambda _t: TextEmotionResult("기쁨", 0.9, True, {}),
                             llm=llm or ProbLlm(), question="점심은 드셨어요?", prev_signals=None,
                             is_last_turn=False, turn_no=1, stt_logprob=lp)


def test_confidence_boundary(settings):
    settings(companion_confirm_mode="logprob", companion_confirm_logprob=-0.3)
    assert _turn("점심에 국수를 먹었어요", -0.3)["replySource"] != "CONFIRM"  # 기준값 자체는 되묻지 않는다
    low = _turn("점심에 국수를 먹었어요", -0.3001)
    assert low["replySource"] == "CONFIRM" and low["needsCheck"] is True and low["signals"] is None
    assert low["reply"]["text"] == LOW_CONFIDENCE_REPLY


def test_low_confidence_never_offers_candidate_words(settings):
    settings(companion_confirm_mode="logprob")
    r = _turn("삼기탕을 먹었어요", -1.0)
    assert r["reply"]["text"] == LOW_CONFIDENCE_REPLY and "삼계탕" not in r["reply"]["text"]


def test_confident_turbo_turn_skips_the_lexicon_rule(settings):
    settings(companion_confirm_mode="logprob")
    assert _turn("삼기탕을 먹었어요", -0.1)["replySource"] != "CONFIRM"


def test_low_confidence_turn_still_runs_risk_model(settings):
    settings(companion_confirm_mode="logprob", companion_risk_prompt="v4")  # 이 가짜 모델은 재희 v4 확률 함수만 있다
    llm = ProbLlm(p_yes=0.9)
    r = _turn("요즘 그냥 다 귀찮아요", -0.8, llm)
    assert llm.prob_calls == 1
    assert r["riskFlag"] is True and r["reply"]["text"] == SAFETY_REPLY and r["replySource"] == "SAFETY"
    r = _turn("숨이 안 쉬어져서 쓰러졌어요", -0.8, ProbLlm(0.0))  # 낱말 규칙
    assert r["riskFlag"] is True and r["reply"]["text"] == SAFETY_REPLY
    r = _turn("국수를 먹었어요", -0.8, ProbLlm(0.1))
    assert r["riskFlag"] is False and r["reply"]["text"] == LOW_CONFIDENCE_REPLY


def test_text_answers_and_sensevoice_keep_the_lexicon_rule(settings):
    settings(companion_confirm_mode="logprob")
    r = _turn("삼기탕을 먹었어요", None)
    assert r["replySource"] == "CONFIRM" and "삼계탕" in r["reply"]["text"]


def test_lexicon_mode_is_jaehee_behaviour(settings):
    settings(companion_confirm_mode="lexicon")
    llm = ProbLlm(0.9)
    r = _turn("삼기탕을 먹었어요", -1.0, llm)
    assert r["replySource"] == "CONFIRM" and "삼계탕" in r["reply"]["text"]
    assert llm.prob_calls == 0  # 재희 확인 턴은 위험 모델을 부르지 않는다
    assert _turn("점심에 국수를 먹었어요", -1.0)["replySource"] != "CONFIRM"


# ---------- 챗봇 받아쓰기 API 엔진 ----------

def test_stt_api_sensevoice_still_selectable(settings):
    """되돌리기용: COMPANION_STT_API_ENGINE=sensevoice 면 예전 SenseVoice 경로 그대로 (2026-11-11 기본은 turbo)."""
    settings(companion_stt_api_engine="sensevoice")
    assert stt_engine.get_stt_api_transcriber() == sensevoice_service.transcribe_pcm


def test_stt_api_turbo_uses_vad_and_turbo(settings, monkeypatch):
    settings(companion_stt_api_engine="turbo", companion_vad="silero")
    engine = FakeEngine(text="내일 병원 예약해 줘")
    monkeypatch.setattr(stt_engine, "_engine", lambda _n: engine)
    run = stt_engine.get_stt_api_transcriber()
    monkeypatch.setattr(vad.silero_vad, "speech_seconds", lambda _p: 0.0)
    assert run(_noise(2.0, 0.05)).text == "" and engine.calls == 0  # 무음에서 "감사합니다"를 지어내지 않는다
    monkeypatch.setattr(vad.silero_vad, "speech_seconds", lambda _p: 1.2)
    assert run(_noise(2.0, 0.05)).text == "내일 병원 예약해 줘"


def test_defaults_are_the_new_behaviour():
    s = get_settings()
    assert (s.companion_stt_engine, s.companion_vad, s.companion_confirm_mode) == ("turbo", "silero", "logprob")
    assert s.companion_stt_collapse_repeats is True and s.companion_confirm_logprob == -0.3
    assert s.companion_stt_api_engine == "turbo"  # 2026-11-11: 챗봇 받아쓰기도 turbo
