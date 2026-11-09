"""11/09 말벗 음성 단계 기능 점검(docs/audits/2026-11-09-companion-voice-stage-audit.md) 반영 고정.

H-1 되묻는 문장 반복 유지 · H-2 안전 문장 반복 유지 · M-1 토큰 상한·조용한 곳에서 나누기 · M-2 연속 되묻기 종료
L-1 스위치 값 정규화 · L-2 실행 중 예외 503 · L-3 로드 실패 재시도. 모델·LLM은 모두 가짜.
"""

from __future__ import annotations

import io
import logging
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core import config
from app.core.config import get_settings
from app.routers.companion_router import get_llm, get_pcm_transcriber, get_text_classifier, get_transcriber
from app.services.companion import turbo_service as turbo_mod
from app.services.companion.llm_client import LlmUnavailable
from app.services.companion.pipeline import LOW_CONFIDENCE_REPLY
from app.services.companion.reply_service import FALLBACK_LAST, SAFETY_REPLY
from app.services.companion.stt_service import SttResult, SttUnavailable
from app.services.companion.text_emotion_service import TextEmotionResult
from app.services.companion.turbo_service import WINDOW_SEC, TurboService, split_points
from scripts.companion.dev_app import app

KEY = "test-companion-key"
HDR = {"X-API-Key": KEY}
SR = 16000


def _wav(seconds: float = 1.0) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.zeros(int(SR * seconds), dtype=np.int16).tobytes())
    return buf.getvalue()


def _multipart(data: bytes):
    boundary = "----auditfix"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"a.wav\"\r\n"
            "Content-Type: audio/wav\r\n\r\n").encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return body, {**HDR, "content-type": f"multipart/form-data; boundary={boundary}"}


class ProbLlm:
    def __init__(self, p: float):
        self.p = p

    def generate(self, *_a, **_k):
        raise LlmUnavailable("off")

    def next_token_probs(self, *_a, **_k):
        return {"있": self.p, "없": 1 - self.p}


@pytest.fixture
def s(monkeypatch, tmp_path):
    st = get_settings()
    monkeypatch.setattr(st, "companion_api_key", KEY)
    monkeypatch.setattr(st, "companion_backend_base_url", "")
    monkeypatch.setattr(st, "companion_confirm_mode", "logprob")
    monkeypatch.setattr(st, "companion_keep_fixed_replies", True)
    monkeypatch.setattr(st, "companion_confirm_streak_limit", 3)
    monkeypatch.setattr(st, "companion_risk_prompt", st.companion_risk_prompt)  # 테스트에서 바꿔도 끝나면 되돌린다
    yield st
    app.dependency_overrides.clear()


def _dialogue(turns, risk_p=0.0, transcribe=None):
    """turns: (글, 로그확률) 목록. 한 대화에서 차례로 음성 턴을 보내고 응답 data 목록을 돌려준다."""
    it = iter(turns)
    app.dependency_overrides[get_transcriber] = lambda: transcribe or (lambda _a: _next(it))
    app.dependency_overrides[get_text_classifier] = lambda: (lambda t: TextEmotionResult("기쁨", 0.9, True, {}))
    app.dependency_overrides[get_llm] = lambda: ProbLlm(risk_p)
    c = TestClient(app)
    body = {"wardId": "u-audit", "sessionId": "s-audit", "slot": "09:00", "question": "잘 주무셨어요?", "questionId": "q-audit"}
    did = c.post("/companion/dialogues", json=body, headers=HDR).json()["data"]["dialogueId"]
    out = []
    for _ in turns:
        content, headers = _multipart(_wav())
        r = c.post(f"/companion/dialogues/{did}/reply/audio", content=content, headers=headers)
        out.append(r)
    return out


def _next(it):
    text, lp = next(it)
    return SttResult(text, 0.5, 1, avg_logprob=lp)


def _data(r):
    assert r.status_code == 200, r.text
    return r.json()["data"]


# ---------- H-1 · H-2 ----------

def test_repeated_low_confidence_turns_keep_asking(s):
    s.companion_confirm_streak_limit = 0  # 종료 규칙 없이 반복만 본다
    out = [_data(r) for r in _dialogue([("국수를 먹었어요", -0.8)] * 3)]
    assert [d["reply"]["text"] for d in out] == [LOW_CONFIDENCE_REPLY] * 3


def test_repeated_risk_turns_keep_the_safety_reply(s):
    s.companion_risk_prompt = "v4"  # 이 가짜 모델은 재희 v4 확률 함수만 있다 (11/10 기본 v9 는 기준 0.99)
    out = [_data(r) for r in _dialogue([("요즘 다 귀찮고 그래요 정말", -0.1)] * 2, risk_p=0.9)]  # 일반 턴
    assert [d["reply"]["text"] for d in out] == [SAFETY_REPLY] * 2
    out = [_data(r) for r in _dialogue([("요즘 다 귀찮아요", -0.8)] * 2, risk_p=0.9)]  # 되묻는 턴
    assert [d["reply"]["text"] for d in out] == [SAFETY_REPLY] * 2
    assert all(d["riskFlag"] for d in out)


def test_keep_fixed_off_is_jaehee_replacement(s):
    s.companion_keep_fixed_replies = False
    s.companion_confirm_streak_limit = 0
    out = [_data(r) for r in _dialogue([("국수를 먹었어요", -0.8)] * 2)]
    assert out[0]["reply"]["text"] == LOW_CONFIDENCE_REPLY and out[1]["reply"]["text"] != LOW_CONFIDENCE_REPLY


# ---------- M-2 ----------

def test_three_low_confidence_turns_in_a_row_end_the_dialogue(s):
    out = [_data(r) for r in _dialogue([("웅성웅성", -0.9)] * 3)]
    assert [d["ended"] for d in out] == [False, False, True]
    assert out[2]["endReason"] == "NO_RESPONSE"
    assert out[2]["reply"]["text"] == FALLBACK_LAST[0]  # 끝나는 턴에는 다시 말해 달라고 하지 않는다


def test_a_clear_answer_resets_the_streak(s):
    out = [_data(r) for r in _dialogue([("웅성웅성", -0.9), ("웅성웅성", -0.9), ("국수를 먹었어요", -0.1),
                                        ("웅성웅성", -0.9), ("웅성웅성", -0.9)])]
    assert not any(d["ended"] for d in out)


def test_risky_low_confidence_turns_do_not_end_the_dialogue(s):
    s.companion_risk_prompt = "v4"  # 이 가짜 모델은 재희 v4 확률 함수만 있다 (11/10 기본 v9 는 기준 0.99)
    out = [_data(r) for r in _dialogue([("다 귀찮아요", -0.9)] * 3, risk_p=0.9)]
    assert not any(d["ended"] for d in out) and all(d["reply"]["text"] == SAFETY_REPLY for d in out)


def test_streak_limit_zero_is_jaehee(s):
    s.companion_confirm_streak_limit = 0
    out = [_data(r) for r in _dialogue([("웅성웅성", -0.9)] * 4)]
    assert not any(d["ended"] for d in out)


# ---------- M-1 ----------

def test_default_token_cap_fits_a_fast_thirty_second_answer():
    assert get_settings().companion_turbo_max_new_tokens == 224  # 30초 150음절 x 0.83토큰 ≈ 125 < 224 < 디코더 448


def test_split_cuts_at_the_quietest_point_near_the_window_end():
    pcm = (np.random.default_rng(1).standard_normal(SR * 70) * 0.1).astype(np.float32)
    pcm[SR * 27:SR * 27 + 640] = 0.0
    parts = split_points(pcm)
    assert parts[0][0] == 0 and SR * 27 <= parts[0][1] <= SR * 27 + 640
    assert all(b - a <= WINDOW_SEC * SR for a, b in parts)
    assert parts[-1][1] == pcm.size and all(parts[i][1] == parts[i + 1][0] for i in range(len(parts) - 1))
    assert split_points(np.zeros(SR * 30, dtype=np.float32)) == [(0, SR * 30)]


# ---------- L-1 ----------

@pytest.mark.parametrize("raw, expected", [("Silero", "silero"), (" RMS ", "rms"), ("", "silero"), ("webrtc", "silero")])
def test_switch_values_are_normalised(monkeypatch, caplog, raw, expected):
    monkeypatch.setenv("COMPANION_VAD", raw)
    with caplog.at_level(logging.WARNING, logger="app.config"):
        assert config._as_choice("COMPANION_VAD", "silero", ("silero", "rms")) == expected
    assert ("[CONFIG]" in caplog.text) == (raw == "webrtc")
    assert "webrtc" not in caplog.text  # 값 자체는 로그에 남기지 않는다


# ---------- L-2 ----------

def test_runtime_error_in_turn_stt_is_503(s):
    def boom(_a):
        raise RuntimeError("CUDA out of memory 세부 내용")

    r = _dialogue([("x", -0.1)], transcribe=boom)[0]
    assert r.status_code == 503 and r.json()["code"] == "STT_UNAVAILABLE"
    assert "CUDA" not in r.text


def test_runtime_error_in_stt_api_is_503(s):
    def boom(_p):
        raise RuntimeError("CUDA out of memory")

    app.dependency_overrides[get_pcm_transcriber] = lambda: boom
    content, headers = _multipart(_wav(1.0))
    r = TestClient(app).post("/companion/stt", content=content, headers=headers)
    assert r.status_code == 503 and r.json()["code"] == "STT_UNAVAILABLE" and "CUDA" not in r.text


# ---------- L-3 ----------

def test_turbo_load_failure_is_retried_after_the_wait(s, monkeypatch, tmp_path):
    monkeypatch.setattr(s, "model_base_path", str(tmp_path))
    monkeypatch.setattr(s, "companion_turbo_retry_sec", 60.0)
    now = [1000.0]
    monkeypatch.setattr(turbo_mod.time, "monotonic", lambda: now[0])
    svc = TurboService()
    with pytest.raises(SttUnavailable):
        svc.warm_up()
    checks = []
    monkeypatch.setattr(svc, "model_dir", lambda: checks.append(1) or tmp_path / "none")
    with pytest.raises(SttUnavailable):
        svc.warm_up()  # 60초 전: 폴더를 다시 보지 않는다
    assert checks == []
    now[0] += 61
    with pytest.raises(SttUnavailable):
        svc.warm_up()  # 60초 뒤: 다시 시도한다
    assert checks == [1]


def test_turbo_retry_zero_keeps_the_old_behaviour(s, monkeypatch, tmp_path):
    monkeypatch.setattr(s, "model_base_path", str(tmp_path))
    monkeypatch.setattr(s, "companion_turbo_retry_sec", 0.0)
    svc = TurboService()
    with pytest.raises(SttUnavailable):
        svc.warm_up()
    monkeypatch.setattr(turbo_mod.time, "monotonic", lambda: 10 ** 9)
    monkeypatch.setattr(svc, "model_dir", lambda: (_ for _ in ()).throw(AssertionError("다시 시도하면 안 된다")))
    with pytest.raises(SttUnavailable):
        svc.warm_up()
