"""말벗 목소리 (11/13). 실제 Supertonic 은 띄우지 않는다 - 가짜 합성기로 바꿔 끼운다 (메모리 초과로 컨테이너가 죽은 적 있음)."""
from __future__ import annotations

import logging
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.routers import companion_router
from app.routers.companion_router import get_llm, get_text_classifier
from app.services.companion import tts_service as tts_module
from app.services.companion.llm_client import LlmUnavailable
from app.services.companion.text_emotion_service import TextEmotionResult
from app.services.companion.tts_service import TtsNotFound, TtsNotReady, TtsService, fixed_sentences
from scripts.companion.dev_app import app

KEY = "test-companion-key"
HDR = {"X-API-Key": KEY}
QUESTION = "잘 주무셨어요?"
SECRET_WORDS = "비밀스러운어르신대답"


class FakeSynth:
    """문장을 받아 가짜 mp3 바이트를 돌려준다. gate 가 있으면 열릴 때까지 기다린다."""

    def __init__(self, gate: threading.Event | None = None, fail: bool = False, size: int | None = None):
        self.gate = gate
        self.fail = fail
        self.size = size
        self.calls: list[str] = []
        self.lock = threading.Lock()
        self.running = 0
        self.peak = 0

    def __call__(self, text: str) -> bytes:
        with self.lock:
            self.calls.append(text)
            self.running += 1
            self.peak = max(self.peak, self.running)
        try:
            if self.gate is not None:
                assert self.gate.wait(5)
            if self.fail:
                raise RuntimeError(text)  # 예외 메시지에 원문이 있어도 로그에는 클래스명만 남아야 한다
            return b"ID3" + (b"\x00" * (self.size or len(text)))
        finally:
            with self.lock:
                self.running -= 1


@pytest.fixture
def enable(monkeypatch):
    def _set(on: bool = True):
        s = get_settings().model_copy(update={"companion_tts_enabled": on, "companion_api_key": KEY,
                                              "companion_backend_base_url": ""})
        monkeypatch.setattr(tts_module, "get_settings", lambda: s)
        return s
    return _set


def _svc(synth, **kw) -> TtsService:
    return TtsService(loader=lambda: synth, **kw)


# ---- 서비스 ----

def test_start_then_get_returns_audio(enable):
    enable()
    svc = _svc(FakeSynth())
    assert svc.start("w1", "s1", 1, "안녕하세요")
    assert svc.get("s1", 1, 2).startswith(b"ID3")


def test_disabled_does_not_synthesize(enable):
    enable(False)
    synth = FakeSynth()
    svc = _svc(synth)
    assert svc.start("w1", "s1", 1, "안녕하세요") is False
    with pytest.raises(TtsNotFound):
        svc.get("s1", 1, 0)
    assert synth.calls == []


def test_not_ready_then_ready(enable):
    enable()
    gate = threading.Event()
    svc = _svc(FakeSynth(gate=gate))
    svc.start("w1", "s1", 1, "안녕하세요")
    with pytest.raises(TtsNotReady):
        svc.get("s1", 1, 0.1)
    gate.set()
    assert svc.get("s1", 1, 2)


def test_failure_and_too_large_are_not_found(enable):
    enable()
    svc = _svc(FakeSynth(fail=True))
    svc.start("w1", "s1", 1, "안녕하세요")
    with pytest.raises(TtsNotFound):
        svc.get("s1", 1, 2)
    big = _svc(FakeSynth(size=2000), max_bytes=1000)
    big.start("w1", "s1", 1, "안녕하세요")
    with pytest.raises(TtsNotFound):
        big.get("s1", 1, 2)


def test_load_failure_turns_off_until_restart(enable):
    enable()
    calls = []

    def bad_loader():
        calls.append(1)
        raise FileNotFoundError("/app/models/companion/tts-supertonic3")

    svc = TtsService(loader=bad_loader)
    assert svc.start("w1", "s1", 1, "안녕하세요")
    with pytest.raises(TtsNotFound):
        svc.get("s1", 1, 2)
    assert svc.start("w1", "s1", 2, "안녕하세요") is False  # 다시 올리지 않는다
    assert len(calls) == 1


def test_expired_after_ttl(enable, monkeypatch):
    enable()
    svc = _svc(FakeSynth(), ttl_sec=60)
    svc.start("w1", "s1", 1, "안녕하세요")
    svc.get("s1", 1, 2)
    now = time.monotonic()
    monkeypatch.setattr(tts_module.time, "monotonic", lambda: now + 61)
    with pytest.raises(TtsNotFound):
        svc.get("s1", 1, 0)


def test_max_items_drops_oldest(enable):
    enable()
    svc = _svc(FakeSynth(), max_items=2)
    for i in range(3):
        svc.start("w1", "s1", i, f"문장{i}")
        svc.get("s1", i, 2)
    with pytest.raises(TtsNotFound):
        svc.get("s1", 0, 0)
    assert svc.get("s1", 2, 0)


def test_concurrency_limit_and_skip_when_full(enable):
    enable()
    gate = threading.Event()
    synth = FakeSynth(gate=gate)
    svc = _svc(synth, workers=2, max_pending=3)
    started = [svc.start("w1", "s1", i, f"문장{i}") for i in range(6)]
    assert started == [True] * 5 + [False]  # 동시 2 + 대기 3 까지만, 넘치면 포기
    time.sleep(0.2)
    assert synth.peak == 2
    gate.set()
    for i in range(5):
        assert svc.get("s1", i, 3)
    assert synth.peak == 2
    assert svc.start("w1", "s1", 9, "문장9")  # 자리가 나면 다시 받는다


def test_fixed_sentence_reused_without_synthesis(enable):
    enable()
    synth = FakeSynth()
    svc = _svc(synth)
    texts = fixed_sentences()
    assert svc.prewarm(texts) == len(texts)
    n = len(synth.calls)
    assert svc.start("w1", "s1", 3, texts[0])
    assert svc.get("s1", 3, 0)  # 이미 만들어 둔 소리라 기다리지 않는다
    assert len(synth.calls) == n


def test_fixed_sentences_cover_ai_fixed_replies():
    from app.services.companion import reply_service
    from app.services.companion.pipeline import LOW_CONFIDENCE_REPLY

    texts = fixed_sentences()
    for t in (*reply_service.FALLBACK_LAST, reply_service.SAFETY_REPLY, reply_service.BODY_SAFETY_REPLY,
              LOW_CONFIDENCE_REPLY, *reply_service.FALLBACK_NO_RESPONSE, *reply_service.FALLBACK_AVOID):
        assert t in texts
    assert len(texts) == len(set(texts))


def test_question_cached_by_text_but_model_reply_is_not(enable):
    enable()
    synth = FakeSynth()
    svc = _svc(synth)
    svc.start("w1", "s1", 0, QUESTION, cache_text=True)
    svc.get("s1", 0, 2)
    svc.start("w2", "s2", 0, QUESTION, cache_text=True)  # 다른 대화의 같은 질문
    svc.get("s2", 0, 0)
    svc.start("w1", "s1", 1, SECRET_WORDS)
    svc.get("s1", 1, 2)
    svc.start("w2", "s2", 1, SECRET_WORDS)
    svc.get("s2", 1, 2)
    assert synth.calls.count(QUESTION) == 1
    assert synth.calls.count(SECRET_WORDS) == 2  # 모델 대답은 문장 기준 보관에 넣지 않는다


def test_delete_ward_removes_only_that_ward(enable):
    enable()
    svc = _svc(FakeSynth())
    svc.start("w1", "s1", 1, "가")
    svc.start("w2", "s2", 1, "나")
    svc.get("s1", 1, 2), svc.get("s2", 1, 2)
    assert svc.delete_ward("w1") == 1
    with pytest.raises(TtsNotFound):
        svc.get("s1", 1, 0)
    assert svc.get("s2", 1, 0)


def test_logs_have_no_text(enable, caplog):
    enable()
    caplog.set_level(logging.INFO, logger="companion.tts")
    svc = _svc(FakeSynth())
    svc.start("w1", "secret-session-77", 1, SECRET_WORDS)
    svc.get("secret-session-77", 1, 2)
    bad = _svc(FakeSynth(fail=True))
    bad.start("w1", "secret-session-77", 2, SECRET_WORDS)
    with pytest.raises(TtsNotFound):
        bad.get("secret-session-77", 2, 2)
    text = caplog.text
    assert "result=ok" in text and "fail_RuntimeError" in text
    assert SECRET_WORDS not in text and "secret-session-77" not in text


# ---- 경로 ----

class _DownLlm:
    """연결이 안 되는 가짜 LLM - 대체 문장으로 답하게 한다 (모델을 띄우지 않는다)."""

    def generate(self, prompt, **kw):
        raise LlmUnavailable("down")


@pytest.fixture
def client(enable, monkeypatch):
    enable()
    synth = FakeSynth()
    svc = _svc(synth)
    monkeypatch.setattr(companion_router, "tts_service", svc)
    settings = get_settings().model_copy(update={"companion_api_key": KEY, "companion_backend_base_url": ""})
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_text_classifier] = lambda: (lambda t: TextEmotionResult("기쁨", 0.9, True, {}))
    app.dependency_overrides[get_llm] = lambda: _DownLlm()
    yield TestClient(app), svc, synth
    app.dependency_overrides.clear()


def _open(c, session_id="s-tts"):
    body = {"wardId": "u1", "sessionId": session_id, "slot": "09:00", "question": QUESTION, "questionId": "q1"}
    return c.post("/companion/dialogues", json=body, headers=HDR)


def _audio(c, session_id, index, wait=2):
    return c.get(f"/companion/sessions/{session_id}/utterances/{index}/audio?wait={wait}", headers=HDR)


def test_open_and_turn_carry_audio_index_and_serve_mp3(client):
    c, _svc_, synth = client
    opened = _open(c).json()["data"]
    assert opened["audio"] == {"index": 0}
    r = _audio(c, "s-tts", 0)
    assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg"
    assert r.headers["cache-control"] == "no-store"
    turn = c.post(f"/companion/dialogues/{opened['dialogueId']}/reply", json={"text": "잘 잤어요"}, headers=HDR)
    data = turn.json()["data"]
    assert data["audio"] == {"index": data["turnNo"]} == {"index": 1}
    assert _audio(c, "s-tts", 1).status_code == 200
    assert synth.calls[-1] == data["reply"]["text"]  # 최종(반복 막기 뒤) 문장으로 만든다


def test_audio_is_null_when_disabled(client, enable):
    c, _, synth = client
    enable(False)
    opened = _open(c).json()["data"]
    assert opened["audio"] is None
    turn = c.post(f"/companion/dialogues/{opened['dialogueId']}/reply", json={"text": "잘 잤어요"}, headers=HDR)
    assert turn.json()["data"]["audio"] is None
    r = _audio(c, "s-tts", 0, wait=0)
    assert r.status_code == 404 and r.json()["code"] == "TTS_NOT_FOUND"
    assert synth.calls == []


def test_route_404_503_and_wait_validation(client, monkeypatch):
    c, svc, _ = client
    assert _audio(c, "nope", 0, wait=0).json()["code"] == "TTS_NOT_FOUND"
    gate = threading.Event()
    slow = TtsService(loader=lambda: FakeSynth(gate=gate))
    monkeypatch.setattr(companion_router, "tts_service", slow)
    _open(c, "s-slow")
    r = _audio(c, "s-slow", 0, wait=0.2)
    assert r.status_code == 503 and r.json()["code"] == "TTS_NOT_READY"
    gate.set()
    assert _audio(c, "s-slow", 0, wait=3).status_code == 200
    assert _audio(c, "s-slow", 0, wait=9).status_code == 422
    assert _audio(c, "s-slow", 0, wait=-1).status_code == 422
    assert _audio(c, "s-slow", 101, wait=0).status_code == 404


def test_owner_header_must_match(client):
    """BE 점검 L-2(11/13): X-Companion-Ward-Id 가 있으면 대화 주인과 같을 때만 준다. 없으면 옛 BE 호환으로 그대로 준다."""
    c, _, _ = client
    _open(c)  # wardId u1
    ok = c.get("/companion/sessions/s-tts/utterances/0/audio?wait=2", headers={**HDR, "X-Companion-Ward-Id": "u1"})
    assert ok.status_code == 200
    other = c.get("/companion/sessions/s-tts/utterances/0/audio?wait=2", headers={**HDR, "X-Companion-Ward-Id": "u2"})
    assert other.status_code == 404 and other.json()["code"] == "TTS_NOT_FOUND"
    assert _audio(c, "s-tts", 0).status_code == 200


def test_owner_mismatch_does_not_wait(enable):
    """남의 소리는 합성 중이어도 기다리지 않고 바로 없음 - 응답 시간으로 있는지 드러나지 않게."""
    enable()
    gate = threading.Event()
    svc = _svc(FakeSynth(gate=gate))
    assert svc.start("w1", "s1", 0, "안녕하세요")
    started = time.monotonic()
    with pytest.raises(TtsNotFound):
        svc.get("s1", 0, 3, ward_id="w2")
    assert time.monotonic() - started < 1
    with pytest.raises(TtsNotReady):
        svc.get("s1", 0, 0.1, ward_id="w1")
    gate.set()
    assert svc.get("s1", 0, 3, ward_id="w1").startswith(b"ID3")


def test_audio_route_requires_key(client):
    c, _, _ = client
    _open(c)
    assert c.get("/companion/sessions/s-tts/utterances/0/audio").status_code == 401


def test_delete_ward_data_drops_audio(client):
    c, _, _ = client
    _open(c)
    assert _audio(c, "s-tts", 0).status_code == 200
    out = c.delete("/companion/wards/u1/data", headers=HDR).json()["data"]
    assert set(out) == {"wardId", "dialogues", "reviewItems"}  # 응답 형식은 그대로
    assert _audio(c, "s-tts", 0, wait=0).status_code == 404


# ---- 11/13 기능 점검 반영 (M-1·L-1·L-2·L-3) ----

def _wait_until(cond, sec=3.0):
    end = time.monotonic() + sec
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def test_sweeper_removes_expired_without_requests(enable):
    """M-1: 요청이 없어도 주기 정리가 만료 소리를 지운다."""
    enable()
    svc = _svc(FakeSynth(), ttl_sec=0.2, sweep_interval_sec=0.05)
    svc.start("w1", "s1", 1, "안녕하세요")
    svc.get("s1", 1, 2)
    assert _wait_until(lambda: len(svc._items) == 0)


def test_single_executor_under_simultaneous_first_use(enable):
    """L-1: 첫 사용이 동시에 몰려도 실행기는 하나라 동시 합성은 2건을 넘지 않는다."""
    enable()
    gate = threading.Event()
    synth = FakeSynth(gate=gate)
    svc = _svc(synth, workers=2, max_pending=20)
    barrier = threading.Barrier(8)

    def go(i):
        barrier.wait()
        svc.start("w1", "s1", i, f"문장{i}")
    threads = [threading.Thread(target=go, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert _wait_until(lambda: synth.peak == 2)
    time.sleep(0.1)
    assert synth.peak == 2
    gate.set()
    for i in range(8):
        assert svc.get("s1", i, 3)
    assert svc._pool() is svc._pool()


def test_load_failure_is_not_retried_by_queued_jobs(enable):
    """L-2: 적재가 실패하면 대기열에 있던 작업이 저마다 다시 올리지 않는다."""
    enable()
    gate = threading.Event()
    calls = []

    def slow_bad_loader():
        calls.append(1)
        assert gate.wait(5)
        raise MemoryError()

    svc = TtsService(loader=slow_bad_loader, workers=2, max_pending=8)
    for i in range(5):
        assert svc.start("w1", "s1", i, f"문장{i}")
    gate.set()
    for i in range(5):
        with pytest.raises(TtsNotFound):
            svc.get("s1", i, 3)
    assert len(calls) == 1


def test_prewarm_counts_toward_queue_limit(enable):
    """L-3: 예열 중인 합성도 대기 개수에 센다."""
    enable()
    gate = threading.Event()
    svc = _svc(FakeSynth(gate=gate), workers=2, max_pending=1)  # 합쳐서 3건까지
    t = threading.Thread(target=svc.prewarm, args=(("고정 문장",),))
    t.start()
    assert _wait_until(lambda: svc._inflight == 1)
    assert [svc.start("w1", "s1", i, f"문장{i}") for i in range(3)] == [True, True, False]
    gate.set()
    t.join(5)
    assert svc._inflight == 0 or _wait_until(lambda: svc._inflight == 0)
