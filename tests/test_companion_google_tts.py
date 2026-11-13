"""말벗 목소리 구글 엔진 + F1 자동 전환 (11/14). 실제 구글·Supertonic 은 부르지 않는다 (가짜 opener·가짜 합성기)."""
from __future__ import annotations

import base64
import io
import json
import logging
import socket
import urllib.error
from datetime import date

import pytest

from app.core.config import get_settings
from app.services.companion import tts_service as tts_module
from app.services.companion.google_tts import COOLDOWN_SEC, GoogleTts, GoogleTtsError, _NoRedirect
from app.services.companion.tts_service import TtsService

SECRET_KEY = "AIzaSy-SECRET-test-key-value"
SECRET_TEXT = "비밀스러운어르신대답을되짚는문장"
STYLE = "어르신을 대하는 친절한 상담사처럼 따뜻하고 차분하게"
WAV = b"RIFF" + b"\x00" * 40


def _ok_body(wav: bytes = WAV) -> bytes:
    return json.dumps({"candidates": [{"content": {"parts": [
        {"inlineData": {"mimeType": "audio/wav", "data": base64.b64encode(wav).decode()}}]}}]}).encode()


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class FakeOpener:
    """open() 이 부를 때마다 outcomes 를 차례로 쓴다: bytes 면 응답 본문, int 면 그 상태 코드 HTTPError, 예외면 그대로 낸다."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests = []
        self.timeouts = []

    def open(self, req, timeout=None):
        self.requests.append(req)
        self.timeouts.append(timeout)
        out = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(out, BaseException):
            raise out
        if isinstance(out, int):
            raise urllib.error.HTTPError(req.full_url, out, "err", {}, io.BytesIO(b'{"error":{"message":"' +
                                                                                SECRET_TEXT.encode() + b'"}}'))
        return _Resp(out)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _google(opener, limit=50000, clock=None, today=None):
    return GoogleTts(api_key=SECRET_KEY, model="gemini-3.8-flash-tts", voice="Kore", style=STYLE, timeout_sec=8.0,
                     daily_char_limit=limit, encode=lambda wav: b"ID3" + wav[:4], opener=opener,
                     clock=clock or Clock(), today=today or (lambda: date(2026, 11, 14)))


# ---- 구글 엔진 단위 ----
def test_request_shape_and_key_only_in_header():
    op = FakeOpener(_ok_body())
    out = _google(op)("안녕하세요")
    assert out == b"ID3RIFF"  # WAV 를 mp3 변환기에 넘긴다
    req = op.requests[0]
    assert req.full_url.endswith("/publishers/google/models/gemini-3.8-flash-tts:generateContent")
    assert SECRET_KEY not in req.full_url
    assert req.get_header("X-goog-api-key") == SECRET_KEY
    body = json.loads(req.data)
    part = body["contents"][0]["parts"][0]
    assert part["text"] == "안녕하세요"  # 말투 지시는 본문에 섞지 않는다 (소리로 읽히지 않게)
    assert part["speechMetadata"]["style"] == STYLE
    assert body["generationConfig"]["speechConfig"]["voiceConfig"]["voice"] == "Kore"
    assert body["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert op.timeouts == [8.0]


@pytest.mark.parametrize("status,code", [(400, "http_4xx"), (403, "http_4xx"), (500, "http_5xx"), (503, "http_5xx"),
                                         (302, "http_3xx")])
def test_http_errors_map_to_fixed_codes(status, code):
    with pytest.raises(GoogleTtsError) as ei:
        _google(FakeOpener(status))("문장")
    assert ei.value.code == code
    assert SECRET_TEXT not in str(ei.value) and SECRET_KEY not in str(ei.value)


def test_redirect_is_not_followed():
    assert _NoRedirect().redirect_request(None, None, 302, "Found", {}, "https://evil.example/") is None


@pytest.mark.parametrize("exc,code", [(socket.timeout("timed out"), "timeout"),
                                      (urllib.error.URLError(socket.timeout("x")), "timeout"),
                                      (urllib.error.URLError(OSError("refused")), "network")])
def test_network_errors(exc, code):
    with pytest.raises(GoogleTtsError) as ei:
        _google(FakeOpener(exc))("문장")
    assert ei.value.code == code


def test_429_pauses_then_resumes_after_cooldown():
    clock = Clock()
    op = FakeOpener(429, _ok_body())
    g = _google(op, clock=clock)
    with pytest.raises(GoogleTtsError) as ei:
        g("문장")
    assert ei.value.code == "429"
    with pytest.raises(GoogleTtsError) as ei:
        g("문장")
    assert ei.value.code == "cooldown"
    assert len(op.requests) == 1  # 멈춘 동안은 부르지 않는다
    clock.t += COOLDOWN_SEC + 1
    assert g("문장") == b"ID3RIFF"


def test_daily_char_limit_pauses_and_resets_next_day():
    clock = Clock()
    day = [date(2026, 11, 14)]
    op = FakeOpener(_ok_body())
    g = _google(op, limit=10, clock=clock, today=lambda: day[0])
    g("일이삼사오육")  # 6자
    with pytest.raises(GoogleTtsError) as ei:
        g("일이삼사오육")  # 12자 > 10
    assert ei.value.code == "limit"
    assert len(op.requests) == 1
    clock.t += COOLDOWN_SEC + 1
    day[0] = date(2026, 11, 15)
    g("일이삼사오육")
    assert g.used_today == 6


def test_zero_limit_means_unlimited():
    g = _google(FakeOpener(_ok_body()), limit=0)
    for _ in range(3):
        g("가" * 1000)


@pytest.mark.parametrize("body", [b"not json", b"{}", _ok_body(b"ID3notwav")])
def test_bad_response(body):
    with pytest.raises(GoogleTtsError) as ei:
        _google(FakeOpener(body))("문장")
    assert ei.value.code == "bad_response"


def test_repr_and_missing_key():
    assert SECRET_KEY not in repr(_google(FakeOpener(_ok_body())))
    with pytest.raises(ValueError):
        GoogleTts(api_key="", model="m", voice="Kore", style="", timeout_sec=8, daily_char_limit=0)


def test_settings_repr_hides_key():
    s = get_settings().model_copy(update={"companion_tts_google_api_key": SECRET_KEY})
    assert SECRET_KEY not in repr(s)


# ---- TtsService 전환 ----
class Synth:
    def __init__(self, tag: bytes, fail_code: str | None = None):
        self.tag = tag
        self.fail_code = fail_code
        self.calls: list[str] = []

    def __call__(self, text: str) -> bytes:
        self.calls.append(text)
        if self.fail_code:
            raise GoogleTtsError(self.fail_code)
        return self.tag + text.encode()


@pytest.fixture
def setup(monkeypatch):
    def _make(engine="google", key=SECRET_KEY, google_fail=None):
        s = get_settings().model_copy(update={"companion_tts_enabled": True, "companion_tts_engine": engine,
                                              "companion_tts_google_api_key": key})
        monkeypatch.setattr(tts_module, "get_settings", lambda: s)
        f1, g = Synth(b"F1:"), Synth(b"GG:", google_fail)
        loads = []

        def g_loader():
            loads.append(1)
            return g

        svc = TtsService(loader=lambda: f1, google_loader=g_loader, sweep_interval_sec=3600)
        return svc, f1, g, loads
    return _make


def _audio(svc, sid, idx, text, cache_text=False):
    assert svc.start("w1", sid, idx, text, cache_text=cache_text)
    return svc.get(sid, idx, 5)


def test_supertonic_engine_never_touches_google(setup):
    svc, f1, g, loads = setup(engine="supertonic")
    assert _audio(svc, "s1", 0, "문장") == b"F1:" + "문장".encode()
    assert loads == [] and g.calls == []


def test_google_engine_ok(setup):
    svc, f1, g, _ = setup()
    assert _audio(svc, "s1", 0, "문장") == b"GG:" + "문장".encode()
    assert f1.calls == []


@pytest.mark.parametrize("code", ["timeout", "429", "http_5xx", "http_4xx", "limit", "cooldown"])
def test_failure_falls_back_to_f1_and_session_stays_f1(setup, code):
    svc, f1, g, _ = setup(google_fail=code)
    assert _audio(svc, "s1", 0, "첫 문장").startswith(b"F1:")
    g.fail_code = None  # 구글이 살아나도
    assert _audio(svc, "s1", 1, "둘째 문장").startswith(b"F1:")  # 이 대화는 끝까지 F1
    assert g.calls == ["첫 문장"]
    assert _audio(svc, "s2", 0, "다른 대화").startswith(b"GG:")  # 다른 대화는 구글


def test_missing_key_uses_f1_and_warns_once(setup, caplog):
    svc, f1, g, loads = setup(key="")
    with caplog.at_level(logging.WARNING, logger="companion.tts"):
        _audio(svc, "s1", 0, "하나")
        _audio(svc, "s1", 1, "둘")
    assert loads == [] and len(f1.calls) == 2
    assert sum("key=missing" in r.getMessage() for r in caplog.records) == 1


def test_fixed_cache_is_per_engine(setup):
    svc, f1, g, _ = setup()
    assert svc.prewarm(("고정 문장",)) == 2  # 구글 + F1(전환용)
    assert g.calls == ["고정 문장"] and f1.calls == ["고정 문장"]
    assert _audio(svc, "s1", 0, "고정 문장").startswith(b"GG:")
    svc._pin("s2")
    assert _audio(svc, "s2", 0, "고정 문장").startswith(b"F1:")  # F1 대화에는 F1 고정 소리
    assert len(g.calls) == 1 and len(f1.calls) == 1  # 둘 다 다시 합성하지 않는다


def test_prewarm_stops_google_on_limit(setup):
    svc, f1, g, _ = setup(google_fail="limit")
    svc.prewarm(("가", "나", "다"))
    assert g.calls == ["가"]  # 상한이면 나머지는 부르지 않는다
    assert f1.calls == ["가", "나", "다"]


def test_f1_broken_google_still_works(setup):
    svc, f1, g, _ = setup()

    def boom():
        raise RuntimeError("load")
    svc._loader = boom
    with pytest.raises(RuntimeError):
        svc._engine()
    assert svc.enabled()
    assert _audio(svc, "s1", 0, "문장").startswith(b"GG:")
    svc._pin("s2")
    assert svc.start("w1", "s2", 0, "문장") is False  # F1 대화인데 F1 이 없으면 소리 없음


def test_logs_have_no_text_or_key(setup, caplog):
    svc, f1, g, _ = setup(google_fail="429")
    with caplog.at_level(logging.DEBUG):
        _audio(svc, "s1", 0, SECRET_TEXT)
        svc.prewarm((SECRET_TEXT + "고정",))
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "engine=google" in text and "result=429 -> f1" in text and "engine=f1" in text
    assert SECRET_TEXT not in text and SECRET_KEY not in text


def test_real_google_errors_logged_without_body(monkeypatch, caplog):
    """실제 GoogleTts(가짜 opener)를 끼워 HTTP 오류 본문(문장이 섞일 수 있음)이 로그에 남지 않음을 본다."""
    s = get_settings().model_copy(update={"companion_tts_enabled": True, "companion_tts_engine": "google",
                                          "companion_tts_google_api_key": SECRET_KEY})
    monkeypatch.setattr(tts_module, "get_settings", lambda: s)
    g = _google(FakeOpener(500))
    svc = TtsService(loader=lambda: Synth(b"F1:"), google_loader=lambda: g, sweep_interval_sec=3600)
    with caplog.at_level(logging.DEBUG):
        assert _audio(svc, "s1", 0, SECRET_TEXT).startswith(b"F1:")
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "result=http_5xx -> f1" in text
    assert SECRET_TEXT not in text and SECRET_KEY not in text
