"""말벗 목소리 구글 엔진 (11/14): Gemini TTS(Agent Platform generateContent)로 문장을 mp3 로 만든다.

- 서비스 계정에 묶인 API 키를 x-goog-api-key 헤더로 보낸다 (주소에 싣지 않는다). 일반 API 키는 401 로 거절된다 (11/14 실측).
- 말투 지시는 speechMetadata.style 로 보낸다 - 본문에 섞으면 소리로 읽힐 수 있다. 3.8 은 언어를 자동 감지해 언어 칸이 없다.
- 응답은 WAV(16bit·24kHz·mono)라 기존 F1 과 같은 mp3(24kHz mono 48k)로 바꾼다 - BE·FE 계약 그대로.
- 리다이렉트를 따라가지 않는다 (키를 실은 요청이 다른 주소로 다시 나가지 않게). 타임아웃 필수.
- 실패는 GoogleTtsError(code) 하나로 낸다. code 는 고정 문자열이라 로그에 그대로 남겨도 된다 (문장·키·응답 본문 없음).
- 429·하루 글자 상한이면 COOLDOWN_SEC 동안 부르지 않는다 (그동안 cooldown). 글자 수는 보내기 전에 센다 (메모리, 재시작하면 0).
"""
from __future__ import annotations

import base64
import binascii
import json
import socket
import threading
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from typing import Callable

ENDPOINT = "https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent"
COOLDOWN_SEC = 10 * 60
MAX_RESPONSE_BYTES = 8 * 1024 * 1024  # WAV 수십 초 + base64 여유
_KST = timezone(timedelta(hours=9))


class GoogleTtsError(Exception):
    """구글 합성 실패. code: timeout | network | http_3xx | http_4xx | http_5xx | 429 | limit | cooldown | bad_response"""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001 - urllib 시그니처
        return None  # 따라가지 않는다 - 3xx 가 HTTPError 로 올라온다


def _kst_today() -> date:
    return datetime.now(_KST).date()


def _default_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(_NoRedirect())


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return True
    reason = getattr(exc, "reason", None)
    return isinstance(reason, (TimeoutError, socket.timeout))


class GoogleTts:
    """문장 -> mp3 bytes (tts_service.Synth 와 같은 모양). 상태(하루 글자 수·멈춤)를 가진다 - 프로세스에 하나만 만든다."""

    def __init__(self, api_key: str, model: str, voice: str, style: str, timeout_sec: float, daily_char_limit: int,
                 encode: Callable[[bytes], bytes] | None = None,
                 opener: urllib.request.OpenerDirector | None = None,
                 clock: Callable[[], float] = time.monotonic, today: Callable[[], date] = _kst_today) -> None:
        if not api_key:
            raise ValueError("api key missing")
        self._api_key = api_key
        self._url = ENDPOINT.format(model=model)
        self._voice = voice
        self._style = style
        self._timeout = timeout_sec
        self._limit = daily_char_limit
        if encode is None:
            from app.services.companion.tts_service import encode_wav_mp3
            encode = encode_wav_mp3
        self._encode = encode
        self._opener = opener or _default_opener()
        self._clock = clock
        self._today = today
        self._lock = threading.Lock()
        self._day: date | None = None
        self._used = 0
        self._paused_until = 0.0

    def __repr__(self) -> str:  # 키가 어디에도 찍히지 않게
        return "GoogleTts()"

    @property
    def used_today(self) -> int:
        return self._used

    def paused(self) -> bool:
        return self._clock() < self._paused_until

    def _pause(self) -> None:
        self._paused_until = self._clock() + COOLDOWN_SEC

    def _reserve(self, chars: int) -> None:
        with self._lock:
            if self.paused():
                raise GoogleTtsError("cooldown")
            day = self._today()
            if day != self._day:
                self._day, self._used = day, 0
            if self._limit > 0 and self._used + chars > self._limit:
                self._pause()
                raise GoogleTtsError("limit")
            self._used += chars

    def _body(self, text: str) -> bytes:
        part: dict = {"text": text}
        if self._style:
            part["speechMetadata"] = {"style": self._style}
        return json.dumps({
            "contents": [{"role": "user", "parts": [part]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"voice": self._voice}}},
        }).encode("utf-8")

    def __call__(self, text: str) -> bytes:
        self._reserve(len(text))
        req = urllib.request.Request(self._url, data=self._body(text), method="POST",
                                     headers={"Content-Type": "application/json", "x-goog-api-key": self._api_key})
        try:
            with self._opener.open(req, timeout=self._timeout) as resp:
                raw = resp.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as exc:
            status = int(exc.code or 0)
            exc.close()
            if status == 429:
                with self._lock:
                    self._pause()
                raise GoogleTtsError("429") from None
            raise GoogleTtsError(f"http_{status // 100}xx") from None
        except Exception as exc:  # noqa: BLE001 - URLError·소켓 오류. 원문은 남기지 않는다
            raise GoogleTtsError("timeout" if _is_timeout(exc) else "network") from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise GoogleTtsError("bad_response")
        try:
            data = json.loads(raw)
            inline = data["candidates"][0]["content"]["parts"][0]["inlineData"]
            wav = base64.b64decode(inline["data"], validate=True)
        except (ValueError, KeyError, IndexError, TypeError, binascii.Error):
            raise GoogleTtsError("bad_response") from None
        if wav[:4] != b"RIFF":
            raise GoogleTtsError("bad_response")
        try:
            return self._encode(wav)
        except Exception:  # noqa: BLE001 - ffmpeg 실패
            raise GoogleTtsError("bad_response") from None


def load_google() -> GoogleTts:
    from app.core.config import get_settings

    s = get_settings()
    return GoogleTts(api_key=s.companion_tts_google_api_key, model=s.companion_tts_google_model,
                     voice=s.companion_tts_google_voice, style=s.companion_tts_google_style,
                     timeout_sec=s.companion_tts_google_timeout_sec,
                     daily_char_limit=s.companion_tts_google_daily_char_limit)
