"""라우터 실패 누적 알림·크레딧 소진 구분(2026-11-08, #187 점검 L-1 후속).

[CHAT-ROUTER-FAILED] 는 건마다 WARN 이라 크레딧 소진(11/04~05)처럼 실패가 이어져도 사람이 알아채지 못했다.
- 실패 종류를 고정 코드로 나눈다(failure_kind): credit(크레딧·결제) / auth(인증) / timeout(시간 초과) / other / off(키 없음).
- 첫 실패에서 시작하는 고정 창 안에서 실패가 기준 횟수에 닿으면 [CHAT-ROUTER-FAILED-BURST] ERROR 한 번(창마다 1회).
- 크레딧·결제 오류는 처음 생긴 순간 [CHAT-ROUTER-CREDIT-EXHAUSTED] ERROR 한 번, 이어지면 정해진 간격마다 1회(지금 키가 마지막 예산이라
  떨어지는 순간을 놓치지 않게). 라우터가 다시 성공하면 소진 상태를 푼다.
- 키가 비어 있는 것(off)은 장애가 아니라 설정이라 세지 않는다.
로그에는 종류·원인 코드와 개수만 - 질문·답변 본문·키·API 오류 문구는 남기지 않는다(문구는 판정에만 쓰고 버린다).
AI 서버는 uvicorn 단일 프로세스라 프로세스 안 상태로 충분하다(여러 프로세스로 늘리면 프로세스별로 센다).
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import Counter
from typing import Callable

from app.services.chat_deadline import ChatDeadlineExceeded

_LOG = logging.getLogger(__name__)
# 원인 코드는 고정 코드(max_tokens·parse·api_<상태>)나 예외 클래스명이다. 이 모양이 아니면 "other" 로만 남긴다.
_SAFE_CAUSE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,39}")
KINDS = ("credit", "auth", "timeout", "other")
# Anthropic SDK(2.x) 기준 크레딧·결제 신호: 본문 error.type, error.details.error_code. 11/04 실제 사례는 400 + "credit balance is too low"
# 문구(type 은 invalid_request_error)라 문구도 본다 - 판정에만 쓰고 로그·기록에는 남기지 않는다.
_CREDIT_TYPES = frozenset({"billing_error"})
_CREDIT_CODES = frozenset({"billing_frozen", "enforced_spend_limit_reached"})
# "billing" 낱말 하나만으로는 보지 않는다(2026-11-09 점검 L-2) - 다른 이유의 400·403 문구에 섞이면 #189 건너뛰기까지 30분 켜진다.
_CREDIT_TEXT = re.compile(r"credit balance|purchase credits|spend limit", re.IGNORECASE)


def _body_error(err: BaseException) -> dict:
    body = getattr(err, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    return error if isinstance(error, dict) else {}


def failure_kind(exc: BaseException) -> str:
    """라우터 실패 종류(고정 코드). failure_cause(원인 코드)를 넓힌 분류 - StructuredCallError 는 원래 SDK 예외를 __cause__ 로 단다."""
    cause = getattr(exc, "cause", None)
    if cause == "not_ready":
        return "off"  # ANTHROPIC_API_KEY 없음 - 클라이언트를 만들지 않았다
    if cause == "credit_exhausted":
        return "credit"  # #189 건너뛰기(skip_when_exhausted)로 생긴 실패 - 원래 SDK 예외가 없다(2026-11-09 점검 L-1)
    if cause == "deadline" or isinstance(exc, (ChatDeadlineExceeded, TimeoutError)):
        return "timeout"
    origin = exc.__cause__ or exc
    if isinstance(origin, (ChatDeadlineExceeded, TimeoutError)):
        return "timeout"
    try:
        import anthropic
    except Exception:  # noqa: BLE001 - SDK 가 없으면 상태 코드로 분류할 수 없다
        return "other"
    if isinstance(origin, anthropic.APITimeoutError):
        return "timeout"
    if isinstance(origin, anthropic.APIStatusError):
        status = getattr(origin, "status_code", None)
        error = _body_error(origin)
        err_type = getattr(origin, "type", None) or error.get("type")
        details = error.get("details") if isinstance(error.get("details"), dict) else {}
        err_code = getattr(origin, "error_code", None) or details.get("error_code")
        text = getattr(origin, "api_message", None) or error.get("message") or ""
        if (status == 402 or err_type in _CREDIT_TYPES or err_code in _CREDIT_CODES
                or (status in (400, 403) and isinstance(text, str) and _CREDIT_TEXT.search(text))):
            return "credit"
        if status in (401, 403) or err_type in ("authentication_error", "permission_error"):
            return "auth"
        if status in (408, 504) or err_type == "timeout_error":
            return "timeout"
    return "other"


class RouterFailureMonitor:
    def __init__(
        self, window_sec: int, threshold: int, credit_repeat_sec: int = 3600, clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._window_sec = window_sec
        self._threshold = threshold
        self._credit_repeat_sec = credit_repeat_sec
        self._clock = clock
        self._lock = threading.Lock()
        self._window_start: float | None = None
        self._causes: Counter[str] = Counter()
        self._alerted = False
        self._credit_active = False
        self._credit_alert_at: float | None = None

    def record(self, cause: str, kind: str = "other") -> bool:
        """실패 한 건을 센다. 이번 호출이 ERROR 를 남겼으면 True. kind=off(키 없음)는 세지 않는다."""
        if kind == "off":
            return False
        kind = kind if kind in KINDS else "other"
        code = cause if isinstance(cause, str) and _SAFE_CAUSE.fullmatch(cause) else "other"
        burst = credit = None
        with self._lock:
            now = self._clock()
            if kind == "credit":
                if not self._credit_active or self._credit_alert_at is None or now - self._credit_alert_at >= self._credit_repeat_sec:
                    credit = "first" if not self._credit_active else "repeat"
                    self._credit_alert_at = now
                self._credit_active = True
            if self._window_start is None or now - self._window_start >= self._window_sec:
                self._window_start = now
                self._causes = Counter()
                self._alerted = False
            self._causes[f"{kind}/{code}"] += 1
            count = sum(self._causes.values())
            # 크레딧 소진 중 창의 실패가 모두 credit 이면 CREDIT-EXHAUSTED 가 같은 사건을 이미 알렸다 - BURST 는 생략한다
            # (2026-11-09 점검 M-2: 1시간 지속에 ERROR 9줄). 생략한 창도 다른 종류가 섞이는 순간 알린다.
            credit_only = self._credit_active and all(name.startswith("credit/") for name in self._causes)
            if not self._alerted and count >= self._threshold and not credit_only:
                self._alerted = True
                burst = (count, ",".join(f"{name}:{n}" for name, n in self._causes.most_common()))
        if credit:
            _LOG.error("[CHAT-ROUTER-CREDIT-EXHAUSTED] alert=%s cause=%s fallback=keywords", credit, code)
        if burst:
            _LOG.error(
                "[CHAT-ROUTER-FAILED-BURST] count=%s window_sec=%s causes=%s fallback=keywords",
                burst[0], self._window_sec, burst[1],
            )
        return bool(credit or burst)

    def record_success(self) -> None:
        """라우터가 답을 냈다 - 크레딧 소진 상태를 푼다(충전·키 교체 뒤). 창의 실패 개수는 그대로 둔다."""
        with self._lock:
            if self._credit_active:
                self._credit_active = False
                self._credit_alert_at = None
                recovered = True
            else:
                recovered = False
        if recovered:
            _LOG.warning("[CHAT-ROUTER-CREDIT-RECOVERED]")

    def status(self, key_configured: bool) -> dict:
        """관리자 집계용 지금 상태: off(키 없음) / credit_exhausted / failing(지금 창의 실패 N회) / ok. 개수만."""
        with self._lock:
            now = self._clock()
            in_window = self._window_start is not None and now - self._window_start < self._window_sec
            recent = sum(self._causes.values()) if in_window else 0
            credit = self._credit_active
        if not key_configured:
            state = "off"
        elif credit:
            state = "credit_exhausted"
        elif recent:
            state = "failing"
        else:
            state = "ok"
        return {"state": state, "recentFailures": recent if key_configured else 0, "windowSec": self._window_sec}


_monitor: RouterFailureMonitor | None = None
_monitor_lock = threading.Lock()


def get_router_failure_monitor() -> RouterFailureMonitor:
    global _monitor
    if _monitor is None:
        with _monitor_lock:
            if _monitor is None:
                from app.core.config import get_settings

                settings = get_settings()
                _monitor = RouterFailureMonitor(
                    settings.chat_router_failure_alert_window_sec,
                    settings.chat_router_failure_alert_threshold,
                    settings.chat_router_credit_alert_repeat_sec,
                )
    return _monitor
