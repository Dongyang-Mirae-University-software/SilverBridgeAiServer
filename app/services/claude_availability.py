"""Anthropic 크레딧 소진 감지와 '한동안 호출 건너뛰기'(2026-11-08).

gosky 의 키는 마지막 예산이라 다 쓰면 다시 채우지 않는다. 소진 뒤에도 매 요청이 Claude 를 부르고 실패를 기다리지 않게,
크레딧 부족 응답을 한 번 보면 SKIP_SEC 동안 '건너뛰기를 켠 호출자'(예약 추출·이어서 물어볼 질문·약 봉투 인식)는
부르지 않고 바로 대체 동작으로 간다. 시간이 지나면 다음 요청 1건이 다시 시도하고, 또 소진이면 다시 건너뛴다.
라우터는 건너뛰기를 켜지 않는다(감지 기록만 공유) - 라우터 실패 처리는 chat_router_service 가 정한다.

판정은 400 응답 본문의 오류 문구만 보고, 그 문구는 어디에도 남기지 않는다(로그는 고정 코드 credit_exhausted 만).
"""

from __future__ import annotations

import logging
import os
import threading
import time

import anthropic

_LOG = logging.getLogger(__name__)

CREDIT_EXHAUSTED = "credit_exhausted"


def _skip_sec() -> float:
    try:
        value = float(os.getenv("CLAUDE_CREDIT_SKIP_SEC", "1800"))
    except ValueError:
        return 1800.0
    return min(max(value, 0.0), 86400.0)


SKIP_SEC = _skip_sec()
# Anthropic 크레딧 부족 오류 문구("Your credit balance is too low ...")의 일부. 대소문자 무시.
_CREDIT_MARKERS = ("credit balance",)

_lock = threading.Lock()
_skip_until = 0.0


def _error_message(exc: anthropic.APIStatusError) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("message"), str):
            return err["message"]
        if isinstance(body.get("message"), str):
            return body["message"]
    message = getattr(exc, "message", "")
    return message if isinstance(message, str) else ""


def is_credit_exhausted(exc: BaseException) -> bool:
    """크레딧·결제 부족 응답인가(400 또는 402). 문구는 판정에만 쓰고 돌려주지 않는다."""
    if not isinstance(exc, anthropic.APIStatusError) or exc.status_code not in (400, 402):
        return False
    text = _error_message(exc).lower()
    return any(marker in text for marker in _CREDIT_MARKERS)


def mark_exhausted() -> None:
    global _skip_until
    with _lock:
        first = time.monotonic() >= _skip_until
        _skip_until = time.monotonic() + SKIP_SEC
    if first:
        _LOG.warning("[CLAUDE-CREDIT] cause=%s skip_sec=%d", CREDIT_EXHAUSTED, int(SKIP_SEC))


def mark_ok() -> None:
    """호출이 성공하면(크레딧을 다시 채운 경우 등) 건너뛰기를 바로 푼다."""
    global _skip_until
    with _lock:
        was = time.monotonic() < _skip_until
        _skip_until = 0.0
    if was:
        _LOG.info("[CLAUDE-CREDIT] cause=recovered")


def skipping() -> bool:
    """지금 건너뛰는 중인가(마지막 소진 감지 뒤 SKIP_SEC 이내)."""
    with _lock:
        return time.monotonic() < _skip_until


def reset_for_tests() -> None:
    global _skip_until
    with _lock:
        _skip_until = 0.0
