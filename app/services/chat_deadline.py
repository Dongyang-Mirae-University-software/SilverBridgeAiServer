"""챗 요청 하나의 전체 마감(O-3, 2026-10-28).

BE 중계가 130초에 연결을 끊으면 사용자는 아무 답도 못 받고, AI 는 그 뒤에도 일을 계속한다(예약이 생길 수도 있다).
그래서 요청 시작 시각 기준 마감 한 곳(CHAT_DEADLINE_SEC)을 두고, 각 호출은 남은 시간만큼만 기다린다 - 호출마다 상한을 새로 세지 않는다.
마감은 contextvar 라 챗 요청 처리 스레드 안에서만 보인다. 챗 밖(예약 계정 API·관리 API)에서는 마감이 없어 예전 상한 그대로다.
로그는 단계 이름·남은 ms 만 남긴다(상담 내용·이름·전화 금지).
"""
from __future__ import annotations

import contextvars
import logging
import time
from contextlib import contextmanager
from typing import Iterator

from app.core.config import get_settings

_LOG = logging.getLogger(__name__)

# perf_counter 기준 마감 시각. None 이면 마감 없음.
_DEADLINE: contextvars.ContextVar[float | None] = contextvars.ContextVar("chat_deadline", default=None)


class ChatDeadlineExceeded(RuntimeError):
    """남은 시간이 모자라 호출을 시작하지 않았다. stage 는 고정 단계 이름이라 로그에 그대로 남겨도 된다."""

    def __init__(self, stage: str) -> None:
        super().__init__(stage)
        self.stage = stage


@contextmanager
def chat_deadline(seconds: float | None = None) -> Iterator[None]:
    """이 블록 안의 호출에 전체 마감을 건다. seconds 를 주지 않으면 CHAT_DEADLINE_SEC."""
    limit = float(get_settings().chat_deadline_sec if seconds is None else seconds)
    token = _DEADLINE.set(time.perf_counter() + limit)
    try:
        yield
    finally:
        _DEADLINE.reset(token)


@contextmanager
def tighter_deadline(seconds: float) -> Iterator[None]:
    """이 블록 안에서만 마감을 min(지금 마감, 지금 + seconds)로 당긴다. 블록을 나가면 원래 마감으로 돌아간다(2026-10-29 L-4).
    마감이 없으면(챗 밖) 아무것도 하지 않는다."""
    current = _DEADLINE.get()
    if current is None:
        yield
        return
    token = _DEADLINE.set(min(current, time.perf_counter() + seconds))
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def remaining_sec() -> float | None:
    end = _DEADLINE.get()
    return None if end is None else end - time.perf_counter()


def log_deadline(stage: str, remaining: float | None) -> None:
    _LOG.warning("[CHAT-DEADLINE] stage=%s remainingMs=%d", stage, int((remaining or 0.0) * 1000))


def call_timeout(stage: str, cap: float, *, reserve: float = 0.0, minimum: float = 1.0) -> float:
    """이번 호출에 줄 시간 = min(cap, 남은 시간 - reserve). 마감이 없으면 cap.
    minimum 보다 작으면 호출하지 않고 ChatDeadlineExceeded - reserve 는 뒤에 이어질 단계가 쓸 시간이다."""
    remaining = remaining_sec()
    if remaining is None:
        return cap
    allowed = min(cap, remaining - reserve)
    if allowed < minimum:
        log_deadline(stage, remaining)
        raise ChatDeadlineExceeded(stage)
    return allowed


def has_time(stage: str, needed: float) -> bool:
    """남은 시간이 needed 이상인가(마감이 없으면 True). 모자라면 [CHAT-DEADLINE] 을 남긴다."""
    remaining = remaining_sec()
    if remaining is None or remaining >= needed:
        return True
    log_deadline(stage, remaining)
    return False


def irreversible_min_sec() -> float:
    return float(get_settings().chat_irreversible_min_remaining_sec)


def gen_min_sec() -> float:
    return float(get_settings().chat_gen_min_remaining_sec)
