"""예약 오케스트레이터와 그 확인·취소 흐름이 함께 쓰는 작은 헬퍼."""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

_LOG = logging.getLogger("app.services.reservation_orchestrator")
_SEOUL_TZ = ZoneInfo("Asia/Seoul")
_WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]


def _date_label(date_text: str) -> str:
    """안내 문구용 날짜. 오늘이면 "오늘", 아니면 "10월 13일(화)"."""
    try:
        target = datetime.strptime(date_text, "%Y-%m-%d").date()
    except ValueError:
        return date_text
    if target == datetime.now(_SEOUL_TZ).date():
        return "오늘"
    return f"{target.month}월 {target.day}일({_WEEKDAYS[target.weekday()]})"


def _digits(phone: str) -> str:
    return re.sub(r"\D", "", phone or "")


def _phone_last4(phone: str | None) -> str:
    digits = _digits(phone or "")
    return digits[-4:] if digits else ""


def _trace(meta: dict[str, Any], event: str, **details: Any) -> None:
    trace = meta.setdefault("decisionTrace", [])
    if isinstance(trace, list):
        payload = {"event": event, **{k: v for k, v in details.items() if v is not None}}
        trace.append(payload)
        _LOG.info("[예약트레이스] %s %s", event, {k: v for k, v in details.items() if v is not None})
