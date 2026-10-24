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


# 사용자가 말한 값(지역·병원명·날짜·시간)은 표준 출력 로그에 원문 대신 길이만 남긴다(2026-10-25 라우터 점검 I-1).
# 챗 기록의 decisionTrace(DB, BE 기록 조회에서 걸러짐)에는 그대로 남긴다. 진료과는 고정 목록 값이라 그대로 둔다.
_LOG_MASKED_KEYS = frozenset({
    "location", "region", "tried_regions", "hospital", "hospital_name",
    "date", "reservation_date", "time", "reservation_time", "requested", "found",
})


def log_len(value: Any) -> str:
    """로그용 길이 표시. 비었으면 (empty), 목록이면 개수."""
    if value is None or value == "" or value == [] or value == ():
        return "(empty)"
    if isinstance(value, (list, tuple, set)):
        return f"count={len(value)}"
    return f"len={len(str(value))}"


def _masked_for_log(details: dict[str, Any]) -> dict[str, Any]:
    return {k: (log_len(v) if k in _LOG_MASKED_KEYS else v) for k, v in details.items() if v is not None}


def _trace(meta: dict[str, Any], event: str, **details: Any) -> None:
    trace = meta.setdefault("decisionTrace", [])
    if isinstance(trace, list):
        payload = {"event": event, **{k: v for k, v in details.items() if v is not None}}
        trace.append(payload)
        _LOG.info("[예약트레이스] %s %s", event, _masked_for_log(details))
