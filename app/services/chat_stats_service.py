from __future__ import annotations

import json
import logging
import re
import time
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.chat_log import ChatLog

_LOG = logging.getLogger(__name__)

KST = timezone(timedelta(hours=9))
MAX_RANGE_DAYS = 31
_ROLES = ("guardian", "ward")
_ROUTES = ("reservation", "symptom", "general")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


class ChatStatsRangeError(ValueError):
    """기간이 잘못됐을 때. 메시지는 고정 문구라 그대로 응답에 써도 된다."""


def parse_range(raw_from: str | None, raw_to: str | None) -> tuple[date, date]:
    if not raw_from or not raw_to:
        raise ChatStatsRangeError("from·to 가 필요합니다(YYYY-MM-DD).")
    try:
        if not (_DATE.fullmatch(raw_from) and _DATE.fullmatch(raw_to)):
            raise ValueError("format")
        start = date.fromisoformat(raw_from)
        end = date.fromisoformat(raw_to)
    except ValueError as exc:
        raise ChatStatsRangeError("날짜 형식은 YYYY-MM-DD 입니다.") from exc
    if start > end:
        raise ChatStatsRangeError("from 이 to 보다 늦습니다.")
    if (end - start).days + 1 > MAX_RANGE_DAYS:
        raise ChatStatsRangeError(f"기간은 최대 {MAX_RANGE_DAYS}일입니다.")
    return start, end


def _kst_day_start_as_utc(day: date) -> datetime:
    # created_at 은 datetime.utcnow()(시간대 없는 UTC)로 저장된다 - KST 자정을 같은 형식으로 바꿔 비교한다.
    return datetime(day.year, day.month, day.day, tzinfo=KST).astimezone(timezone.utc).replace(tzinfo=None)


def _kst_date(created_at: datetime) -> date:
    return created_at.replace(tzinfo=timezone.utc).astimezone(KST).date()


def _is_fallback(context: dict) -> bool:
    """모델·예약 흐름이 답을 만들지 못해 고정 안내로 대신 답한 턴.
    예약 기능을 꺼 둔 안내(reservation_disabled)는 설정이라 세지 않는다."""
    if context.get("engine") == "fallback":
        return True
    trace = context.get("decisionTrace")
    return isinstance(trace, list) and any(
        isinstance(item, dict) and item.get("event") == "reservation_flow_fallback" for item in trace
    )


class ChatStatsService:
    """관리자 통계용 전체 집계. 개수만 돌려주고 상담 원문·userId 는 응답·로그에 싣지 않는다."""

    @staticmethod
    def stats(db: Session, start: date, end: date, now: datetime | None = None) -> dict:
        started = time.perf_counter()
        rows = db.execute(
            select(
                ChatLog.user_id,
                ChatLog.created_at,
                ChatLog.risk_level,
                ChatLog.context_json,
            ).where(
                ChatLog.created_at >= _kst_day_start_as_utc(start),
                ChatLog.created_at < _kst_day_start_as_utc(end + timedelta(days=1)),
            )
        ).all()

        days = (end - start).days + 1
        daily = {start + timedelta(days=i): 0 for i in range(days)}
        by_role = {key: 0 for key in (*_ROLES, "unknown")}
        by_route = {key: 0 for key in (*_ROUTES, "unknown")}
        users: set[str] = set()
        emergency = completed = fallback = 0

        for user_id, created_at, risk_level, context_json in rows:
            users.add(user_id)
            day = _kst_date(created_at)
            if day in daily:
                daily[day] += 1
            try:
                context = json.loads(context_json or "{}")
            except ValueError:
                context = {}
            if not isinstance(context, dict):
                context = {}

            user_context = context.get("userContext")
            role = user_context.get("role") if isinstance(user_context, dict) else None
            role = role.strip().lower() if isinstance(role, str) else ""
            by_role[role if role in _ROLES else "unknown"] += 1

            meta = context.get("upstreamMeta")
            router = meta.get("router") if isinstance(meta, dict) else None
            route = router.get("route") if isinstance(router, dict) else None
            by_route[route if route in _ROUTES else "unknown"] += 1

            if risk_level == "high":
                emergency += 1
            # 예약 서비스가 2xx 로 예약을 만들었을 때만 남는 표시(변경 = 새 예약 생성도 1건).
            if context.get("type") == "tool_result" and context.get("tool") == "make_appointment":
                completed += 1
            if _is_fallback(context):
                fallback += 1

        result = {
            "from": start.isoformat(),
            "to": end.isoformat(),
            "totalChats": len(rows),
            "users": len(users),
            "byRole": by_role,
            "byRoute": by_route,
            "emergencyGuidance": emergency,
            "reservationCompleted": completed,
            "fallbackReplies": fallback,
            "daily": [{"date": day.isoformat(), "totalChats": count} for day, count in daily.items()],
            "generatedAt": (now or datetime.now(KST)).astimezone(KST).isoformat(timespec="seconds"),
        }
        _LOG.info(
            "[CHAT-STATS] from=%s to=%s rows=%s ms=%s",
            result["from"], result["to"], len(rows), int((time.perf_counter() - started) * 1000),
        )
        return result
