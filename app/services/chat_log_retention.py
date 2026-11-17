"""챗 기록(chat_logs) 보관 기간 정리 (2026-11-17 개인정보 점검 H-2, 사용자 결정 90일).

created_at 이 CHAT_LOG_RETENTION_DAYS(기본 90)일보다 오래된 행을 기록째 지운다. 탈퇴·연결 해제 정리는 따로 그대로 돈다.
기동 때 데몬 스레드를 하나 띄워 바로 한 번, 그 뒤 6시간마다 다시 본다. 0 이하이면 꺼진다.
로그에는 지운 개수와 예외 클래스명만 남긴다(상담 내용·사용자 ID 미기록).
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Callable

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.models.chat_log import ChatLog

_LOG = logging.getLogger(__name__)
_LOOP_INTERVAL_SEC = 6 * 3600
# 한 번에 지우는 행 수 - 예약 턴 context_json 이 수십 KB라 한 트랜잭션을 작게 둔다.
_BATCH = 1000

_lock = threading.Lock()
_thread: threading.Thread | None = None
_stop = threading.Event()


def purge_old_chat_logs(db: Session, days: int, now: datetime | None = None) -> int:
    """보관 기한이 지난 행을 나눠 지우고 지운 개수를 돌려준다. created_at 은 UTC naive(datetime.utcnow)로 저장된다."""
    if days <= 0:
        return 0
    cutoff = (now or datetime.now(timezone.utc).replace(tzinfo=None)) - timedelta(days=days)
    total = 0
    while True:
        ids = [row[0] for row in db.query(ChatLog.id).filter(ChatLog.created_at < cutoff).order_by(ChatLog.id).limit(_BATCH).all()]
        if not ids:
            break
        db.query(ChatLog).filter(ChatLog.id.in_(ids)).delete(synchronize_session=False)
        db.commit()
        total += len(ids)
        if len(ids) < _BATCH:
            break
    return total


def run_once(settings: Settings | None = None, session_factory: Callable[[], Session] | None = None) -> int:
    """한 번 정리한다. 예외는 올리지 않는다(-1)."""
    s = settings or get_settings()
    days = int(getattr(s, "chat_log_retention_days", 90))
    if days <= 0:
        return 0
    if session_factory is None:
        from app.database.session import SessionLocal

        session_factory = SessionLocal
    db = session_factory()
    try:
        deleted = purge_old_chat_logs(db, days)
        if deleted:
            _LOG.info("[CHAT-RETENTION] deleted=%d days=%d", deleted, days)
        return deleted
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        _LOG.warning("[CHAT-RETENTION] purge failed type=%s", type(exc).__name__)
        return -1
    finally:
        db.close()


def start_loop(settings: Settings | None = None) -> bool:
    """정리 스레드를 띄운다(이미 돌고 있거나 꺼져 있으면 False). 첫 정리도 스레드에서 해 기동을 막지 않는다."""
    global _thread
    s = settings or get_settings()
    if int(getattr(s, "chat_log_retention_days", 90)) <= 0:
        _LOG.info("[CHAT-RETENTION] disabled")
        return False
    with _lock:
        if _thread is not None and _thread.is_alive():
            return False
        _stop.clear()
        _thread = threading.Thread(target=_loop, args=(s,), name="chat-log-retention", daemon=True)
        thread = _thread
    thread.start()
    return True


def _loop(s: Settings) -> None:
    run_once(s)
    while not _stop.wait(_LOOP_INTERVAL_SEC):
        run_once(s)


def stop_loop() -> None:
    _stop.set()
