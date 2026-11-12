"""상담 기록 목록·상세의 createdAt 시간대 표시와 intent(2026-11-12 FE 실측 반영).
① created_at 은 시간대 없는 UTC로 저장된다 - 내보낼 때 "Z"를 붙여 받는 쪽이 현지 시각으로 잘못 읽지 않게 한다.
② 응답 의도를 기록에 남겨 목록·상세의 intent 로 내보낸다(전에는 저장하지 않아 늘 비어 있었다). 옛 행은 비어 있는 그대로."""
from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.chat_log import ChatLog
from app.schemas.chat_schema import ChatContext, ChatRequest
from app.services.chat_service import ChatService, _utc_iso
from tests.test_reservation_confirm_chat_service import env  # noqa: F401


def _db():
    engine = create_engine(f"sqlite:///{tempfile.mkdtemp(prefix='sb_logtime_')}/t.db")
    ChatLog.__table__.create(bind=engine)
    return sessionmaker(bind=engine)()


def test_utc_iso_marks_naive_utc_with_z_and_converts_aware_values() -> None:
    assert _utc_iso(datetime(2026, 11, 12, 5, 23, 39, 95242)) == "2026-11-12T05:23:39.095242Z"
    kst = timezone(timedelta(hours=9))
    assert _utc_iso(datetime(2026, 11, 12, 14, 23, 39, tzinfo=kst)) == "2026-11-12T05:23:39Z"


def test_list_and_detail_created_at_carry_utc_marker() -> None:
    db = _db()
    row = ChatLog(chat_no="c1", user_id="u", message="q", context_json="{}", reply="a",
                  created_at=datetime(2026, 11, 12, 5, 23, 39, 95242))
    db.add(row)
    db.commit()

    listed = ChatService.list_logs(db, "u")[0]["createdAt"]
    detail = ChatService.get_log(db, row.id, "u")["createdAt"]
    assert listed == detail == "2026-11-12T05:23:39.095242Z"
    # 받는 쪽이 읽으면 한국 시각 14:23 이다(9시간 이르게 보이던 문제)
    parsed = datetime.fromisoformat(listed.replace("Z", "+00:00"))
    assert parsed.astimezone(timezone(timedelta(hours=9))).hour == 14


def test_old_rows_without_intent_still_list_with_none() -> None:
    db = _db()
    row = ChatLog(chat_no="c2", user_id="u", message="q", context_json=json.dumps({"sessionId": "s"}), reply="a")
    db.add(row)
    db.commit()
    assert ChatService.list_logs(db, "u")[0]["intent"] is None
    assert ChatService.get_log(db, row.id, "u")["intent"] is None


def test_new_log_stores_response_intent_and_lists_it(env) -> None:  # noqa: F811
    payload = ChatRequest(
        userId="user-1", message="강남메디내과 내과 14:00 예약해줘", sessionId="s1",
        context=ChatContext(name="홍길동", phone="010-1234-5678", location="서울 강남구"),
    )
    result = env.svc.process_message(env.db, payload)
    assert result["intent"]

    row = env.db.query(ChatLog).order_by(ChatLog.id.desc()).first()
    assert json.loads(row.context_json)["intent"] == result["intent"]
    listed = ChatService.list_logs(env.db, "user-1")[0]
    assert listed["intent"] == result["intent"]
    assert ChatService.get_log(env.db, row.id, "user-1")["intent"] == result["intent"]
