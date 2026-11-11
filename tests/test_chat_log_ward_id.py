"""상담 기록 목록·상세의 wardId 칸(2026-11-11) - FE 상담 기록을 피보호자별로 나눈다.
기록이 어느 피보호자 상담인지는 이력 조립과 같은 규칙(_row_ward_id: router.wardId 먼저, 없으면 userContext.wardId)으로 본다."""
from __future__ import annotations

import json
import tempfile

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.chat_log import ChatLog
from app.services.chat_service import ChatService


def _db():
    engine = create_engine(f"sqlite:///{tempfile.mkdtemp(prefix='sb_wardid_')}/t.db")
    ChatLog.__table__.create(bind=engine)
    return sessionmaker(bind=engine)()


def _add(db, no: str, ctx: dict, user_id: str = "guardian-1") -> int:
    row = ChatLog(chat_no=no, user_id=user_id, message="q", context_json=json.dumps(ctx), reply="a")
    db.add(row)
    db.commit()
    return row.id


def test_list_and_detail_carry_ward_id_of_proxy_consultation() -> None:
    db = _db()
    proxy = _add(db, "c1", {"sessionId": "s1", "upstreamMeta": {"router": {"wardId": "ward-1"}}})
    # router 칸이 없는 기록(내부 오류 대체 응답)은 userContext.wardId 로 본다
    fallback = _add(db, "c2", {"sessionId": "s1", "userContext": {"wardId": "ward-1"}})
    own = _add(db, "c3", {"sessionId": "s2", "upstreamMeta": {"router": {"wardId": ""}}, "userContext": {"wardId": "ward-1"}})
    plain = _add(db, "c4", {"sessionId": "s3"})

    by_id = {item["id"]: item for item in ChatService.list_logs(db, "guardian-1")}
    assert by_id[proxy]["wardId"] == "ward-1"
    assert by_id[fallback]["wardId"] == "ward-1"
    # router 가 본인 상담이라고 정했으면 userContext 값보다 router 를 따른다(이력 조립과 같은 규칙)
    assert by_id[own]["wardId"] is None
    assert by_id[plain]["wardId"] is None

    assert ChatService.get_log(db, proxy, "guardian-1")["wardId"] == "ward-1"
    assert ChatService.get_log(db, plain, "guardian-1")["wardId"] is None


def test_broken_context_json_gives_no_ward_id() -> None:
    db = _db()
    row = ChatLog(chat_no="c9", user_id="u", message="q", context_json="{not json", reply="a")
    db.add(row)
    db.commit()
    assert ChatService.list_logs(db, "u")[0]["wardId"] is None
