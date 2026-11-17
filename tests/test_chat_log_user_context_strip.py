"""챗 기록 userContext 에는 다시 읽는 키(role·wardId)만 저장한다 (2026-11-17 개인정보 점검 H-1).

이름·전화·주소·생년월일 등은 이번 턴 처리(예약 계정·초안·MedGemma)에만 요청 값으로 쓰고 기록에 남기지 않는다.
남긴 두 키로 대리 상담 판정·탈퇴/연결 해제 대리 기록 삭제·관리자 역할 집계가 그대로 동작해야 한다.
"""
from __future__ import annotations

import json

from app.models.chat_log import ChatLog
from app.services import chat_service as cs
from app.services.chat_service import ChatService
from tests.test_chat_history_server_side import _router
from tests.test_chat_profile_timeout import _FULL, _send
from tests.test_reservation_confirm_chat_service import env  # noqa: F401

# 저장 기록 어디에도 남으면 안 되는 프로필 값(_FULL·대리 상담 값)
_PROFILE_VALUES = ("홍길동", "김순자", "010-1234-5678", "010-9999-8888", "a@b.c", "테헤란", "101호", "06234", "1950-03-02")


def _last_ctx(env) -> dict:  # noqa: F811
    row = env.db.query(ChatLog).order_by(ChatLog.id.desc()).first()
    return json.loads(row.context_json)


def test_proxy_consult_stores_only_role_and_ward_id(env, monkeypatch) -> None:  # noqa: F811
    _router(env, monkeypatch, route="symptom")
    _send(env, "어제부터 기침이 나요", age=76, wardId="ward-9", wardName="김순자", wardPhone="010-9999-8888", **_FULL)
    ctx = _last_ctx(env)
    assert ctx["userContext"] == {"role": "GUARDIAN", "wardId": "ward-9"}
    raw = json.dumps(ctx, ensure_ascii=False)
    assert not [v for v in _PROFILE_VALUES if v in raw]


def test_own_consult_stores_only_role(env, monkeypatch) -> None:  # noqa: F811
    _router(env, monkeypatch, route="symptom")
    _send(env, "어제부터 기침이 나요", **{**_FULL, "role": "WARD"})
    assert _last_ctx(env)["userContext"] == {"role": "WARD"}


def test_stored_row_still_drives_ward_lookup_and_purge(env, monkeypatch) -> None:  # noqa: F811
    _router(env, monkeypatch, route="symptom")
    _send(env, "어제부터 기침이 나요", wardId="ward-9", wardName="김순자", wardPhone="010-9999-8888", **_FULL)
    row = env.db.query(ChatLog).order_by(ChatLog.id.desc()).first()
    assert ChatService._row_ward_id(json.loads(row.context_json)) == "ward-9"
    assert "ward-9" in ChatService._proxy_ward_ids(row.context_json)
    # 피보호자 탈퇴 정리는 보호자 ID 로 저장된 이 대리 상담 기록도 지운다
    own, proxy = ChatService.delete_logs(env.db, "ward-9")
    assert (own, proxy) == (0, 1)
    assert env.db.query(ChatLog).count() == 0


def test_router_fallback_row_keeps_ward_id_in_user_context(env, monkeypatch) -> None:  # noqa: F811
    # router 칸이 없는 기록(내부 오류 대체 응답 등)은 userContext.wardId 로 대리 상담을 가린다 - 그 키는 남아야 한다
    ctx = {"sessionId": "s1", "userContext": cs._persistable_user_context({"wardId": "ward-9", "name": "김순자"})}
    assert ChatService._row_ward_id(ctx) == "ward-9"


def test_persistable_user_context_is_allow_list() -> None:
    assert cs._persistable_user_context(None) == {}
    assert cs._persistable_user_context("GUARDIAN") == {}
    assert cs._persistable_user_context({"role": "", "wardId": None, "name": "x"}) == {}
    # 요청에 새 키가 늘어도 저장되지 않는다
    assert cs._persistable_user_context({"role": "WARD", "newProfileKey": "v", "guardianWards": [1]}) == {"role": "WARD"}
