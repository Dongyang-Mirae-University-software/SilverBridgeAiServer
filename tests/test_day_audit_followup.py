"""2026-10-30 통합 점검 반영: 회원 키 못 받을 때 "잠시 후" 안내(M-1) · 정리 보류 중인 턴은 챗 기록을 남기지 않음(L-1)."""
from __future__ import annotations

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.database.session import get_db
from app.models.chat_log import ChatLog
from app.routers import chat_router
from app.schemas.chat_schema import ChatContext, ChatRequest
from app.services import chat_service as cs
from app.services import reservation_credential_service as rcs
from tests.test_chat_history_server_side import _router
from tests.test_reservation_confirm_chat_service import _send, env  # noqa: F401

_RETRY = "예약 정보를 처리하지 못했습니다. 잠시 후 다시 시도해 주세요."
_OFF = "지금은 병원 예약 기능을 사용할 수 없어요"


def test_member_key_unavailable_says_try_again_later(env, monkeypatch) -> None:  # noqa: F811
    monkeypatch.setattr(cs, "get_reservation_credential_service", lambda: SimpleNamespace(ensure_credential=lambda *a, **k: None))
    _router(env, monkeypatch, route="reservation")
    out = _send(env, "내과 예약해 주세요")
    assert out["reply"] == _RETRY and _OFF not in out["reply"]


def test_secret_missing_still_says_reservation_is_off(env, monkeypatch) -> None:  # noqa: F811
    env.svc._settings = SimpleNamespace(has_reservation_account_secret=False, claude_model_name="fake")
    _router(env, monkeypatch, route="reservation")
    out = _send(env, "내과 예약해 주세요")
    assert out["reply"].startswith(_OFF)


def _count(env, user_id: str) -> int:  # noqa: F811
    return env.db.query(ChatLog).filter(ChatLog.user_id == user_id).count()


def test_turn_for_purged_user_is_answered_but_not_logged(env, monkeypatch) -> None:  # noqa: F811
    _router(env, monkeypatch, route="symptom")
    rcs.mark_purged("user-1")
    out = _send(env, "어제부터 기침이 나요")
    assert out["reply"] and out["chatLogId"] is None
    assert _count(env, "user-1") == 0


def test_proxy_turn_about_purged_ward_is_not_logged(env, monkeypatch) -> None:  # noqa: F811
    _router(env, monkeypatch, route="symptom")
    rcs.mark_purged("ward01")
    payload = ChatRequest(userId="user-1", message="어제부터 기침이 나요", sessionId="s1",
                          context=ChatContext(name="보호자", wardId="ward01", wardName="피보호자"))
    out = env.svc.process_message(env.db, payload)
    assert out["reply"] and _count(env, "user-1") == 0


def test_normal_turn_is_logged(env, monkeypatch) -> None:  # noqa: F811
    _router(env, monkeypatch, route="symptom")
    out = _send(env, "어제부터 기침이 나요")
    assert out["chatLogId"] is not None and _count(env, "user-1") == 1


def test_chat_log_delete_api_sets_the_hold(env) -> None:  # noqa: F811
    app = FastAPI()
    app.include_router(chat_router.router)
    app.dependency_overrides[get_db] = lambda: env.db
    with TestClient(app) as client:
        assert client.delete("/api/v1/chat/logs", params={"userId": "gone01"}).status_code == 200
    assert rcs.recently_purged("gone01")
