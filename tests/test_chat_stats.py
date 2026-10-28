from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import get_settings
from app.core.response import error_response
from app.core.security import require_api_key
from app.database.base import Base
from app.database.schema_upgrades import ensure_indexes
from app.database.session import get_db
from app.models.chat_log import ChatLog
from app.routers import chat_router

KEY = "test-key"
SECRET_MESSAGE = "가슴이 답답해요-원문"
SECRET_REPLY = "119에 연락하세요-원문"


def _ctx(role=None, route=None, engine="medgemma", type_="message", tool=None, trace=None, meta=None) -> str:
    ctx: dict = {"engine": engine, "type": type_, "tool": tool, "decisionTrace": trace or []}
    if role is not None:
        ctx["userContext"] = {"role": role, "name": "홍길동"}
    upstream = dict(meta or {})
    if route is not None:
        upstream["router"] = {"route": route}
    if upstream:
        ctx["upstreamMeta"] = upstream
    return json.dumps(ctx, ensure_ascii=False)


def _log(no: str, user: str, created_at: datetime, context_json: str, risk="low", reservation=False) -> ChatLog:
    return ChatLog(
        chat_no=no, user_id=user, message=SECRET_MESSAGE, reply=SECRET_REPLY, context_json=context_json,
        risk_level=risk, reservation_required=reservation, created_at=created_at,
    )


@contextmanager
def _client(rows: list[ChatLog]):
    engine = create_engine("sqlite+pysqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    session.add_all(rows)
    session.commit()
    app = FastAPI()

    # app/main.py 의 예외 핸들러와 같은 규칙(app.main 은 모델 로딩까지 끌고 와서 직접 붙인다) - 오류는 최상위 errorCode.
    @app.exception_handler(HTTPException)
    async def _handler(_: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "success" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(status_code=exc.status_code, content=error_response(str(exc.detail), "HTTP_ERROR", None))

    app.include_router(chat_router.router, dependencies=[Depends(require_api_key)])
    app.dependency_overrides[get_db] = lambda: session
    app.dependency_overrides[get_settings] = lambda: SimpleNamespace(api_key=KEY)
    try:
        with TestClient(app) as client:
            yield client
    finally:
        session.close()


def _get(client: TestClient, params: dict, key: str | None = KEY):
    headers = {"X-API-Key": key} if key else {}
    return client.get("/api/v1/chat/stats", params=params, headers=headers)


# created_at 은 UTC(시간대 없음)로 저장된다. KST 2026-10-28 = UTC 10-27 15:00 ~ 10-28 15:00.
def _rows() -> list[ChatLog]:
    return [
        _log("c-1", "aaaaaa", datetime(2026, 10, 27, 14, 59, 59), _ctx("GUARDIAN", "symptom")),  # KST 10-27 23:59:59 - 밖
        _log("c-2", "aaaaaa", datetime(2026, 10, 27, 15, 0, 0), _ctx("GUARDIAN", "symptom"), risk="high", reservation=True),
        _log("c-3", "bbbbbb", datetime(2026, 10, 28, 3, 0), _ctx("ward", "reservation", engine="claude",
                                                                  type_="tool_result", tool="make_appointment"), reservation=True),
        _log("c-4", "bbbbbb", datetime(2026, 10, 28, 4, 0), _ctx(None, "reservation", engine="claude",
                                                                  type_="tool_error", tool="make_appointment"), reservation=True),
        _log("c-5", "cccccc", datetime(2026, 10, 28, 5, 0), _ctx("ADMIN", None, engine="fallback")),
        _log("c-6", "cccccc", datetime(2026, 10, 28, 6, 0), _ctx("WARD", "reservation", engine="claude",
                                                                  trace=[{"event": "reservation_flow_fallback"}]), reservation=True),
        _log("c-7", "cccccc", datetime(2026, 10, 28, 7, 0), _ctx("WARD", "reservation", engine="claude",
                                                                  trace=[{"event": "reservation_disabled"}]), reservation=True),
        _log("c-8", "dddddd", datetime(2026, 10, 28, 14, 59, 59), "not-json"),  # KST 10-28 23:59:59 - 안
        _log("c-9", "eeeeee", datetime(2026, 10, 28, 15, 0, 0), _ctx("GUARDIAN", "general")),  # KST 10-29 00:00 - 밖
    ]


def test_counts_one_kst_day_with_midnight_boundaries() -> None:
    with _client(_rows()) as client:
        res = _get(client, {"from": "2026-10-28", "to": "2026-10-28"})
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["from"] == "2026-10-28" and data["to"] == "2026-10-28"
    assert data["totalChats"] == 7  # c-2 ~ c-8
    assert data["users"] == 4  # aaaaaa·bbbbbb·cccccc·dddddd
    assert data["byRole"] == {"guardian": 1, "ward": 3, "unknown": 3}  # ADMIN·없음·JSON 아님 = unknown
    assert data["byRoute"] == {"reservation": 4, "symptom": 1, "general": 0, "unknown": 2}
    assert data["emergencyGuidance"] == 1
    assert data["reservationCompleted"] == 1  # tool_error 는 완료가 아니다
    assert data["fallbackReplies"] == 2  # engine=fallback + 예약 흐름 실패, 예약 꺼짐은 제외
    assert data["daily"] == [{"date": "2026-10-28", "totalChats": 7}]
    assert data["generatedAt"].endswith("+09:00")
    assert "reservationSuggested" not in data


def test_daily_includes_zero_days_and_splits_by_kst_date() -> None:
    with _client(_rows()) as client:
        data = _get(client, {"from": "2026-10-26", "to": "2026-10-29"}).json()["data"]
    assert data["daily"] == [
        {"date": "2026-10-26", "totalChats": 0},
        {"date": "2026-10-27", "totalChats": 1},
        {"date": "2026-10-28", "totalChats": 7},
        {"date": "2026-10-29", "totalChats": 1},
    ]
    assert data["totalChats"] == 9


def test_empty_range_is_zero_not_null() -> None:
    with _client(_rows()) as client:
        data = _get(client, {"from": "2026-09-01", "to": "2026-09-03"}).json()["data"]
    assert data["totalChats"] == 0 and data["users"] == 0
    assert data["byRole"] == {"guardian": 0, "ward": 0, "unknown": 0}
    assert data["byRoute"] == {"reservation": 0, "symptom": 0, "general": 0, "unknown": 0}
    assert data["emergencyGuidance"] == 0 and data["reservationCompleted"] == 0 and data["fallbackReplies"] == 0
    assert [d["totalChats"] for d in data["daily"]] == [0, 0, 0]


def test_31_days_ok_32_days_400() -> None:
    with _client([]) as client:
        assert _get(client, {"from": "2026-10-01", "to": "2026-10-31"}).status_code == 200
        res = _get(client, {"from": "2026-10-01", "to": "2026-11-01"})
    assert res.status_code == 400
    assert res.json()["errorCode"] == "CHAT_STATS_INVALID_RANGE"


def test_invalid_ranges_are_400() -> None:
    bad = [
        {},
        {"from": "2026-10-01"},
        {"to": "2026-10-01"},
        {"from": "20261001", "to": "20261002"},
        {"from": "2026-10-1", "to": "2026-10-02"},
        {"from": "2026-02-30", "to": "2026-03-01"},
        {"from": "2026-10-02", "to": "2026-10-01"},
        {"from": "2026-10-01" * 100, "to": "2026-10-02"},
        {"from": "0001-01-01", "to": "0001-01-02"},  # UTC 변환 범위 밖(점검 L-1)
        {"from": "9999-12-30", "to": "9999-12-31"},  # 다음 날 범위 밖
    ]
    with _client([]) as client:
        for params in bad:
            res = _get(client, params)
            assert res.status_code == 400, params
            assert res.json()["errorCode"] == "CHAT_STATS_INVALID_RANGE"


def test_missing_or_wrong_key_is_401() -> None:
    with _client(_rows()) as client:
        assert _get(client, {"from": "2026-10-28", "to": "2026-10-28"}, key=None).status_code == 401
        assert _get(client, {"from": "2026-10-28", "to": "2026-10-28"}, key="wrong").status_code == 401


def test_response_has_no_content_or_user_ids(caplog) -> None:
    caplog.set_level("INFO")
    with _client(_rows()) as client:
        body = _get(client, {"from": "2026-10-26", "to": "2026-10-29"}).text
    for secret in (SECRET_MESSAGE, SECRET_REPLY, "홍길동", "aaaaaa", "bbbbbb", "cccccc", "dddddd", "eeeeee"):
        assert secret not in body
        assert secret not in caplog.text
    assert "[CHAT-STATS]" in caplog.text


def test_main_wires_chat_router_behind_api_key() -> None:
    # 위 테스트는 키 검사를 직접 붙인 앱이라, 운영 연결(main.py)에서 빠지면 못 잡는다(점검 L-4). 연결 줄을 고정한다.
    main_src = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(encoding="utf-8")
    assert "app.include_router(chat_router, dependencies=[Depends(require_api_key)])" in main_src
    assert "ensure_indexes(engine)" in main_src


def test_malformed_contexts_and_role_spacing_are_counted_safely() -> None:
    day = datetime(2026, 10, 28, 3, 0)
    rows = [
        _log("m-1", "u1", day, "[]"),  # dict 가 아닌 JSON
        _log("m-2", "u1", day, json.dumps({"userContext": "GUARDIAN", "upstreamMeta": "x"})),  # 하위 칸이 dict 아님
        _log("m-3", "u2", day, _ctx(" Guardian ", "general")),  # 앞뒤 공백·대소문자
        _log("m-4", "u2", day, json.dumps({"upstreamMeta": {"router": {"route": "other"}}})),  # 모르는 경로
    ]
    with _client(rows) as client:
        data = _get(client, {"from": "2026-10-28", "to": "2026-10-28"}).json()["data"]
    assert data["totalChats"] == 4
    assert data["byRole"] == {"guardian": 1, "ward": 0, "unknown": 3}
    assert data["byRoute"] == {"reservation": 0, "symptom": 0, "general": 1, "unknown": 3}
    assert data["fallbackReplies"] == 0


def test_self_harm_reply_is_emergency_and_fallback_only_when_generation_failed() -> None:
    day = datetime(2026, 10, 28, 3, 0)
    rows = [
        # 라우터·키워드로 바로 낸 자해 고정 응답 = 응급 안내, 대체 응답 아님
        _log("s-1", "u1", day, _ctx("WARD", "symptom", engine="rule", meta={"selfHarm": True}), risk="high"),
        # 생성 실패 뒤 낸 자해 고정 응답 = 응급 안내이면서 대체 응답(점검 L-3)
        _log("s-2", "u1", day, _ctx("WARD", "symptom", engine="rule", meta={"selfHarm": True, "generationFailed": True}),
             risk="high"),
    ]
    with _client(rows) as client:
        data = _get(client, {"from": "2026-10-28", "to": "2026-10-28"}).json()["data"]
    assert data["emergencyGuidance"] == 2
    assert data["fallbackReplies"] == 1


def test_ensure_indexes_adds_created_at_index_to_existing_table() -> None:
    # 운영 DB 처럼 인덱스 없이 만들어진 기존 테이블에도 붙고, 다시 불러도 그대로다(점검 M-1).
    engine = create_engine("sqlite+pysqlite://", poolclass=StaticPool)
    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE chat_logs (id INTEGER PRIMARY KEY, created_at DATETIME)")
    ensure_indexes(engine)
    ensure_indexes(engine)
    names = {ix["name"] for ix in inspect(engine).get_indexes("chat_logs")}
    assert "ix_chat_logs_created_at" in names
    assert ChatLog.__table__.c.created_at.index is True  # 새 DB 는 create_all 이 같은 이름으로 만든다
