"""챗 구간 시간 기록(2026-10-22): timing 은 챗 기록 upstreamMeta 와 [CHAT-TIMING] 로그에만 남고 응답 필드는 그대로다.
평가 스크립트(tests/eval/run_eval.py)가 이 값을 기록 상세 API 로 읽는다."""
from __future__ import annotations

import json
import logging
from typing import Any

from app.models.chat_log import ChatLog
from app.schemas.chat_schema import ChatContext, ChatRequest
from tests.test_reservation_confirm_chat_service import env  # noqa: F401

# 응답 계약(이번 변경 전과 같은 키 집합). timing·meta 가 응답에 새어 나오면 여기서 걸린다.
RESPONSE_KEYS = {
    "sessionId", "reply", "riskLevel", "recommendedAction", "reservationRequired", "intent", "engine", "modelName",
    "type", "ui", "tool", "toolData", "summary", "possibleCauses", "homeCare", "visitHospitalIf", "emergencyWarning",
    "chatLogId", "chatNo",
}
TIMING_KEYS = {"routerMs", "llmMs", "reservationMs", "postMs", "totalMs"}
SENSITIVE = "숨을 못 쉬겠어요 홍길순 010-5555-6666"


def _send(env, message: str) -> dict[str, Any]:  # noqa: F811
    payload = ChatRequest(
        userId="user-1", message=message, sessionId="s1",
        context=ChatContext(name="홍길동", phone="010-1234-5678", location="서울 강남구"),
    )
    return env.svc.process_message(env.db, payload)


def _last_meta(env) -> dict[str, Any]:  # noqa: F811
    row = env.db.query(ChatLog).order_by(ChatLog.id.desc()).first()
    return json.loads(row.context_json)["upstreamMeta"]


def test_symptom_turn_records_router_and_llm_time_without_changing_response(env) -> None:  # noqa: F811
    out = _send(env, "숨을 못 쉬겠어요")  # 응급 키워드라 증상 상담(LLM)으로 간다
    assert set(out) == RESPONSE_KEYS
    timing = _last_meta(env)["timing"]
    assert set(timing) == TIMING_KEYS
    assert isinstance(timing["routerMs"], int) and isinstance(timing["llmMs"], int)
    assert timing["reservationMs"] is None
    assert timing["totalMs"] >= timing["llmMs"]
    assert _last_meta(env)["router"]["route"] == "symptom"  # 라우터 정보도 그대로 남는다


def test_reservation_turn_records_reservation_time_and_still_hides_the_key(env) -> None:  # noqa: F811
    out = _send(env, "강남메디내과 내과 14:00 예약해줘")
    assert set(out) == RESPONSE_KEYS
    meta = _last_meta(env)
    assert isinstance(meta["timing"]["reservationMs"], int) and meta["timing"]["llmMs"] is None
    raw = env.db.query(ChatLog).order_by(ChatLog.id.desc()).first().context_json
    assert "fake-key" not in raw
    assert not {"reservationApiKey", "storedFlow", "rawMessage"} & set(meta)


def test_llm_failure_still_records_timing(env, monkeypatch) -> None:  # noqa: F811
    def boom(*a, **k):  # noqa: ARG001
        raise RuntimeError(SENSITIVE)

    monkeypatch.setattr(env.llm, "generate", boom)
    out = _send(env, "숨을 못 쉬겠어요")
    assert out["engine"] == "fallback"
    assert isinstance(_last_meta(env)["timing"]["llmMs"], int)


def test_timing_log_line_has_only_route_engine_and_ms(env, caplog) -> None:  # noqa: F811
    with caplog.at_level(logging.INFO, logger="app.services.chat_service"):
        _send(env, SENSITIVE)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[CHAT-TIMING]")]
    assert len(lines) == 1
    line = lines[0]
    assert "route=symptom" in line and "total_ms=" in line and "save_ms=" in line
    assert "숨을" not in line and "홍길순" not in line and "010-5555-6666" not in line
