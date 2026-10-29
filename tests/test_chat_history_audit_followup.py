"""2026-10-29 재점검(#99) 반영: 오류 턴 피보호자 구분(M-1) · 환자 이름·전화 이력 제외(L-1) · 지역 버튼(L-2) ·
직전 기록 한 번 읽기(L-3) · 응급 답 다음 일반 답에 119 한 줄(mt.01, 사용자 결정 (b))."""
from __future__ import annotations

from typing import Any

from sqlalchemy import event

from app.services import chat_service as cs
from tests.reservation_flow_helpers import FUTURE, FakeReservation, make_orchestrator, make_request
from tests.test_chat_history_server_side import _router, _send
from tests.test_reservation_confirm_chat_service import env  # noqa: F401

FOLLOWUP = "증상이 다시 생기거나 심해지면 바로 119에 전화하세요."


def _llm_replies(env, monkeypatch, risks: list[str]) -> None:  # noqa: F811
    """가짜 MedGemma - 호출 순서대로 위험도를 돌려준다(high 면 119 문장)."""
    queue = list(risks)

    def generate(*a, **k):  # noqa: ARG001
        env.llm.calls += 1
        risk = queue.pop(0)
        reply = "지금 바로 119에 전화해 주세요." if risk == "high" else "괜찮아지셨다니 다행이에요."
        return {"reply": reply, "riskLevel": risk, "recommendedAction": "x", "reservationRequired": False,
                "intent": "medical_advice", "engine": "fake", "modelName": "fake"}

    monkeypatch.setattr(env.llm, "generate", generate)


# ---- M-1 ---------------------------------------------------------------------------------------------------


def test_error_turn_stays_in_its_ward_session(env, monkeypatch) -> None:  # noqa: F811
    _router(env, monkeypatch, route="reservation")

    def boom(*a, **k):  # noqa: ARG001
        raise RuntimeError("x")

    monkeypatch.setattr(env.svc, "_run_reservation", boom)
    _send(env, "강남메디내과 내과 예약해줘", ward="w1")  # 내부 오류 → router 칸 없이 저장
    monkeypatch.undo()
    seen = _router(env, monkeypatch)
    _send(env, "그럼 어떻게 해요", ward="w1")
    assert [h["content"] for h in seen.histories[-1] if h["role"] == "user"] == ["강남메디내과 내과 예약해줘"]
    _send(env, "본인 상담이에요")  # 같은 sessionId 의 본인 상담에는 피보호자 턴이 섞이지 않는다
    assert [h["content"] for h in seen.histories[-1] if h["role"] == "user"] == []


# ---- L-1 ---------------------------------------------------------------------------------------------------


def test_patient_name_and_phone_buttons_are_not_history(env, monkeypatch) -> None:  # noqa: F811
    _send(env, ui=("hospital", "강남메디내과"))
    _send(env, ui=("patient_name", "김순자"))
    _send(env, ui=("phone", "010-2222-3333"))
    seen = _router(env, monkeypatch)
    _send(env, "고마워요")
    assert [h["content"] for h in seen.histories[-1] if h["role"] == "user"] == ["강남메디내과"]


# ---- L-2 ---------------------------------------------------------------------------------------------------


def _button(context: dict[str, Any], extracted_location: str) -> bool:
    orch = make_orchestrator(FakeReservation(), {"reservation_date": FUTURE, "location": extracted_location})
    out = orch.handle(make_request("근처 내과 예약해줘", context=context), {"rawMessage": "근처 내과 예약해줘"})
    assert out["tool"] == "search_hospital"
    return bool(out.get("ui"))


def test_other_region_button_not_for_user_given_detailed_region() -> None:
    profile = {"location": "서울 강남구"}
    assert _button(profile, "서울 강남구") is True  # 프로필 지역 그대로
    assert _button(profile, "서울 강남구 역삼동") is False  # 사용자가 동네까지 말함
    assert _button({"location": None, "address": "서울특별시 강남구 테헤란로 123"}, "서울특별시 강남구") is True


# ---- L-3 ---------------------------------------------------------------------------------------------------


def test_chat_logs_are_read_once_per_request(env, monkeypatch) -> None:  # noqa: F811
    _send(env, "강남메디내과 내과 14:00 예약해줘")  # 예약 경로(기본 라우터) - 직전 기록 확인 3곳을 모두 지난다
    selects: list[str] = []

    def count(conn, cursor, statement, *a):  # noqa: ARG001
        # 직전 기록 조회(최신 순 정렬)만 센다 - 저장 뒤 db.refresh(id 조회)는 대상이 아니다.
        if statement.lstrip().upper().startswith("SELECT") and "chat_logs" in statement and "ORDER BY" in statement.upper():
            selects.append(statement)

    engine = env.db.get_bind()
    event.listen(engine, "before_cursor_execute", count)
    try:
        _send(env, "내일로 바꿔줘")
    finally:
        event.remove(engine, "before_cursor_execute", count)
    assert len(selects) == 1


# ---- mt.01 (b) ---------------------------------------------------------------------------------------------


def test_ordinary_reply_after_emergency_gets_119_line(env, monkeypatch) -> None:  # noqa: F811
    _router(env, monkeypatch, route="symptom")
    _llm_replies(env, monkeypatch, ["high", "medium", "medium"])
    _send(env, "가슴이 조이고 식은땀이 나요")
    after = _send(env, "이제 좀 괜찮아졌어요")
    assert after["reply"].endswith(FOLLOWUP) and after["riskLevel"] == "medium"
    again = _send(env, "고마워요")  # 직전이 medium 이면 붙이지 않는다(한 번만)
    assert FOLLOWUP not in again["reply"]


def test_no_119_line_without_previous_emergency_or_in_other_session(env, monkeypatch) -> None:  # noqa: F811
    _router(env, monkeypatch, route="symptom")
    _llm_replies(env, monkeypatch, ["medium", "medium", "high", "medium"])
    _send(env, "어제부터 기침이 나요")
    assert FOLLOWUP not in _send(env, "이제 좀 괜찮아졌어요")["reply"]
    _send(env, "가슴이 조이고 식은땀이 나요")
    assert FOLLOWUP not in _send(env, "이제 좀 괜찮아졌어요", session="s2")["reply"]  # 다른 세션


def test_no_119_line_after_crisis_reply(env, monkeypatch) -> None:  # noqa: F811
    _router(env, monkeypatch, route="general")
    _llm_replies(env, monkeypatch, ["medium"])
    _send(env, "죽고 싶어요")  # 위기 고정 응답(high, selfHarm) - 다음 턴은 109 안내가 따로 붙는다
    out = _send(env, "이제 좀 괜찮아졌어요")
    assert FOLLOWUP not in out["reply"]


def test_cs_constant_matches() -> None:
    assert cs._EMERGENCY_FOLLOWUP == FOLLOWUP
