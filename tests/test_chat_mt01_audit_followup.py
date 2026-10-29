"""#100 가벼운 점검(#103) 반영(2026-10-29): 응급 다음 턴 내부 오류도 119 안내(M-1) · 119 한 줄 중복 방지(L-1) ·
저장 초안 없을 때 버튼으로 입력한 환자 이름·전화 채우기(L-2) · 라우터 예약 상태의 피보호자 구분 기준(L-3)."""
from __future__ import annotations

import json

from app.models.chat_log import ChatLog
from app.schemas.chat_schema import ChatContext, ChatRequest, UiSelection
from app.services import chat_service as cs
from tests.test_chat_history_audit_followup import FOLLOWUP, _llm_replies
from tests.test_chat_history_server_side import _router, _send
from tests.test_reservation_confirm_chat_service import env  # noqa: F401


def _llm_fixed(env, monkeypatch, replies: list[tuple[str, str]]) -> None:  # noqa: F811
    queue = list(replies)

    def generate(*a, **k):  # noqa: ARG001
        risk, reply = queue.pop(0)
        return {"reply": reply, "riskLevel": risk, "recommendedAction": "x", "reservationRequired": False,
                "intent": "medical_advice", "engine": "fake", "modelName": "fake"}

    monkeypatch.setattr(env.llm, "generate", generate)


# ---- M-1 ---------------------------------------------------------------------------------------------------


def test_internal_error_right_after_emergency_still_gives_119(env, monkeypatch) -> None:  # noqa: F811
    _router(env, monkeypatch, route="symptom")
    _llm_replies(env, monkeypatch, ["high"])
    _send(env, "가슴이 조이고 식은땀이 나요")

    def boom(*a, **k):  # noqa: ARG001
        raise RuntimeError("x")

    monkeypatch.setattr(env.svc, "_department_question_reply", boom)  # 바깥 except 로 떨어지게
    out = _send(env, "이제 좀 괜찮아졌어요")
    assert out["riskLevel"] == "high" and "119" in out["reply"] and out["emergencyWarning"]
    assert "안정 상태로 보입니다" not in out["reply"]


def test_internal_error_without_previous_emergency_is_unchanged(env, monkeypatch) -> None:  # noqa: F811
    _router(env, monkeypatch, route="symptom")
    _llm_replies(env, monkeypatch, ["medium"])
    _send(env, "어제부터 기침이 나요")

    def boom(*a, **k):  # noqa: ARG001
        raise RuntimeError("x")

    monkeypatch.setattr(env.svc, "_department_question_reply", boom)
    out = _send(env, "이제 좀 괜찮아졌어요")
    assert out["riskLevel"] != "high" and not out["emergencyWarning"]


# ---- L-1 ---------------------------------------------------------------------------------------------------


def test_no_duplicate_119_when_reply_already_has_it(env, monkeypatch) -> None:  # noqa: F811
    _router(env, monkeypatch, route="symptom")
    _llm_fixed(env, monkeypatch, [("high", "지금 바로 119에 전화해 주세요."),
                                  ("medium", "다행이에요. 갑자기 심해지면 바로 119에 연락하세요.")])
    _send(env, "가슴이 조이고 식은땀이 나요")
    out = _send(env, "이제 좀 괜찮아졌어요")
    assert out["reply"].count("119") == 1 and FOLLOWUP not in out["reply"]


# ---- L-2 ---------------------------------------------------------------------------------------------------


def _send_no_profile(env, message: str = "", ui: tuple[str, str] | None = None):  # noqa: F811
    """본인 상담 - FE처럼 프로필 이름·전화 없이 보낸다."""
    payload = ChatRequest(
        userId="user-1", message=message, sessionId="s1", context=ChatContext(location="서울 강남구"),
        uiSelection=UiSelection(field=ui[0], value=ui[1]) if ui else None,
    )
    return env.svc.process_message(env.db, payload)


def test_contact_buttons_refill_a_rebuilt_draft_without_entering_history(env, monkeypatch) -> None:  # noqa: F811
    _send_no_profile(env, ui=("patient_name", "김순자"))
    _send_no_profile(env, ui=("phone", "010-2222-3333"))
    card = _send_no_profile(env, "강남메디내과 내과 14:00 예약해줘")  # 새 요청 - 저장 초안을 이어 쓰지 않고 다시 만든다
    assert card["ui"].field == "reservation_confirm"
    flow = env.svc._last_flow_state(env.db, "user-1", "s1")
    draft = flow["pendingAction"]["draft"]
    assert draft["patient_name"] == "김순자" and draft["phone"] == "010-2222-3333"
    raw = env.db.query(ChatLog).order_by(ChatLog.id.desc()).first().context_json
    assert "storedContact" not in raw  # 챗 기록에 다시 저장하지 않는다
    seen = _router(env, monkeypatch)
    _send_no_profile(env, "고마워요")
    users = [h["content"] for h in seen.histories[-1] if h["role"] == "user"]
    assert "김순자" not in users and "010-2222-3333" not in users  # 이력(Claude)에는 여전히 없다


def test_contact_from_other_session_is_not_used(env) -> None:  # noqa: F811
    payload = ChatRequest(userId="user-1", sessionId="other", context=ChatContext(location="서울 강남구"),
                          uiSelection=UiSelection(field="phone", value="010-9999-0000"))
    env.svc.process_message(env.db, payload)
    out = _send_no_profile(env, "강남메디내과 내과 14:00 예약해줘")
    assert out["ui"].field == "patient_name"  # 다른 세션의 값은 쓰지 않고 이름부터 묻는다


# ---- L-3 ---------------------------------------------------------------------------------------------------


def test_reservation_state_uses_user_context_ward_when_router_has_no_ward(env) -> None:  # noqa: F811
    ctx = {"sessionId": "s1", "userContext": {"wardId": "w1"},
           "upstreamMeta": {"router": {"route": "reservation", "normalized_message": "병원 예약: 강남메디내과"}}}
    env.db.add(ChatLog(chat_no="CHAT-old", user_id="user-1", message="예약", context_json=json.dumps(ctx), reply="r"))
    env.db.commit()
    assert cs.ChatService._last_reservation_state(env.db, "user-1", "s1", "w1") == "병원 예약: 강남메디내과"
    assert cs.ChatService._last_reservation_state(env.db, "user-1", "s1", "") is None
