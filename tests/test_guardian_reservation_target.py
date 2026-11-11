"""보호자 챗봇 예약 대상 고르기(2026-11-11).

보호자 챗봇 예약은 피보호자만 대상이다. wardId 없이 예약 흐름이 시작되면 예약하지 않고 "누구의 병원 예약인가요?"를 묻고,
고른 피보호자(이 턴 wardId 가 그 값일 때만)로 예약을 시작한다. wardId 없이 가족 이야기면 MedGemma 에 나이·성별을 넘기지 않는다.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.chat_log import ChatLog
from app.schemas.chat_schema import ChatContext, ChatRequest, GuardianWard, UiSelection
from app.services import chat_service as cs
from app.services import follow_up_questions, reservation_target
from app.services import router_failure_monitor as rfm
from app.services.medical_llm_service import MedicalLlmService as M

WARDS = [GuardianWard(wardId="w1", name="김순자", relation="어머니"), GuardianWard(wardId="w2", name="이철수", relation="아버지")]


class _Router:
    def __init__(self) -> None:
        self.out: dict[str, Any] = {}

    def route(self, message, history, ctx, prev_reservation=None) -> dict[str, Any]:  # noqa: ARG002
        route = "reservation" if "예약" in message else "symptom"
        return {"route": route, "normalized_message": f"병원 예약: {message}" if route == "reservation" else message,
                "symptom_summary": "" if route == "reservation" else message, "emergency": False, "self_harm": False,
                "cardiac_arrest": False, "urgent_condition": "", "reason": "t", "engine": "fake", **self.out}


class _Llm:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, message, ctx, history, intent=None, **kwargs) -> dict[str, Any]:  # noqa: ARG002
        self.calls.append({"message": message, "ctx": dict(ctx), "intent": intent, "proxy": kwargs.get("proxy")})
        emergency = intent == "emergency_guidance"
        return {"reply": "쉬시면서 물을 자주 드세요.", "riskLevel": "high" if emergency else "medium",
                "recommendedAction": "guardian_contact", "reservationRequired": False, "intent": intent, "engine": "medgemma",
                "modelName": "m", "summary": None, "possibleCauses": [], "homeCare": [], "visitHospitalIf": [],
                "emergencyWarning": [M.EMERGENCY_WARNING] if emergency else []}


class _Orch:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def handle(self, req, meta):
        ward = (req.context.wardId or "") if req.context else ""
        self.calls.append({"message": req.message, "raw": meta.get("rawMessage"), "ward": ward,
                           "ui": (req.uiSelection.field, req.uiSelection.value) if req.uiSelection else None})
        return {"reply": "어느 지역에서 병원을 찾을까요?", "riskLevel": "medium", "recommendedAction": "guardian_contact",
                "reservationRequired": True, "intent": "hospital_reservation", "engine": "claude", "modelName": "fake",
                "type": "message", "ui": None, "tool": None, "data": None, "meta": dict(meta)}


class _Cred:
    def __init__(self) -> None:
        self.users: list[str] = []

    def ensure_credential(self, db, **kwargs):  # noqa: ARG002
        self.users.append(kwargs["user_id"])
        return "sbk_x.y"


@pytest.fixture()
def env(monkeypatch):
    engine = create_engine(f"sqlite:///{tempfile.mkdtemp(prefix='sb_target_')}/t.db")
    ChatLog.__table__.create(bind=engine)
    db = sessionmaker(bind=engine)()
    h = SimpleNamespace(db=db, router=_Router(), llm=_Llm(), orch=_Orch(), cred=_Cred())
    monkeypatch.setattr(cs, "get_chat_router_service", lambda: h.router)
    monkeypatch.setattr(cs, "get_medical_llm_service", lambda: h.llm)
    monkeypatch.setattr(cs, "get_reservation_orchestrator", lambda: h.orch)
    monkeypatch.setattr(cs, "get_reservation_credential_service", lambda: h.cred)
    monkeypatch.setattr(cs, "get_reservation_api_client", lambda: object())
    monkeypatch.setattr(cs, "get_router_failure_monitor", lambda: rfm.RouterFailureMonitor(600, 5))
    monkeypatch.setattr(cs, "_medgemma_enabled", lambda: True)
    monkeypatch.setattr(follow_up_questions, "_model_questions", lambda *a, **k: [])
    h.svc = cs.ChatService()
    h.svc._settings = SimpleNamespace(has_reservation_account_secret=True, claude_model_name="fake",
                                      follow_up_enabled=True, follow_up_timeout_sec=2)
    yield h
    db.close()


_UNSET = object()


def _send(env, message: str = "", *, ui: tuple[str, str] | None = None, ward: str | None = None, wards: Any = _UNSET,
          role: str = "GUARDIAN", session: str = "s1") -> dict[str, Any]:
    ctx = ChatContext(
        name="보호자", role=role, age=45, gender="FEMALE", wardId=ward,
        wardName={"w1": "김순자", "w2": "이철수"}.get(ward or ""),
        guardianWards=WARDS if wards is _UNSET else wards,
    )
    return env.svc.process_message(env.db, ChatRequest(
        userId="g-1", message=message, sessionId=session, context=ctx,
        uiSelection=UiSelection(field=ui[0], value=ui[1]) if ui else None))


def _ctx(env, out: dict[str, Any]) -> dict[str, Any]:
    return json.loads(env.db.query(ChatLog).filter(ChatLog.id == out["chatLogId"]).one().context_json)


def _flat(text: str) -> str:
    """답 모양 처리(한 줄 한 문장)를 지운 비교용 문장."""
    return " ".join(text.split())


_ASK_NORMALIZED: list[str] = []


def _ui(out: dict[str, Any]) -> dict[str, Any]:
    ui = out["ui"]
    return ui if isinstance(ui, dict) else ui.model_dump()


# ---- 대상 질문 조건 -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("wards", [WARDS, WARDS[:1]], ids=["two", "one"])
def test_reservation_without_ward_asks_who_first(env, wards, caplog) -> None:
    caplog.set_level(logging.INFO)
    out = _send(env, "내과 예약해줘", wards=wards)
    ui = _ui(out)
    assert out["type"] == "ui" and ui["field"] == reservation_target.TARGET_FIELD
    assert out["reply"] == reservation_target.QUESTION
    assert [o["value"] for o in ui["options"]] == [w.wardId for w in wards]
    assert ui["options"][0]["label"] == "김순자(어머니)"
    assert "__self__" not in [o["value"] for o in ui["options"]]  # 보호자 본인 예약은 없다
    assert env.orch.calls == [] and env.cred.users == []  # 예약 서비스·예약 키를 부르지 않는다
    assert out["intent"] == "hospital_reservation" and out["suggestedQuestions"] == []
    # 이름은 로그·userContext 에 남기지 않는다(선택지 ui 만).
    assert "김순자" not in caplog.text
    ask = _ctx(env, out)["upstreamMeta"]["reservationTargetAsk"]
    assert ask["wardIds"] == [w.wardId for w in wards] and "김순자" not in json.dumps(ask, ensure_ascii=False)
    assert "guardianWards" not in _ctx(env, out)["userContext"]


def test_no_connected_ward_gets_notice_without_booking(env) -> None:
    out = _send(env, "내과 예약해줘", wards=[])
    assert _flat(out["reply"]) == reservation_target.NO_WARD_REPLY
    assert out["intent"] == "general_chat" and out["ui"] is None and out["suggestedQuestions"] == []
    assert env.orch.calls == [] and env.cred.users == []


def test_old_be_without_guardian_wards_keeps_previous_behaviour(env) -> None:
    out = _send(env, "내과 예약해줘", wards=None)
    assert env.orch.calls and env.orch.calls[0]["ward"] == ""
    assert env.cred.users == ["g-1"]
    assert out["reply"] == "어느 지역에서 병원을 찾을까요?"


def test_ward_already_chosen_does_not_ask(env) -> None:
    _send(env, "내과 예약해줘", ward="w1")
    assert env.orch.calls[0]["ward"] == "w1" and env.cred.users == ["w1"]


def test_ward_route_ignores_injected_guardian_wards(env) -> None:
    _send(env, "내과 예약해줘", role="WARD")
    assert env.orch.calls and env.orch.calls[0]["ward"] == ""


def test_cancel_request_also_asks_but_how_to_question_does_not(env) -> None:
    out = _send(env, "예약 취소할래")
    assert _ui(out)["field"] == reservation_target.TARGET_FIELD and env.orch.calls == []
    _send(env, "예약 취소는 어떻게 해요?", session="s2")
    assert len(env.orch.calls) == 1


# ---- 선택 턴 ---------------------------------------------------------------------------------------------------


def test_selected_ward_starts_the_reservation_as_that_ward(env) -> None:
    ask = _send(env, "내과 예약해줘")
    _ASK_NORMALIZED[:] = [_ctx(env, ask)["upstreamMeta"]["router"]["normalized_message"]]
    out = _send(env, ui=(reservation_target.TARGET_FIELD, "w2"), ward="w2")
    call = env.orch.calls[-1]
    assert call["ward"] == "w2" and call["ui"] is None
    # 대상 질문 기록에 남은 규격화 문장(router.normalized_message)과 시작 문장을 그대로 넘긴다.
    assert call["message"] == _ASK_NORMALIZED[0] and call["raw"] == "내과 예약해줘"
    assert env.cred.users == ["w2"]  # 그 피보호자 계정으로
    assert _ctx(env, out)["upstreamMeta"]["router"]["wardId"] == "w2"  # 다음 턴(wardId=w2)이 이어 쓴다


def test_selection_without_ward_in_context_asks_again_and_keeps_the_request(env) -> None:
    _send(env, "내과 예약해줘")
    out = _send(env, ui=(reservation_target.TARGET_FIELD, "w1"))  # FE 가 wardId 를 싣지 않음
    assert _ui(out)["field"] == reservation_target.TARGET_FIELD and out["reply"].endswith(reservation_target.QUESTION)
    assert "골라" in out["reply"] and env.orch.calls == []
    _send(env, ui=(reservation_target.TARGET_FIELD, "w1"), ward="w1")
    assert env.orch.calls[-1]["raw"] == "내과 예약해줘" and env.orch.calls[-1]["ward"] == "w1"


def test_selection_with_a_different_ward_in_context_is_not_booked(env) -> None:
    _send(env, "내과 예약해줘")
    _send(env, ui=(reservation_target.TARGET_FIELD, "w1"), ward="w2")
    assert env.orch.calls == []


@pytest.mark.parametrize("case", ["unknown", "expired", "other_session", "no_question"])
def test_stale_selection_does_not_book(env, case, monkeypatch) -> None:
    if case != "no_question":
        _send(env, "내과 예약해줘")
    if case == "expired":
        real = time.time
        monkeypatch.setattr(reservation_target.time, "time", lambda: real() + reservation_target.TARGET_TTL_SECONDS + 5)
    value = "w9" if case == "unknown" else "w1"
    out = _send(env, ui=(reservation_target.TARGET_FIELD, value), ward=value,
                session="s2" if case == "other_session" else "s1")
    assert _flat(out["reply"]) == reservation_target.STALE_REPLY and out["intent"] == "general_chat"
    assert env.orch.calls == [] and out["suggestedQuestions"] == []


def test_button_start_is_replayed_after_selection(env) -> None:
    out = _send(env, ui=("department", "내과"))
    assert _ui(out)["field"] == reservation_target.TARGET_FIELD and env.orch.calls == []
    _send(env, ui=(reservation_target.TARGET_FIELD, "w1"), ward="w1")
    call = env.orch.calls[-1]
    assert call["ui"] == ("department", "내과") and call["ward"] == "w1" and not call["raw"]


def test_later_turns_with_the_ward_continue_without_asking(env) -> None:
    _send(env, "내과 예약해줘")
    _send(env, ui=(reservation_target.TARGET_FIELD, "w1"), ward="w1")
    _send(env, ui=("location", "서울 강남구"), ward="w1")
    assert len(env.orch.calls) == 2 and env.orch.calls[-1]["ward"] == "w1"


# ---- 응급·자해가 먼저 ----------------------------------------------------------------------------------------------


def test_emergency_beats_the_target_question(env) -> None:
    out = _send(env, "숨을 못 쉬어요 병원 예약해줘")
    assert out["riskLevel"] == "high" and out["ui"] is None and env.orch.calls == []


def test_self_harm_beats_the_target_question(env) -> None:
    out = _send(env, "죽고 싶어요 병원 예약해줘")
    assert out["ui"] is None and "109" in out["reply"] and env.orch.calls == []


def test_emergency_while_the_question_is_open(env) -> None:
    _send(env, "내과 예약해줘")
    out = _send(env, "숨을 못 쉬겠어요")
    assert out["riskLevel"] == "high" and out["ui"] is None and env.orch.calls == []


def test_text_reservation_answer_to_the_question_asks_again(env) -> None:
    _send(env, "내과 예약해줘")
    out = _send(env, "엄마 예약이요")
    assert _ui(out)["field"] == reservation_target.TARGET_FIELD and "골라" in out["reply"] and env.orch.calls == []


# ---- MedGemma 프로필 ---------------------------------------------------------------------------------------------


def test_family_subject_without_ward_drops_the_speaker_profile(env) -> None:
    _send(env, "어머니가 어제부터 계속 어지럽고 속이 메스껍다고 하세요")
    assert env.llm.calls[-1]["ctx"] == {} and env.llm.calls[-1]["proxy"] is True


def test_own_consultation_keeps_the_profile(env) -> None:
    _send(env, "제가 어제부터 계속 어지럽고 속이 메스꺼워요")
    assert env.llm.calls[-1]["ctx"] == {"age": 45, "gender": "FEMALE"}


def test_ward_consultation_keeps_the_ward_profile(env) -> None:
    _send(env, "어머니가 어제부터 계속 어지럽고 속이 메스껍다고 하세요", ward="w1")
    assert env.llm.calls[-1]["ctx"] == {"age": 45, "gender": "FEMALE"}  # wardId 가 있으면 BE 가 피보호자 값을 넣는다


def test_option_label_shapes() -> None:
    assert reservation_target.option_label(GuardianWard(wardId="a", name="김순자")) == "김순자"
    assert reservation_target.option_label(GuardianWard(wardId="a")) == "피보호자"
    assert reservation_target.option_label(GuardianWard(wardId="a", name="가" * 40, relation="어머니")) == "가" * 20 + "(어머니)"


def test_duplicate_wards_are_listed_once() -> None:
    req = ChatRequest(userId="g", message="x", context=ChatContext(role="GUARDIAN", guardianWards=[WARDS[0], WARDS[0]]))
    assert [w.wardId for w in reservation_target.guardian_wards(req)] == ["w1"]
