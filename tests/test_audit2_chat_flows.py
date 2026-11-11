"""2차 전체 점검(2026-11-10) 묶음 C - 예약·대리 상담·단계 되묻기 흐름의 기능 간 상호작용.

I-2 라우터 실패 중 예약 흐름 유지 / I-3 다른 피보호자 행으로 초안 복사 금지 + 삭제 범위 / I-4 더블 탭에도 증상 문장 고정 /
I-5 질문 낱말을 부정으로 되받은 답은 119 로 가지 않음 / I-6 발급 중 삭제 재확인 / I-7 턴 시작 뒤 삭제면 기록 안 씀 /
I-8 MedGemma 꺼짐이면 되묻기 시작 안 함 / I-9 버튼 턴의 응급·109 줄·피보호자 구분 / R-2 긴 호출 전 DB 연결 반납.
라우터 정상·실패 두 경로를 모두 고정한다. 실제 라우터·MedGemma·예약 서비스는 부르지 않는다."""
from __future__ import annotations

import json
import tempfile
import time
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.base import Base
from app.models.chat_log import ChatLog
from app.models.reservation_credential import ReservationCredential
from app.schemas.chat_schema import ChatContext, ChatRequest, UiSelection
from app.services import chat_service as cs
from app.services import follow_up_questions
from app.services import reservation_credential_service as rcs
from app.services import router_failure_monitor as rfm
from app.services.claude_structured_client import StructuredCallError
from app.services.medical_llm_service import MedicalLlmService as M
from app.services.reservation_confirm_flow import _PENDING_TTL_SECONDS, draft_to_dict
from app.services.reservation_intake import ReservationDraft
from tests.reservation_flow_helpers import FUTURE, FakeLoader, FakeReservation, make_orchestrator, make_request

CRISIS_109 = "힘드시면 언제든 자살예방 상담전화 109(24시간)에 이야기해 보세요."
EMERGENCY = "가슴이 너무 아프고 숨을 못 쉬겠어요"


class _FailRouter:
    def route(self, message, history, ctx, prev_reservation=None) -> dict[str, Any]:  # noqa: ARG002
        raise StructuredCallError("api_400")


class _OkRouter:
    def __init__(self, **out: Any) -> None:
        self.out = out
        self.in_transaction: list[bool] = []
        self.db: Any = None

    def route(self, message, history, ctx, prev_reservation=None) -> dict[str, Any]:  # noqa: ARG002
        if self.db is not None:
            self.in_transaction.append(self.db.in_transaction())
        return {"route": "symptom", "normalized_message": message, "symptom_summary": "", "emergency": False, "self_harm": False,
                "cardiac_arrest": False, "urgent_condition": "", "reason": "t", "engine": "fake", **self.out}


class _Llm:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.on_generate: Any = None

    def generate(self, message, ctx, history, intent=None, **kwargs) -> dict[str, Any]:  # noqa: ARG002
        self.calls.append({"message": message, "intent": intent})
        if self.on_generate:
            self.on_generate()
        emergency = intent == "emergency_guidance"
        return {"reply": "쉬시면서 물을 자주 드세요.", "riskLevel": "high" if emergency else "medium",
                "recommendedAction": "guardian_contact", "reservationRequired": False, "intent": intent, "engine": "medgemma",
                "modelName": "m", "summary": None, "possibleCauses": [], "homeCare": [], "visitHospitalIf": [],
                "emergencyWarning": [M.EMERGENCY_WARNING] if emergency else []}


class _Orch:
    """예약 오케스트레이터 자리 - 넘겨받은 문장·meta 만 기록하고 다음 턴이 이어 쓸 초안을 남긴다."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.on_handle: Any = None

    def handle(self, req, meta):
        self.calls.append((req.message, dict(meta)))
        if self.on_handle:
            self.on_handle()
        ward = (req.context.wardId or "") if req.context else ""
        return {"reply": "어느 지역에서 병원을 찾을까요?", "riskLevel": "medium", "recommendedAction": "guardian_contact",
                "reservationRequired": True, "intent": "hospital_reservation", "engine": "claude", "modelName": "fake",
                "type": "message", "ui": None, "tool": None, "data": None,
                "meta": {**meta, "flowState": _flow_state(ward)}}


class _Cred:
    def __init__(self) -> None:
        self.key: str | None = "sbk_x.y"

    def ensure_credential(self, db, **kwargs):  # noqa: ARG002
        return self.key


def _flow_state(ward: str = "", age: int = 0) -> dict[str, Any]:
    draft = ReservationDraft(message="", department="내과", patient_name="홍길동", phone="010-1234-5678")
    return {"draft": draft_to_dict(draft), "replaces": None, "wardId": ward, "createdAt": int(time.time()) - age}


@pytest.fixture()
def env(monkeypatch):
    engine = create_engine(f"sqlite:///{tempfile.mkdtemp(prefix='sb_audit2c_')}/t.db")
    ChatLog.__table__.create(bind=engine)
    db = sessionmaker(bind=engine)()
    h = SimpleNamespace(db=db, router=_OkRouter(), llm=_Llm(), orch=_Orch(), cred=_Cred(), medgemma=True)
    monkeypatch.setattr(cs, "get_chat_router_service", lambda: h.router)
    monkeypatch.setattr(cs, "get_medical_llm_service", lambda: h.llm)
    monkeypatch.setattr(cs, "get_reservation_orchestrator", lambda: h.orch)
    monkeypatch.setattr(cs, "get_reservation_credential_service", lambda: h.cred)
    monkeypatch.setattr(cs, "get_reservation_api_client", lambda: object())
    monkeypatch.setattr(cs, "get_router_failure_monitor", lambda: rfm.RouterFailureMonitor(600, 5))
    monkeypatch.setattr(cs, "_medgemma_enabled", lambda: h.medgemma)
    monkeypatch.setattr(follow_up_questions, "_model_questions", lambda *a, **k: [])
    h.svc = cs.ChatService()
    h.svc._settings = SimpleNamespace(has_reservation_account_secret=True, claude_model_name="fake",
                                      follow_up_enabled=True, follow_up_timeout_sec=2)
    yield h
    db.close()


def _send(env, message: str = "", *, ui: tuple[str, str] | None = None, session: str = "s1", ward: str | None = None,
          user: str = "user-1", role: str | None = "WARD") -> dict[str, Any]:
    # 단계 되묻기는 피보호자 본인(WARD)만(2026-11-11) - 이 파일의 대화 흐름은 기본 WARD 로 본다.
    ctx = ChatContext(name="홍길동", wardId=ward, role=role) if ward else ChatContext(name="홍길동", role=role)
    return env.svc.process_message(env.db, ChatRequest(
        userId=user, message=message, sessionId=session, context=ctx,
        uiSelection=UiSelection(field=ui[0], value=ui[1]) if ui else None))


def _ctx(env, out: dict[str, Any]) -> dict[str, Any]:
    return json.loads(env.db.query(ChatLog).filter(ChatLog.id == out["chatLogId"]).one().context_json)


def _seed_flow(env, *, session: str = "s1", ward: str = "", flow_ward: str | None = None, age: int = 0,
               ui_field: str = "location", user: str = "user-1", pending: bool = False, chat_no: str = "seed") -> None:
    """직전 턴이 예약 흐름(지역을 묻는 턴)이었던 기록."""
    state = _flow_state(ward if flow_ward is None else flow_ward, age)
    meta: dict[str, Any] = {"router": {"route": "reservation", "normalized_message": "병원 예약: 내과", "wardId": ward}}
    meta["pendingAction" if pending else "flowState"] = {**state, **({"kind": "create", "token": "t"} if pending else {})}
    ctx = {"sessionId": session, "userContext": {"wardId": ward} if ward else {}, "ui": {"kind": "text", "field": ui_field},
           "upstreamMeta": meta}
    env.db.add(ChatLog(chat_no=chat_no, user_id=user, message="내과 예약하고 싶어요", reply="어느 지역에서 병원을 찾을까요?",
                       context_json=json.dumps(ctx, ensure_ascii=False), risk_level="medium",
                       recommended_action="guardian_contact", reservation_required=True))
    env.db.commit()


# ---- I-2 라우터 실패 중 예약 흐름 유지 -------------------------------------------------------------------------


@pytest.mark.parametrize("message", ["서울 강남구", "내일 오후 3시", "열한 시로 해줘"])
def test_router_failure_keeps_free_text_in_the_reservation_flow(env, message: str) -> None:
    _seed_flow(env)
    env.router = _FailRouter()
    out = _send(env, message)
    assert len(env.orch.calls) == 1 and env.llm.calls == []
    routed, meta = env.orch.calls[0]
    assert routed == f"병원 예약: 내과 / {message}" and meta["rawMessage"] == message
    assert meta["awaitingField"] == "location" and "flowState" in meta["storedFlow"]
    ctx = _ctx(env, out)
    assert ctx["upstreamMeta"]["router"]["route"] == "reservation" and ctx["upstreamMeta"]["routerFailed"] is True
    assert "awaitingField" not in ctx["upstreamMeta"]  # 기록에는 남기지 않는다


def test_router_failure_keeps_flow_for_pending_confirm_card(env) -> None:
    _seed_flow(env, pending=True, ui_field="reservation_confirm")
    env.router = _FailRouter()
    _send(env, "내일 오후 3시")
    assert len(env.orch.calls) == 1


def test_router_ok_turn_is_not_pinned_to_reservation(env) -> None:
    # 라우터가 정상인 턴은 라우터 판단 그대로 - 직전 예약 흐름이 있어도 고정하지 않는다
    _seed_flow(env)
    env.router = _OkRouter(route="general")
    out = _send(env, "서울 강남구")
    assert env.orch.calls == [] and len(env.llm.calls) == 1
    assert _ctx(env, out)["upstreamMeta"]["router"]["route"] == "general"


@pytest.mark.parametrize("message", [EMERGENCY, "죽고 싶어요", "다 끝내고 싶어"])
def test_router_failure_emergency_and_self_harm_still_win_over_the_flow(env, message: str) -> None:
    _seed_flow(env)
    env.router = _FailRouter()
    out = _send(env, message)
    assert env.orch.calls == [] and out["riskLevel"] == "high"
    assert ("119" in out["reply"]) or ("109" in out["reply"])


@pytest.mark.parametrize("seed", [
    {"age": _PENDING_TTL_SECONDS + 60},  # 유효 시간 지남
    {"session": "other"},  # 다른 세션
    {"flow_ward": "ward02"},  # 초안이 다른 피보호자 것
    {"ward": "ward01"},  # 직전 기록이 다른 피보호자 상담
])
def test_router_failure_without_a_valid_flow_is_not_pinned(env, seed: dict[str, Any]) -> None:
    _seed_flow(env, **seed)
    env.router = _FailRouter()
    _send(env, "서울 강남구")
    assert env.orch.calls == []


@pytest.mark.parametrize("message", ["고마워요", "됐어요 그만할게요"])
def test_router_failure_closing_words_leave_the_flow(env, message: str) -> None:
    _seed_flow(env)
    env.router = _FailRouter()
    _send(env, message)
    assert env.orch.calls == []


class _FailingLoader(FakeLoader):
    def generate_structured_text(self, *a, **k):  # noqa: ARG002
        raise RuntimeError("credit")


def _region_turn(awaiting: str | None, text: str = "수원시 영통구") -> tuple[dict[str, Any], FakeReservation]:
    fake = FakeReservation()
    orch = make_orchestrator(fake)
    orch._loader = _FailingLoader()
    stored = {"flowState": _flow_state()}
    meta: dict[str, Any] = {"storedFlow": stored, "rawMessage": text}
    if awaiting:
        meta["awaitingField"] = awaiting
    out = orch.handle(make_request(f"병원 예약: 내과 / {text}", context={"location": None}), meta)
    assert out is not None
    return out, fake


def test_typed_region_is_applied_when_router_and_extraction_both_failed() -> None:
    out, fake = _region_turn("location")
    assert fake.searched_regions and fake.searched_regions[0] == "수원시 영통구"
    assert out["meta"]["flowState"]["draft"]["location"] == "수원시 영통구"


def test_typed_region_needs_the_awaiting_hint_and_a_region_shaped_text() -> None:
    out, fake = _region_turn(None)  # 라우터가 정상인 턴(힌트 없음)은 지금 동작 그대로
    assert fake.searched_regions == [] and out["meta"]["flowState"]["draft"].get("location") is None
    out, fake = _region_turn("location", "잘 모르겠어요")
    assert fake.searched_regions == [] and out["meta"]["flowState"]["draft"].get("location") is None
    out, fake = _region_turn("date")
    assert fake.searched_regions == []


# ---- I-5 질문 낱말을 부정으로 되받은 답 -------------------------------------------------------------------------


def _routers() -> list[Any]:
    return [pytest.param(lambda: _OkRouter(), id="router-ok"), pytest.param(lambda: _FailRouter(), id="router-failed")]


@pytest.mark.parametrize("make_router", _routers())
@pytest.mark.parametrize(("symptom", "answer", "phrase"), [
    ("기침이 나요", "숨쉬기 힘들지 않아요", "숨쉬기는 괜찮아요"),
    ("숨이 좀 차요", "가슴이 조이진 않아요", "그런 증상은 없어요"),
])
def test_negated_question_words_do_not_force_119(env, make_router, symptom: str, answer: str, phrase: str) -> None:
    env.router = make_router()
    first = _send(env, symptom)
    assert _ctx(env, first)["upstreamMeta"].get("followupAsked") is True
    out = _send(env, answer)
    assert out["riskLevel"] != "high" and not out["emergencyWarning"]
    assert env.llm.calls[-1] == {"message": f"{symptom} / {phrase}", "intent": "medical_advice"}
    assert _ctx(env, out)["upstreamMeta"]["router"]["emergency"] is False


@pytest.mark.parametrize("make_router", _routers())
@pytest.mark.parametrize("answer", ["숨쉬기 힘들지 않은데 의식이 없어요", "아니요 근데 피를 토했어요", "숨쉬기 힘들어요"])
def test_other_emergency_words_in_the_answer_still_win(env, make_router, answer: str) -> None:
    env.router = make_router()
    _send(env, "기침이 나요")
    out = _send(env, answer)
    assert out["riskLevel"] == "high" and "119" in out["reply"]


def test_router_emergency_flag_wins_over_a_negated_answer(env) -> None:
    _send(env, "기침이 나요")
    env.router = _OkRouter(emergency=True)
    out = _send(env, "숨쉬기 힘들지 않아요")
    assert out["riskLevel"] == "high" and env.llm.calls[-1]["intent"] == "emergency_guidance"


def test_negated_words_outside_a_step_flow_are_unchanged(env) -> None:
    # 단계 되묻기 중이 아니면 키워드 응급 강제는 그대로다
    out = _send(env, "숨쉬기 힘들지 않아요")
    assert out["riskLevel"] == "high"


def test_router_failure_negated_reservation_word_does_not_start_a_reservation(env) -> None:
    env.router = _FailRouter()
    _send(env, "기침이 나요")
    out = _send(env, "예약은 안 했어요")
    assert env.orch.calls == [] and out["intent"] != "hospital_reservation"
    # 단계 되묻기 중이 아니면 키워드 분류 그대로(예약 낱말 → 예약 경로)
    _send(env, "예약은 안 했어요", session="s2")
    assert len(env.orch.calls) == 1


def test_router_failure_real_reservation_request_during_a_step_still_goes_to_reservation(env) -> None:
    env.router = _FailRouter()
    _send(env, "기침이 나요")
    _send(env, "내과 예약해 주세요")
    assert len(env.orch.calls) == 1


# ---- I-3 다른 피보호자 행으로 초안 복사 금지 + 삭제 범위 ---------------------------------------------------------


def test_flow_for_ward_keeps_only_the_current_wards_draft() -> None:
    a, b = _flow_state("ward01"), _flow_state("ward02")
    assert cs.ChatService._flow_for_ward({"flowState": a, "pendingAction": b}, "ward01") == {"flowState": a}
    assert cs.ChatService._flow_for_ward({"flowState": a}, "ward02") == {}
    assert cs.ChatService._flow_for_ward({"flowState": {"draft": {}}}, "") == {}  # wardId 가 없는 옛 상태
    assert cs.ChatService._flow_for_ward({"flowState": _flow_state("")}, "") == {"flowState": _flow_state("")}


@pytest.mark.parametrize(("next_ward", "carried"), [("ward01", True), ("ward02", False)])
def test_credential_failure_turn_does_not_copy_another_wards_draft(env, next_ward: str, carried: bool) -> None:
    _seed_flow(env, ward="ward01")
    env.cred.key = None  # 예약 계정 키를 못 받은 턴 - 직전 초안을 다음 턴으로 넘기는 경로
    out = _send(env, ui=("date", FUTURE), ward=next_ward)
    meta = _ctx(env, out)["upstreamMeta"]
    assert ("flowState" in meta) is carried
    assert "ward01" not in json.dumps(meta.get("flowState") or {}) or carried


def _row(db, chat_no: str, user: str, row_ward: str, flow_ward: str | None) -> None:
    meta: dict[str, Any] = {"router": {"route": "reservation", "wardId": row_ward}}
    if flow_ward is not None:
        meta["flowState"] = _flow_state(flow_ward)
    ctx = {"sessionId": "s1", "userContext": {"wardId": row_ward} if row_ward else {}, "upstreamMeta": meta}
    db.add(ChatLog(chat_no=chat_no, user_id=user, message="m", reply="r", context_json=json.dumps(ctx, ensure_ascii=False)))


def _delete_db():
    engine = create_engine(f"sqlite:///{tempfile.mkdtemp(prefix='sb_audit2c_del_')}/t.db")
    ChatLog.__table__.create(bind=engine)
    db = sessionmaker(bind=engine)()
    _row(db, "b-with-a-draft", "guard1", "ward02", "ward01")  # B 표시 행에 A 초안이 복사된 옛 행
    _row(db, "b-normal", "guard1", "ward02", "ward02")
    _row(db, "b-plain", "guard1", "ward02", None)
    _row(db, "a-normal", "guard1", "ward01", "ward01")
    _row(db, "g-self", "guard1", "", "")
    _row(db, "other-guardian-b", "guard2", "ward02", "ward02")
    db.commit()
    return db


def _left(db) -> list[str]:
    return sorted(r.chat_no for r in db.query(ChatLog).all())


def test_proxy_delete_also_removes_rows_holding_that_wards_draft() -> None:
    db = _delete_db()
    assert cs.ChatService.delete_proxy_logs(db, "guard1", "ward01") == 2
    # 다른 피보호자의 정상 행·보호자 본인 행·다른 보호자 행은 남는다
    assert _left(db) == ["b-normal", "b-plain", "g-self", "other-guardian-b"]


def test_ward_withdrawal_delete_also_removes_rows_holding_that_wards_draft() -> None:
    db = _delete_db()
    assert cs.ChatService.delete_logs(db, "ward01") == (0, 2)
    assert _left(db) == ["b-normal", "b-plain", "g-self", "other-guardian-b"]
    assert cs.ChatService._proxy_ward_ids(json.dumps({"upstreamMeta": {"pendingAction": {"wardId": "ward09"}}})) == {"ward09"}


# ---- I-4 더블 탭(같은 답이 두 번 저장) ---------------------------------------------------------------------------


def _duplicate_latest_row(env) -> None:
    last = env.db.query(ChatLog).order_by(ChatLog.id.desc()).first()
    env.db.add(ChatLog(chat_no=f"{last.chat_no}-dup", user_id=last.user_id, message=last.message, reply=last.reply,
                       context_json=last.context_json, risk_level=last.risk_level,
                       recommended_action=last.recommended_action, reservation_required=last.reservation_required))
    env.db.commit()


@pytest.mark.parametrize("make_router", _routers())
def test_double_saved_answer_does_not_shift_the_symptom_sentence(env, make_router) -> None:
    env.router = make_router()
    first = _send(env, "머리가 아파요")
    assert _ctx(env, first)["upstreamMeta"]["followupStep"]["anchorLogId"] == 0
    second = _send(env, "아니요")
    assert _ctx(env, second)["upstreamMeta"]["followupStep"]["anchorLogId"] == 0
    _duplicate_latest_row(env)  # 같은 직전 기록을 읽은 두 턴이 둘 다 저장된 모양
    _send(env, "오늘부터요")
    _send(env, "아니요")
    assert env.llm.calls[-1]["message"] == "머리가 아파요 / 갑자기 아주 심한 건 아니에요 / 오늘부터요 / 열은 없어요"


def test_anchor_points_after_earlier_rows_of_the_same_user(env) -> None:
    _send(env, "안녕하세요", session="s0")  # 다른 세션의 앞선 기록
    before = env.db.query(ChatLog).order_by(ChatLog.id.desc()).first().id
    first = _send(env, "배가 아파요")
    assert _ctx(env, first)["upstreamMeta"]["followupStep"]["anchorLogId"] == before
    _send(env, "아니요")
    _duplicate_latest_row(env)
    _send(env, "며칠 됐어요")
    _send(env, "네")
    assert env.llm.calls[-1]["message"].startswith("배가 아파요 / 참을 만해요 / 며칠 됐어요")


def test_legacy_step_state_without_anchor_still_works(env) -> None:
    _send(env, "머리가 아파요")
    for _ in range(2):
        last = env.db.query(ChatLog).order_by(ChatLog.id.desc()).first()
        ctx = json.loads(last.context_json)
        ctx["upstreamMeta"]["followupStep"].pop("anchorLogId", None)  # 배포 전에 저장된 옛 형식
        last.context_json = json.dumps(ctx, ensure_ascii=False)
        env.db.commit()
        _send(env, "아니요" if _ == 0 else "오늘부터요")
    _send(env, "아니요")
    assert env.llm.calls[-1]["message"] == "머리가 아파요 / 갑자기 아주 심한 건 아니에요 / 오늘부터요 / 열은 없어요"


# ---- I-6 발급 중 삭제 재확인 -------------------------------------------------------------------------------------


class _IssuingClient:
    def __init__(self, purge_during_issue: bool) -> None:
        self._settings = SimpleNamespace(has_reservation_account_secret=True, reservation_account_secret="s")
        self.purge_during_issue = purge_during_issue
        self.revoked: list[int] = []

    def has_valid_reservation_key(self, key):  # noqa: ARG002
        return False

    def login_account(self, *a):  # noqa: ARG002
        return {"httpStatus": 200, "data": {"accessToken": "tok"}}

    def issue_api_key(self, token, name):  # noqa: ARG002
        if self.purge_during_issue:
            rcs.mark_purged("ward01")  # 발급을 기다리는 사이에 회원 데이터 삭제가 끝났다
        return {"httpStatus": 201, "data": {"apiKey": "sbk_new.secret"}}

    def list_api_keys(self, token):  # noqa: ARG002
        return {"httpStatus": 200, "data": {"keys": [
            {"id": 7, "name": "AI-ward01", "isActive": True, "keyPrefix": "new"},
            {"id": 8, "name": "web", "isActive": True, "keyPrefix": "web"},
        ]}}

    def revoke_api_key(self, token, key_id):  # noqa: ARG002
        self.revoked.append(key_id)
        return {"httpStatus": 200, "data": {}}


def _cred_session():
    engine = create_engine("sqlite+pysqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_purge_during_issue_discards_the_new_key() -> None:
    db, client = _cred_session(), _IssuingClient(purge_during_issue=True)
    assert rcs.ReservationCredentialService().ensure_credential(db, user_id="ward01", client=client) is None
    assert db.query(ReservationCredential).count() == 0
    assert client.revoked == [7]  # 방금 발급한 AI 키만 폐기(다른 이름의 키는 그대로)


def test_issue_without_purge_still_saves_the_key() -> None:
    db, client = _cred_session(), _IssuingClient(purge_during_issue=False)
    assert rcs.ReservationCredentialService().ensure_credential(db, user_id="ward01", client=client) == "sbk_new.secret"
    assert db.query(ReservationCredential).count() == 1 and client.revoked == []


# ---- I-7 턴 시작 뒤에 표시된 삭제 --------------------------------------------------------------------------------


def test_purged_since_ignores_the_hold_time_and_entries_outlive_it(monkeypatch) -> None:
    now = {"t": 1000.0}
    monkeypatch.setattr(rcs.time, "monotonic", lambda: now["t"])
    turn_started = now["t"] - 50
    rcs.mark_purged("user-1")
    rcs.mark_proxy_purged("guard1", "ward01")
    assert rcs.purged_since("user-1", turn_started) and not rcs.purged_since("user-1", now["t"] + 1)
    assert rcs.proxy_purged_since("guard1", "ward01", turn_started) and not rcs.proxy_purged_since("guard1", "ward02", turn_started)
    assert not rcs.purged_since("nobody", turn_started)
    now["t"] += rcs._PURGE_HOLD_SEC + 100  # 보류 시간은 지났다
    rcs.mark_purged("other")  # 목록 정리가 돌아도 진행 중 턴이 물을 수 있는 동안은 남는다
    assert not rcs.recently_purged("user-1") and rcs.purged_since("user-1", turn_started)
    now["t"] += rcs._TURN_TRACK_SEC
    rcs.mark_purged("other")
    assert "user-1" not in rcs._PURGED and rcs._TURN_TRACK_SEC <= 600


def test_turn_that_outlives_the_hold_does_not_write_a_log(env, monkeypatch) -> None:
    # 보류 판단(시간 기준)이 이미 풀린 것으로 두고, 턴 도중에 표시된 삭제만으로 기록을 쓰지 않는지 본다
    monkeypatch.setattr(cs, "recently_purged", lambda *a: False)
    monkeypatch.setattr(cs, "recently_proxy_purged", lambda *a: False)
    env.router = _OkRouter(route="general")
    env.llm.on_generate = lambda: rcs.mark_purged("user-1")
    out = _send(env, "안녕하세요")
    assert out["chatLogId"] is None and env.db.query(ChatLog).count() == 0
    env.llm.on_generate = None
    assert _send(env, "안녕하세요")["chatLogId"] is not None  # 턴 시작 전에 표시된 삭제는 이 판단에 걸리지 않는다


def test_proxy_purge_during_a_proxy_turn_does_not_write_a_log(env, monkeypatch) -> None:
    monkeypatch.setattr(cs, "recently_purged", lambda *a: False)
    monkeypatch.setattr(cs, "recently_proxy_purged", lambda *a: False)
    env.router = _OkRouter(route="general")
    env.llm.on_generate = lambda: rcs.mark_proxy_purged("user-1", "ward01")
    assert _send(env, "안녕하세요", ward="ward01")["chatLogId"] is None
    assert _send(env, "안녕하세요", ward="ward02", session="s2")["chatLogId"] is not None


# ---- I-8 MedGemma 꺼짐 -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("make_router", _routers())
def test_followup_is_not_started_while_medgemma_is_off(env, make_router) -> None:
    env.router = make_router()
    env.medgemma = False
    out = _send(env, "머리가 아파요")
    assert "followupAsked" not in _ctx(env, out)["upstreamMeta"] and len(env.llm.calls) == 1
    env.medgemma = True
    out = _send(env, "머리가 아파요", session="s2")
    assert _ctx(env, out)["upstreamMeta"]["followupAsked"] is True and len(env.llm.calls) == 1


def test_followup_in_progress_continues_after_medgemma_is_turned_off(env) -> None:
    _send(env, "머리가 아파요")
    env.medgemma = False
    out = _send(env, "아니요")
    assert _ctx(env, out)["upstreamMeta"]["followupStep"]["index"] == 1 and env.llm.calls == []


def test_medgemma_enabled_reads_the_loader_switch(monkeypatch) -> None:
    monkeypatch.setattr(cs, "get_medgemma_loader", lambda: SimpleNamespace(enabled=False))
    assert cs._medgemma_enabled() is False
    monkeypatch.setattr(cs, "get_medgemma_loader", lambda: SimpleNamespace(enabled=True))
    assert cs._medgemma_enabled() is True

    def boom():
        raise RuntimeError("x")

    monkeypatch.setattr(cs, "get_medgemma_loader", boom)
    assert cs._medgemma_enabled() is True


# ---- I-9 버튼 턴 --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("make_router", [
    pytest.param(lambda: _OkRouter(route="reservation"), id="router-ok"), pytest.param(lambda: _FailRouter(), id="router-failed")])
def test_emergency_sentence_with_a_button_selection_goes_to_emergency(env, make_router) -> None:
    env.router = make_router()
    out = _send(env, EMERGENCY, ui=("date", FUTURE))
    assert env.orch.calls == [] and out["riskLevel"] == "high" and env.llm.calls[-1]["intent"] == "emergency_guidance"


def test_plain_sentence_with_a_button_selection_still_goes_to_reservation(env) -> None:
    env.router = _OkRouter(route="general")
    _send(env, "이 날로 할게요", ui=("date", FUTURE))
    assert len(env.orch.calls) == 1


def test_button_turn_right_after_a_crisis_reply_gets_the_109_line_once(env) -> None:
    crisis = _send(env, "죽고 싶어요")
    assert _ctx(env, crisis)["upstreamMeta"]["selfHarm"] is True
    out = _send(env, ui=("department", "내과"))
    assert out["reply"].startswith(CRISIS_109) and len(env.orch.calls) == 1
    assert _ctx(env, out)["upstreamMeta"]["crisisFollowup"] is True
    assert out["layout"]["urgent"] is True and out["layout"]["headline"] == CRISIS_109
    again = _send(env, ui=("date", FUTURE))
    assert CRISIS_109 not in again["reply"] and again["layout"] is None


def test_button_turn_without_a_previous_crisis_has_no_109_line(env) -> None:
    out = _send(env, ui=("department", "내과"))
    assert CRISIS_109 not in out["reply"]


@pytest.mark.parametrize(("next_ward", "carried"), [("ward01", True), ("ward02", False), (None, False)])
def test_crisis_mark_does_not_cross_to_another_wards_consult(env, next_ward: str | None, carried: bool) -> None:
    _send(env, "죽고 싶어요", ward="ward01")
    env.router = _OkRouter(route="general")
    out = _send(env, "오늘 점심은 드셨대요", ward=next_ward)
    assert out["reply"].startswith(CRISIS_109) is carried
    out = _send(env, ui=("department", "내과"), ward=next_ward, session="s1")
    assert CRISIS_109 not in out["reply"]


def test_previous_turn_meta_checks_the_ward_only_when_given() -> None:
    row = SimpleNamespace(context_json=json.dumps({"sessionId": "s1", "userContext": {"wardId": "ward01"},
                                                   "upstreamMeta": {"selfHarm": True}}))
    meta = cs.ChatService._previous_turn_meta
    assert meta(None, "u", "s1", latest=row)["selfHarm"] is True  # 옛 호출(피보호자 구분 없음)은 그대로
    assert meta(None, "u", "s1", latest=row, ward_id="ward01")["selfHarm"] is True
    assert meta(None, "u", "s1", latest=row, ward_id="ward02") == {} and meta(None, "u", "s1", latest=row, ward_id="") == {}
    assert meta(None, "u", "other", latest=row, ward_id="ward01") == {}


# ---- R-2 긴 호출 전 DB 연결 반납 ---------------------------------------------------------------------------------


def test_db_transaction_is_closed_before_router_llm_and_reservation_calls(env) -> None:
    _send(env, "안녕하세요", session="s0")  # 이력이 있는 상태에서
    seen: dict[str, bool] = {}
    env.router = _OkRouter(route="general")
    env.router.db = env.db
    env.llm.on_generate = lambda: seen.setdefault("llm", env.db.in_transaction())
    out = _send(env, "오늘 기분이 좋아요", session="s0")
    assert env.router.in_transaction == [False] and seen == {"llm": False}
    assert out["chatLogId"] is not None  # 저장은 그대로
    env.orch.on_handle = lambda: seen.setdefault("reservation", env.db.in_transaction())
    _send(env, ui=("department", "내과"), session="s0")
    assert seen["reservation"] is False


def test_history_and_previous_turn_checks_survive_the_early_release(env) -> None:
    # 연결을 일찍 돌려준 뒤에도 직전 기록(되묻기 상태·이력)을 그대로 읽는다 - 만료된 ORM 객체를 다시 읽지 않는다
    _send(env, "머리가 아파요")
    queries: list[str] = []
    from sqlalchemy import event

    event.listen(env.db.get_bind(), "before_cursor_execute", lambda conn, cur, stmt, *a: queries.append(stmt.split()[0].upper()))
    out = _send(env, "아니요")
    assert _ctx(env, out)["upstreamMeta"]["followupStep"]["index"] == 1
    assert queries.count("SELECT") == 3  # 최근 기록 1번 + 저장 뒤 refresh 1번 + 이 테스트의 _ctx 조회 1번
