"""예약 재점검 수정분 점검(2026-10-22) 이슈 반영 검증: 챗 기록의 예약 API 키(H-1), 원문 로그(M-1), 라우터 기억·옛 고치기 상태 wardId(L-1·L-2),
피보호자 전화(L-3), 날짜 뒤 조사(L-4)."""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from app.models.chat_log import ChatLog
from app.schemas.chat_schema import ChatContext, ChatRequest, UiSelection
from app.services import reservation_intake as intake
from app.services.chat_service import ChatService
from tests.reservation_flow_helpers import FakeReservation, make_orchestrator
from tests.test_reservation_confirm_cancel import FULL, WARD, _confirm_card, _turn, _value
from tests.test_reservation_confirm_chat_service import _token, env  # noqa: F401

SENSITIVE = "010-5555-6666 홍길순 무릎이 시큰"


def _send_as(env, ctx: dict[str, Any], message: str = "", ui: tuple[str, str] | None = None) -> dict[str, Any]:  # noqa: F811
    payload = ChatRequest(
        userId="user-1", message=message, sessionId="s1", context=ChatContext(**ctx),
        uiSelection=UiSelection(field=ui[0], value=ui[1]) if ui else None,
    )
    return env.svc.process_message(env.db, payload)


def _stored_contexts(env) -> list[str]:  # noqa: F811
    return [row.context_json for row in env.db.query(ChatLog).order_by(ChatLog.id).all()]


# ---- H-1 챗 기록에 예약 API 키·직전 상태 사본·원문을 남기지 않는다 ----------------------------------------


def test_chat_log_never_stores_the_reservation_api_key(env) -> None:  # noqa: F811
    ctx = {"name": "홍길동", "phone": "010-1234-5678", "location": "서울 강남구"}
    card = _send_as(env, ctx, "강남메디내과 내과 14:00 예약해줘")
    done = _send_as(env, ctx, ui=("reservation_confirm", _token(card, "confirm:")))
    assert env.fake.names() == ["create"] and done["tool"] == "make_appointment"  # 다음 턴 상태 복원은 그대로 동작한다
    stored = _stored_contexts(env)
    assert len(stored) == 2
    for raw in stored:
        meta = json.loads(raw)["upstreamMeta"]
        assert "fake-key" not in raw  # _Credentials.ensure_credential 이 돌려준 예약 API 키
        assert not {"reservationApiKey", "storedFlow", "rawMessage"} & set(meta)
    assert "pendingAction" in json.loads(stored[0])["upstreamMeta"]  # 확인 대기 상태는 남긴다


def test_chat_logs_api_does_not_hand_out_the_key(env) -> None:  # noqa: F811
    _send_as(env, {"name": "홍길동", "phone": "010-1234-5678", "location": "서울 강남구"}, "강남메디내과 내과 14:00 예약해줘")
    assert "fake-key" not in json.dumps(env.svc.list_logs(env.db, user_id="user-1"), ensure_ascii=False, default=str)


# ---- M-1 원문이 트레이스·로그에 남지 않는다 -----------------------------------------------------------


def test_extraction_failure_keeps_only_the_exception_class(caplog) -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake, FULL)

    def boom(*a, **k):  # noqa: ARG001
        raise RuntimeError(SENSITIVE)

    orch._loader.generate_structured_text = boom  # type: ignore[method-assign]
    with caplog.at_level(logging.INFO):
        out = _turn(orch, None, "강남메디내과 내과 14:00 예약해줘")
    traces = [t for t in out["meta"]["decisionTrace"] if t["event"] == "reservation_extraction_failed"]
    assert traces and traces[0]["error"] == "RuntimeError"
    assert SENSITIVE not in json.dumps(out["meta"]["decisionTrace"], ensure_ascii=False)
    assert SENSITIVE not in caplog.text


def test_reservation_service_error_text_is_not_traced_or_logged(caplog) -> None:
    fake = FakeReservation()
    fake.create_result = {"httpStatus": 400, "data": {"code": "BAD_REQUEST", "message": f"잘못된 전화번호 {SENSITIVE}"}}
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    with caplog.at_level(logging.INFO):
        out = _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")))
    failed = [t for t in out["meta"]["decisionTrace"] if t["event"] == "reservation_submit_failed"]
    assert failed and failed[0] == {"event": "reservation_submit_failed", "http_status": 400, "code": "BAD_REQUEST"}
    assert SENSITIVE not in caplog.text
    assert out["error"]["code"] == "BAD_REQUEST"  # 사용자 안내(응답)는 그대로다


def test_medgemma_parse_fallback_log_has_no_model_output(caplog) -> None:
    from app.services.medical_llm_service import MedicalLlmService

    class _Loader:
        state = SimpleNamespace(model_name="fake")

        def ensure_loaded(self) -> None:
            return None

        def generate_structured_text(self, *a, **k) -> str:  # noqa: ARG002
            return f'{{"empathy": "{SENSITIVE}", "advice": "잘린 출력'  # 닫히지 않은 JSON

    svc = MedicalLlmService(SimpleNamespace(chat_history_keep_turns=6, chat_enable_llm=True, medgemma_gen_wait_timeout_sec=60, medgemma_gen_wait_timeout_emergency_sec=3), _Loader())  # type: ignore[arg-type]
    with caplog.at_level(logging.WARNING, logger="app.services.medical_llm_service"):
        svc.generate("무릎이 아파요", {}, [])
    lines = [r.getMessage() for r in caplog.records if "parse fallback" in r.getMessage()]
    assert lines and all(SENSITIVE not in line and "홍길순" not in line for line in lines)


# ---- L-1 라우터 기억은 같은 피보호자 상담에서만 이어 쓴다 ---------------------------------------------------


def _db_with(router: dict[str, Any], session_id: str = "s1"):
    log = SimpleNamespace(context_json=json.dumps({"sessionId": session_id, "upstreamMeta": {"router": router}}))
    chain = SimpleNamespace()
    chain.filter = lambda *_a, **_k: chain
    chain.order_by = lambda *_a, **_k: chain
    chain.first = lambda: log
    return SimpleNamespace(query=lambda *_a, **_k: chain)


def test_router_memory_is_bound_to_the_ward() -> None:
    a = {"route": "reservation", "normalized_message": "병원 예약: 강남메디내과", "wardId": "ward-1"}
    assert ChatService._last_reservation_state(_db_with(a), "u", "s1", "ward-1") == "병원 예약: 강남메디내과"
    assert ChatService._last_reservation_state(_db_with(a), "u", "s1", "ward-2") is None
    assert ChatService._last_reservation_state(_db_with(a), "u", "s1", "") is None
    old = {"route": "reservation", "normalized_message": "병원 예약: 서초내과"}  # wardId 를 남기기 전의 기록 = 본인 상담
    assert ChatService._last_reservation_state(_db_with(old), "u", "s1") == "병원 예약: 서초내과"
    assert ChatService._last_reservation_state(_db_with(old), "u", "s1", "ward-1") is None


def test_router_memory_records_the_ward_and_a_button_turn_for_another_ward_starts_fresh(env) -> None:  # noqa: F811
    first = _send_as(env, WARD, "강남메디내과 내과 예약해줘")
    stored = json.loads(_stored_contexts(env)[-1])["upstreamMeta"]["router"]
    assert stored["wardId"] == "ward-1"
    assert first["ui"] is not None
    other = {**WARD, "wardId": "ward-2", "wardName": "박철수", "wardPhone": "010-8888-1111", "name": "박철수", "phone": "010-8888-1111"}
    _send_as(env, other, ui=("department", "내과"))
    router = json.loads(_stored_contexts(env)[-1])["upstreamMeta"]["router"]
    assert router["wardId"] == "ward-2" and "강남메디내과" not in router["normalized_message"]


# ---- L-2 wardId 키가 없는 옛 고치기 상태는 이어 쓰지 않는다 ---------------------------------------------------


def _legacy(out: dict[str, Any]) -> dict[str, Any]:
    out["meta"]["flowState"].pop("wardId")  # 배포 전에 저장된 형태
    return out


def test_legacy_edit_state_without_ward_does_not_continue_into_self_chat() -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake, **WARD)
    edit = _legacy(_turn(orch, card, ui=("reservation_confirm", _value(card, "edit:")), **WARD))
    stale = orch.handle(make_self_request(ui=("reservation_edit", _value(edit, "time:"))), {"storedFlow": {"flowState": edit["meta"]["flowState"]}})
    assert stale is not None and "시간이 지났" in stale["reply"]


def test_legacy_flow_state_is_not_continued_by_text_in_self_chat() -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake, **WARD)
    edit = _turn(orch, card, ui=("reservation_confirm", _value(card, "edit:")), **WARD)
    pick = _legacy(_turn(orch, edit, ui=("reservation_edit", _value(edit, "time:")), **WARD))
    out = orch.handle(make_self_request("예약 시간 10:00으로 해주세요"), {"storedFlow": {"flowState": pick["meta"]["flowState"]}})
    assert out is not None and "김영희" not in out["reply"]
    draft = next(t for t in out["meta"]["decisionTrace"] if t["event"] == "reservation_draft")
    assert draft["phone_last4"] == "5678"  # 앞 피보호자(9999-0000)의 초안이 아니라 본인 정보로 새로 만들었다


def make_self_request(message: str = "", ui: tuple[str, str] | None = None) -> ChatRequest:
    return ChatRequest(
        userId="user-1", message=message, sessionId="s1", history=[],
        context=ChatContext(name="홍길동", phone="010-1234-5678", location="서울 강남구"),
        uiSelection=UiSelection(field=ui[0], value=ui[1]) if ui else None,
    )


# ---- L-3 피보호자 전화가 없으면 context.phone 이 입력 연락처를 덮지 않는다 ---------------------------------------


def test_entered_contact_is_kept_when_the_ward_has_no_phone_even_if_context_phone_lingers() -> None:
    fake = FakeReservation()
    lingering = {**WARD, "wardPhone": None, "phone": "010-1111-2222"}  # BE 계약상 오지 않는 형태(방어)
    card = _confirm_card(make_orchestrator(fake, {**FULL, "phone": "010-7777-3333"}), fake, **lingering)
    assert "010-****-3333" in card["reply"] and "환자: 김영희님" in card["reply"]


# ---- L-4 날짜 뒤에 조사가 붙어도 읽는다 --------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2026-10-30에 예약해줘", "2026-10-30"),
        ("2026.11.02로 해주세요", "2026-11-02"),
        ("예약 2026/12/01", "2026-12-01"),
        ("12026-10-30", None),  # 앞에 숫자가 더 붙은 값은 날짜가 아니다
        ("2026-10-301", None),
    ],
)
def test_absolute_date_with_a_trailing_particle_is_parsed(text: str, expected: str | None) -> None:
    assert intake._find_date(text) == expected
