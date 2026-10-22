"""예약 흐름 재점검(2026-10-22) 이슈 반영 검증: 상대 날짜 KST(H-1), 대리 환자 고정(M-1), 고치기 wardId 결속(L-1)."""
from __future__ import annotations

import datetime as _dt
from zoneinfo import ZoneInfo

import app.services.reservation_intake as intake
from tests.reservation_flow_helpers import FUTURE, FakeReservation, make_orchestrator
from tests.test_reservation_confirm_cancel import FULL, WARD, _confirm_card, _turn, _value
from tests.test_reservation_confirm_chat_service import _send, env  # noqa: F401

WARD2 = {"wardId": "ward-2", "wardName": "박철수", "wardPhone": "010-8888-1111", "name": "박철수", "phone": "010-8888-1111"}


def _lag_one_day(monkeypatch) -> _dt.date:
    """UTC 컨테이너의 KST 00:00~08:59 를 흉내: 로컬 date.today()가 KST 전날을 돌려준다. KST 오늘을 반환."""
    kst_today = _dt.datetime.now(ZoneInfo("Asia/Seoul")).date()

    class UtcLagDate(_dt.date):
        @classmethod
        def today(cls):
            return kst_today - _dt.timedelta(days=1)

    monkeypatch.setattr(intake, "date", UtcLagDate)
    return kst_today


# ---- H-1 상대 날짜는 KST 기준 -------------------------------------------------------------


def test_relative_dates_are_counted_from_kst_today_even_on_a_utc_host(monkeypatch) -> None:
    kst_today = _lag_one_day(monkeypatch)
    assert intake._find_date("오늘 갈게요") == kst_today.isoformat()
    assert intake._find_date("내일 예약") == (kst_today + _dt.timedelta(days=1)).isoformat()
    assert intake._find_date("모레요") == (kst_today + _dt.timedelta(days=2)).isoformat()


def test_tomorrow_on_a_utc_host_morning_is_not_shown_as_today(monkeypatch) -> None:
    kst_today = _lag_one_day(monkeypatch)
    fake = FakeReservation()
    out = _turn(make_orchestrator(fake, {}), None, "내일 강남메디내과 내과 오후 2시 예약해줘")
    draft = out["meta"]["pendingAction"]["draft"]
    assert draft["reservation_date"] == (kst_today + _dt.timedelta(days=1)).isoformat()
    assert "날짜: 오늘" not in out["reply"]


# ---- M-1 보호자 대리 예약의 환자는 BE가 넘긴 피보호자다 -----------------------------------------


def test_on_behalf_patient_is_pinned_to_ward_even_if_the_chat_names_someone_else() -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake, {**FULL, "patient_name": "이민수", "phone": "010-7777-2222"})
    card = _confirm_card(orch, fake, **WARD)
    assert "김영희님의 예약을 보호자님이 대신" in card["reply"] and "이민수" not in card["reply"]
    _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")), **WARD)
    kw = fake.calls[-1][1]
    assert (kw["patient_name"], kw["phone"]) == ("김영희", "010-9999-0000")


def test_on_behalf_patient_stays_the_ward_when_editing_by_text() -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake, **WARD)
    edit = _turn(orch, card, ui=("reservation_confirm", _value(card, "edit:")), **WARD)
    pick = _turn(orch, edit, ui=("reservation_edit", _value(edit, "time:")), **WARD)
    orch._loader.extracted = {**FULL, "patient_name": "이민수", "phone": "010-7777-2222"}  # 고치기 중 다른 이름을 말함
    out = _turn(orch, pick, "예약 환자는 이민수로, 시간은 10:00으로 해주세요", **WARD)
    assert out["ui"].field == "reservation_confirm", out["reply"]
    assert "환자: 김영희님" in out["reply"] and "이민수" not in out["reply"]


def test_on_behalf_without_ward_phone_still_asks_for_a_contact() -> None:
    """피보호자 전화가 없으면(BE가 비움) 연락처는 입력받는다 - 막으면 예약 자체가 불가능하다."""
    fake = FakeReservation()
    no_phone = {**WARD, "phone": None, "wardPhone": None}
    out = _turn(make_orchestrator(fake, {**FULL, "phone": "010-7777-2222"}), None, "강남메디내과 내과 14:00 예약해줘", **no_phone)
    assert out["ui"].field == "reservation_confirm" and "010-****-2222" in out["reply"] and "환자: 김영희님" in out["reply"]


def test_self_chat_still_accepts_a_typed_patient_name() -> None:
    fake = FakeReservation()
    card = _confirm_card(make_orchestrator(fake, {**FULL, "patient_name": "이민수"}), fake)
    assert "환자: 이민수님" in card["reply"]


# ---- L-1 고치기·변경 상태는 시작한 피보호자에게 묶인다 ----------------------------------------------


def test_switching_ward_during_edit_expires_the_flow_without_booking() -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake, **WARD)
    edit = _turn(orch, card, ui=("reservation_confirm", _value(card, "edit:")), **WARD)
    other = _turn(orch, edit, ui=("reservation_edit", _value(edit, "time:")), **WARD2)
    assert "시간이 지났" in other["reply"] and fake.names() == []


def test_switching_ward_while_continuing_by_text_does_not_carry_the_draft() -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake, **WARD)
    edit = _turn(orch, card, ui=("reservation_confirm", _value(card, "edit:")), **WARD)
    pick = _turn(orch, edit, ui=("reservation_edit", _value(edit, "time:")), **WARD)
    typed = _turn(orch, pick, f"예약 {FUTURE} 10:00으로 해주세요", **WARD2)
    # 앞 피보호자의 초안(병원·진료과)을 이어 쓰지 않고 새 요청으로 다시 묻는다
    assert "김영희" not in typed["reply"] and "강남메디내과" not in typed["reply"]
    assert typed["ui"] is None or typed["ui"].field != "reservation_confirm"


def test_switching_ward_during_a_change_does_not_carry_the_old_reservation() -> None:
    from tests.test_reservation_confirm_cancel import _with_appointments

    fake = _with_appointments()
    orch = make_orchestrator(fake, FULL)
    picked = _turn(orch, _turn(orch, None, "예약 바꿔줘", **WARD), ui=("reservation_change_pick", "5"), **WARD)
    assert picked["meta"]["flowState"]["replaces"]["id"] == 5
    typed = _turn(orch, picked, f"예약 {FUTURE} 10:00으로 해주세요", **WARD2)
    assert "기존 예약" not in typed["reply"]
    assert ((typed.get("meta") or {}).get("flowState") or {}).get("replaces") is None


# ---- L-3 확인 카드는 30분 안에만 동작한다 ---------------------------------------------------------


def test_confirm_card_older_than_thirty_minutes_expires_without_booking(monkeypatch) -> None:
    import time as _time
    from types import SimpleNamespace

    import app.services.reservation_confirm_flow as flow

    fake = FakeReservation()
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    later = _time.time() + flow._PENDING_TTL_SECONDS + 1
    monkeypatch.setattr(flow, "time", SimpleNamespace(time=lambda: later))
    out = _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")))
    assert fake.names() == [] and "시간이 지났" in out["reply"]


def test_confirm_card_saved_before_the_time_limit_existed_is_treated_as_expired() -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    card["meta"]["pendingAction"].pop("createdAt")  # 배포 전에 저장된 카드
    out = _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")))
    assert fake.names() == [] and "시간이 지났" in out["reply"]


def test_cancel_card_also_expires_after_thirty_minutes(monkeypatch) -> None:
    import time as _time
    from types import SimpleNamespace

    import app.services.reservation_confirm_flow as flow
    from tests.test_reservation_confirm_cancel import _with_appointments

    fake = _with_appointments()
    orch = make_orchestrator(fake)
    confirm = _turn(orch, _turn(orch, None, "예약 취소할래"), ui=("reservation_cancel_pick", "5"))
    later = _time.time() + flow._PENDING_TTL_SECONDS + 1
    monkeypatch.setattr(flow, "time", SimpleNamespace(time=lambda: later))
    out = _turn(orch, confirm, ui=("reservation_cancel_confirm", _value(confirm, "yes:")))
    assert "cancel" not in fake.names() and "시간이 지났" in out["reply"]


# ---- L-2 라우터·폴백 로그에 상담 내용·예외 문구를 남기지 않는다 -------------------------------------------


def test_router_log_keeps_only_route_and_length(env, caplog, monkeypatch) -> None:  # noqa: F811
    import logging

    from app.services import chat_service as cs

    class _SymptomRouter:
        def route(self, message, history, ctx, prev_reservation=None):  # noqa: ARG002
            return {"route": "symptom", "normalized_message": message, "symptom_summary": "무릎이 시큰거림", "emergency": False, "engine": "fake"}

    monkeypatch.setattr(cs, "get_chat_router_service", lambda: _SymptomRouter())
    with caplog.at_level(logging.INFO, logger="app.services.chat_service"):
        _send(env, "무릎이 시큰거려요")
    router_lines = [r.getMessage() for r in caplog.records if "[라우터]" in r.getMessage()]
    assert router_lines and all("시큰" not in line for line in router_lines)


def test_fallback_log_has_exception_class_and_place_but_not_its_text(env, caplog, monkeypatch) -> None:  # noqa: F811
    import logging

    class _Boom:
        def handle(self, *a, **k):  # noqa: ARG002
            raise RuntimeError("010-1234-5678 원문이 섞인 오류")

    from app.services import chat_service as cs

    monkeypatch.setattr(cs, "get_reservation_orchestrator", lambda: _Boom())
    with caplog.at_level(logging.INFO, logger="app.services.chat_service"):
        _send(env, "강남메디내과 내과 14:00 예약해줘")
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "RuntimeError" in text and "test_reservation_recheck_findings.py:" in text
    assert "010-1234-5678" not in text and "원문" not in text
    assert not any(r.exc_info for r in caplog.records)
