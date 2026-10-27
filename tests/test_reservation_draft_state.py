"""R-8: 일반 예약 흐름도 초안을 서버가 저장해(meta.flowState) 다음 턴이 이력 없이 이어 쓴다.
TTL·피보호자 변경·새 요청이면 버리고 이력 재구성(폴백)으로 돌아간다. 이어 쓸 때 바뀐 값에 엮인 칸(병원 id·시간)은 비운다."""
from __future__ import annotations

import time
from typing import Any

from app.services.reservation_confirm_flow import _PENDING_TTL_SECONDS, draft_to_dict, fresh_flow_state
from app.services.reservation_intake import ReservationDraft
from app.services.reservation_orchestrator import _clear_stale_links, _linked_fields
from tests.reservation_flow_helpers import FUTURE, FUTURE2, FakeReservation, make_orchestrator
from tests.test_reservation_confirm_cancel import _stored, _turn


def _flow(draft: ReservationDraft, *, ward_id: str = "", age: int = 0) -> dict[str, Any]:
    return {"flowState": {"draft": draft_to_dict(draft), "replaces": None, "wardId": ward_id, "createdAt": int(time.time()) - age}}


def _chosen(**kw: Any) -> ReservationDraft:
    base = {
        "message": "", "department": "내과", "location": "서울 강남구", "hospital_name": "강남메디내과", "hospital_id": 1,
        "patient_name": "홍길동", "phone": "010-1234-5678",  # 실제 흐름은 첫 턴에 프로필 값이 초안에 들어간다
    }
    return ReservationDraft(**{**base, **kw})


def _handle(orch, stored: dict[str, Any], message: str = "", ui: tuple[str, str] | None = None, **ctx: Any) -> dict[str, Any]:
    from tests.reservation_flow_helpers import make_request
    out = orch.handle(make_request(message, ui=ui, context=ctx or None), {"storedFlow": stored})
    assert out is not None
    return out


def test_general_flow_saves_draft_every_step_and_continues_without_history() -> None:
    fake = FakeReservation()
    orch = make_orchestrator(fake, {"department": "내과"})
    listed = _turn(orch, None, "내과 예약하고 싶어요")
    assert listed["tool"] == "search_hospital"
    assert listed["meta"]["flowState"]["draft"]["department"] == "내과"
    # 이후 턴은 이력 없이(make_request history 비움) 저장된 초안만으로 이어진다
    date_ask = _turn(orch, listed, ui=("hospital_confirm", "강남메디내과"))
    assert date_ask["ui"].field == "date", date_ask["reply"]
    assert date_ask["meta"]["flowState"]["draft"]["hospital_name"] == "강남메디내과"
    time_ask = _turn(orch, date_ask, ui=("date", FUTURE))
    assert time_ask["ui"].field == "time", time_ask["reply"]
    card = _turn(orch, time_ask, ui=("time", "14:00"))
    assert card["ui"].field == "reservation_confirm", card["reply"]
    assert "강남메디내과" in card["reply"] and "14:00" in card["reply"]
    assert "flowState" not in card["meta"], "확인 카드 턴은 pendingAction 만 남긴다"


def test_without_stored_draft_the_history_fallback_still_runs() -> None:
    # 저장 초안이 없으면(옛 기록) 예전처럼 이력·문장에서 다시 만든다 - 날짜 버튼만으로는 병원이 없어 병원부터 다시 찾는다
    orch = make_orchestrator(FakeReservation(), {"department": "내과"})
    out = _handle(orch, {}, ui=("date", FUTURE))
    assert out["tool"] == "search_hospital" or (out["ui"] and out["ui"].field != "time")


def test_expired_or_legacy_state_is_not_continued() -> None:
    assert fresh_flow_state(_flow(_chosen())["flowState"])
    assert not fresh_flow_state(_flow(_chosen(), age=_PENDING_TTL_SECONDS + 5)["flowState"])
    legacy = _flow(_chosen())["flowState"]
    del legacy["createdAt"]
    assert not fresh_flow_state(legacy), "createdAt 없는 배포 전 상태는 이어 쓰지 않는다"
    orch = make_orchestrator(FakeReservation(), {"department": "내과"})
    out = _handle(orch, _flow(_chosen(), age=_PENDING_TTL_SECONDS + 5), ui=("date", FUTURE))
    assert not (out["ui"] and out["ui"].field == "time"), "만료된 초안의 병원을 이어 쓰지 않는다"


def test_continues_only_for_the_same_ward() -> None:
    orch = make_orchestrator(FakeReservation(), {"department": "내과"})
    same = _handle(orch, _flow(_chosen(), ward_id="ward01"), ui=("date", FUTURE), wardId="ward01", wardName="김영희")
    assert same["ui"].field == "time"
    other = _handle(orch, _flow(_chosen(), ward_id="ward01"), ui=("date", FUTURE), wardId="ward02", wardName="박철수")
    assert not (other["ui"] and other["ui"].field == "time"), "다른 피보호자에게 이어 쓰지 않는다"


def test_changing_date_clears_the_old_time_but_a_dated_time_button_keeps_it() -> None:
    orch = make_orchestrator(FakeReservation(), {"department": "내과"})
    picked = _chosen(reservation_date=FUTURE, reservation_time="14:00")
    moved = _handle(orch, _flow(picked), ui=("date", FUTURE2))
    assert moved["ui"].field == "time", "날짜가 바뀌면 그 시간이 비어 있는지 다시 고르게 한다"
    kept = _handle(orch, _flow(_chosen(reservation_date=FUTURE)), ui=("time", f"{FUTURE2} 10:00"))
    assert kept["ui"].field == "reservation_confirm"
    assert "10:00" in kept["reply"]


def test_new_request_sentence_drops_the_stored_draft() -> None:
    orch = make_orchestrator(FakeReservation(), {"department": "안과"})
    out = _handle(orch, _flow(_chosen(reservation_date=FUTURE)), "근처 안과 예약해줘")
    assert out["meta"]["flowState"]["draft"].get("department") == "안과"
    assert out["meta"]["flowState"]["draft"].get("hospital_name") != "강남메디내과"


def test_clear_stale_links_rules() -> None:
    meta: dict[str, Any] = {}
    d = _chosen(reservation_date=FUTURE, reservation_time="14:00")
    before = _linked_fields(d)
    d.hospital_name = "서초내과"  # 병원 이름만 바뀜
    _clear_stale_links(before, d, meta)
    assert d.hospital_id is None and d.reservation_time is None
    d2 = _chosen(reservation_date=FUTURE, reservation_time="14:00")
    before2 = _linked_fields(d2)
    d2.reservation_date, d2.reservation_time = FUTURE2, "10:00"  # 날짜·시간이 함께 바뀜(날짜를 담은 시간 버튼)
    _clear_stale_links(before2, d2, meta)
    assert d2.reservation_time == "10:00" and d2.hospital_id == 1
    d3 = _chosen(reservation_date=FUTURE, reservation_time="14:00")
    before3 = _linked_fields(d3)
    _clear_stale_links(before3, d3, meta)  # 아무것도 안 바뀜
    assert d3.reservation_time == "14:00" and d3.hospital_id == 1


def test_stored_state_round_trip_in_the_helper() -> None:
    out = _turn(make_orchestrator(FakeReservation(), {"department": "내과"}), None, "내과 예약하고 싶어요")
    assert "flowState" in _stored(out)
