"""R-8 남은 것: 저장 초안을 이어 쓰는 턴의 추출은 최근 대화 대신 그 초안을 [현재 상태 JSON]으로 받는다.
확인 카드를 띄운 사이 자정을 넘겨 날짜가 지나면(예약 서비스 INVALID_REQUEST) 날짜부터 다시 고르게 한다."""
from __future__ import annotations

import json
import time
from datetime import date, timedelta
from typing import Any

from app.services.reservation_confirm_flow import draft_to_dict
from app.services.reservation_intake import ReservationDraft
from tests.reservation_flow_helpers import FUTURE, FakeLoader, FakeReservation, make_orchestrator, make_request
from tests.test_reservation_confirm_cancel import FULL, _confirm_card, _turn, _value

OLD_HISTORY = [
    {"role": "user", "content": "지난주에 서초내과 10시로 잡았던 거 있잖아"},
    {"role": "assistant", "content": "서초내과 10:00 예약이 완료되었습니다."},
]


class RecordingLoader(FakeLoader):
    def __init__(self, extracted: dict[str, Any] | None = None) -> None:
        super().__init__(extracted)
        self.prompts: list[str] = []

    def generate_structured_text(self, system: str, prompt: str, schema: Any = None) -> str:
        self.prompts.append(prompt)
        return super().generate_structured_text(system, prompt, schema)


def _orch(fake: FakeReservation, loader: RecordingLoader):
    orch = make_orchestrator(fake)
    orch._loader = loader
    return orch


def _stored() -> dict[str, Any]:
    d = ReservationDraft(message="", department="내과", location="서울 강남구", hospital_name="강남메디내과", hospital_id=1,
                         patient_name="홍길동", phone="010-1234-5678")
    return {"flowState": {"draft": draft_to_dict(d), "replaces": None, "wardId": "", "createdAt": int(time.time())}}


def _section(prompt: str, title: str) -> str:
    return prompt.split(f"[{title}]\n", 1)[1].split("\n\n[", 1)[0]


def test_continuing_turn_sends_stored_draft_instead_of_history() -> None:
    loader = RecordingLoader({"reservation_date": FUTURE})
    orch = _orch(FakeReservation(), loader)
    req = make_request("모레로 할게요 예약", history=OLD_HISTORY)
    orch.handle(req, {"storedFlow": _stored(), "rawMessage": "모레로 할게요"})
    prompt = loader.prompts[-1]
    assert "서초내과" not in prompt and "10:00" not in prompt, "이력 속 지난 예약 값이 추출 입력에 들어가지 않는다"
    state = json.loads(_section(prompt, "현재 상태 JSON"))
    assert state["facts"] == {"department": "내과", "hospital_name": "강남메디내과", "location": "서울 강남구"}
    assert "patient_name" not in state["facts"] and "phone" not in state["facts"], "환자 정보는 상태로 넘기지 않는다"


def test_fresh_turn_still_reads_history() -> None:
    loader = RecordingLoader({"department": "내과"})
    orch = _orch(FakeReservation(), loader)
    orch.handle(make_request("내과 예약하고 싶어요", history=OLD_HISTORY), {"storedFlow": {}})
    assert "서초내과" in _section(loader.prompts[-1], "최근 대화")


def test_date_passed_while_card_was_open_asks_date_again() -> None:
    fake = FakeReservation()
    fake.create_result = {"httpStatus": 400, "data": {"success": False, "code": "INVALID_REQUEST", "message": "예약 날짜는 오늘 이후여야 합니다."}}
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    # 카드를 만든 뒤 자정이 지난 상황: 저장된 초안의 날짜를 어제로 바꿔 둔다
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    card["meta"]["pendingAction"]["draft"]["reservation_date"] = yesterday
    out = _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")))
    assert out["ui"].field == "date", out["reply"]
    assert out["reply"].startswith("고르신 날짜가 지났어요.")
    assert out.get("type") != "tool_error"


def test_other_invalid_request_is_still_an_error() -> None:
    fake = FakeReservation()
    fake.create_result = {"httpStatus": 400, "data": {"success": False, "code": "INVALID_REQUEST", "message": "잘못된 요청"}}
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)  # 날짜는 미래 그대로
    out = _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")))
    assert out["type"] == "tool_error" and out["error"]["code"] == "INVALID_REQUEST"


def test_hospital_day_off_on_create_asks_date_again() -> None:
    fake = FakeReservation()
    fake.create_result = {"httpStatus": 400, "data": {"success": False, "code": "HOSPITAL_DAY_OFF", "message": "그날은 병원이 쉬는 날이에요."}}
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    out = _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")))
    assert out["ui"].field == "date", out["reply"]
    assert out["reply"].startswith("그날은 병원이 쉬는 날이에요. 다른 날짜를 골라 주세요.")
