from __future__ import annotations

import json

from app.services.chat_router_service import ChatRouterService
from app.services.medical_llm_service import MedicalLlmService


class _FakeGpt:
    def __init__(self, reply: dict) -> None:
        self._reply = reply
        self.state = type("S", (), {"model_name": "fake-gpt"})()

    def generate_structured_text(self, system_prompt: str, user_prompt: str) -> str:
        assert "[사용자 문장]" in user_prompt
        return json.dumps(self._reply, ensure_ascii=False)


def test_router_reservation_normalizes_message() -> None:
    svc = ChatRouterService(_FakeGpt({"route": "reservation", "normalized_message": "서울 강남구 / 이비인후과", "emergency": False}), 8)
    out = svc.route("강남역 근처에 이비인후과 예약 좀 해줘", [], {})
    assert out["route"] == "reservation"
    assert out["normalized_message"].startswith("병원 예약:")  # 예약 로직의 키워드 게이트를 통과해야 한다
    assert "서울 강남구" in out["normalized_message"]


def test_router_symptom_gives_processed_text() -> None:
    svc = ChatRouterService(_FakeGpt({"route": "symptom", "symptom_summary": "주증상: 기침. 유발: 추위.", "emergency": False}), 8)
    out = svc.route("오늘 아침에 추워서 기침이 계속 나오네", [], {})
    assert out["route"] == "symptom"
    assert out["symptom_summary"].startswith("주증상")


def test_router_rejects_unknown_route() -> None:
    svc = ChatRouterService(_FakeGpt({"route": "??"}), 8)
    try:
        svc.route("안녕", [], {})
    except RuntimeError:
        return
    raise AssertionError("unknown route must raise so chat_service falls back to keywords")


def test_reply_no_longer_duplicates_sections() -> None:
    reply = MedicalLlmService.format_answer({
        "finalMessage": "따뜻한 물을 드세요.",
        "summary": "추위로 인한 기침 안내입니다.",
        "possibleCauses": ["기관지 자극"],
        "homeCare": ["따뜻한 차"],
    })
    assert "가능한 원인" not in reply and "기관지 자극" not in reply
    assert "따뜻한 물을 드세요." in reply and "의료진 상담" in reply


def test_department_from_current_message_beats_history() -> None:
    from app.services.reservation_intake import build_reservation_draft

    draft = build_reservation_draft("병원 예약: 서울 강남구 / 내과", history_text="user: 강남역 이비인후과 예약 좀 해줘")
    assert draft.department == "내과"
