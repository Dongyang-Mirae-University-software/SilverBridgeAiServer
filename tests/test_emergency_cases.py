from __future__ import annotations

from app.services.chat_service import ChatService
from app.services.medical_llm_service import MedicalLlmService

# docs/chat-eval-set.md 의 응급 케이스 중 키워드 층에서 결정되는 것만. LLM 판단이 필요한 케이스는 문서에만 둔다.
EMERGENCY = [
    "숨을 못 쉬겠어요",
    "숨이 안 쉬어져요",
    "가슴이 조이고 식은땀이 나요",
    "할머니가 쓰러지셨어요",
    "의식이 없어요 불러도 반응이 없어요",
    "갑자기 말이 어눌해지고 한쪽 팔에 힘이 안 들어가요",
    "피를 토했어요",
    "피가 안 멈춰요",
    "경련을 해요",
    "약을 한꺼번에 많이 먹었어요",
    "죽고 싶어요",
    "응급 증상이 있어요",
    "가슴 통증이 있는데 응급실 예약해줘",
    "넘어져서 못 일어나요",
]
NOT_EMERGENCY = [
    "어제부터 기침이 나요",
    "무릎이 시려요",
    "내일 내과 예약해줘",
    "고마워요",
    "혈압약 먹었어요",
    "속이 더부룩해요",
]


def test_emergency_phrasings_are_classified_as_emergency() -> None:
    for message in EMERGENCY:
        assert ChatService.classify_intent(message) == "emergency_guidance", message
        assert MedicalLlmService.classify_intent(message) == "emergency_guidance", message
        assert MedicalLlmService.emergency_signals(message), message


def test_normal_phrasings_are_not_emergency() -> None:
    for message in NOT_EMERGENCY:
        assert ChatService.classify_intent(message) != "emergency_guidance", message


def test_force_emergency_overrides_reservation_route() -> None:
    out = ChatService._force_emergency(
        {"route": "reservation", "emergency": False, "normalized_message": "병원 예약: 응급의학과", "symptom_summary": ""},
        "가슴 통증이 있는데 응급실 예약해줘",
    )
    assert out["route"] == "symptom" and out["emergency"] is True
    assert "가슴통증" in out["emergency_signals"]
    assert out["symptom_summary"]  # 비어 있으면 원문으로 메운다


def test_force_emergency_respects_router_flag_without_keyword() -> None:
    out = ChatService._force_emergency({"route": "general", "emergency": True, "symptom_summary": "x"}, "할아버지가 이상해요")
    assert out["route"] == "symptom"


def test_force_emergency_leaves_normal_route_alone() -> None:
    info = {"route": "reservation", "emergency": False}
    assert ChatService._force_emergency(info, "내일 내과 예약해줘") is info
    assert ChatService._force_emergency({}, "") == {}


def test_emergency_wording_tells_to_call_119_first() -> None:
    out = MedicalLlmService._normalize_payload({"symptoms": "가슴 통증", "advice": "내과 진료를 권해드려요."}, "", "가슴이 조여요", "emergency_guidance", True)
    reply = MedicalLlmService.format_answer(out, emergency=True)
    assert "119에 전화" in reply and "진료를 권해드려요" not in reply
