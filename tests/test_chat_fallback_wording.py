from __future__ import annotations

from app.services.chat_service import ChatService


def _fallback(message: str) -> dict:
    # 폴백은 LLM·DB 없이 키워드로만 만든다
    return ChatService()._fallback_reply(message, [])


def test_medium_risk_fallback_does_not_name_guardian() -> None:
    # 보호자 본인에게도, 연결된 보호자가 없는 피보호자에게도 맞는 문구여야 한다
    reply = _fallback("어제부터 기침이 계속 나요")["reply"]
    assert "보호자" not in reply
    assert "가족이나 가까운 분" in reply
    assert "진료가 필요할 수 있습니다" in reply


def test_high_risk_fallback_still_guides_to_emergency() -> None:
    reply = _fallback("가슴이 너무 아파서 숨을 못 쉬겠어요")["reply"]
    assert "119" in reply


def test_recommended_action_code_is_unchanged() -> None:
    # recommendedAction 값은 API 계약이라 문구와 별개로 유지한다
    assert _fallback("어제부터 기침이 계속 나요")["recommendedAction"] == "guardian_contact"
