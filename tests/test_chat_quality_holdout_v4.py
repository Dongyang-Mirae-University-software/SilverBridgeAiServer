"""가족이 대신 묻는 생활·집 안전 질문의 상담 경로(2026-11-04, 홀드아웃 v4 판정 ⑤).

라우터가 general 로 보내도 생활 관리 주제 + 묻는 말이면 상담(symptom)으로 올린다. 응급·자해 판정은 건드리지 않는다.
생성 지시 변경(①·②)은 응답 시간 기준(+20%)을 넘어 반영하지 않았다 - docs/eval-results/2026-11-04-quality-v4.md.
"""
from __future__ import annotations

import pytest

from app.services import chat_keywords
from app.services.chat_router_service import SYSTEM_PROMPT as ROUTER_PROMPT
from app.services.chat_service import ChatService
from app.services.medical_llm_service import MedicalLlmService


# ---- ⑤ 생활·집 안전 질문 경로 ------------------------------------------------------------------------

LIFESTYLE = [
    "혼자 사시는 아버지 집에 안 넘어지게 뭘 해 드리면 좋을까요",
    "엄마 욕실이 미끄러운데 어떻게 손봐야 해요",
    "할머니 겨울에 난방 어떻게 챙겨 드려야 하나요",
    "73살인데 운동은 일주일에 얼마나 해야 돼요",
    "혈압 있으면 소금은 하루에 몇 그램까지 되나요",
    "기억력이 떨어진 어머니 혼자 계실 때 뭘 조심해야 해요",
]

CHITCHAT = [
    "오늘 산책 다녀왔어요",
    "식사하셨어요?",
    "고마워요 잘 지내세요",
    "운동 열심히 할게요",
    "날씨가 좋네요",
]


@pytest.mark.parametrize("text", LIFESTYLE)
def test_lifestyle_question_detected(text: str) -> None:
    assert chat_keywords.is_health_lifestyle_question(text)


@pytest.mark.parametrize("text", CHITCHAT)
def test_chitchat_not_lifestyle(text: str) -> None:
    assert not chat_keywords.is_health_lifestyle_question(text)


def _route(route: str, **extra) -> dict:
    return {"route": route, "normalized_message": "", "symptom_summary": "", "emergency": False, "self_harm": False,
            "engine": "claude", **extra}


@pytest.mark.parametrize("text", LIFESTYLE)
def test_router_general_lifted_to_symptom(text: str) -> None:
    out = ChatService._apply_router_overrides(_route("general"), text)
    assert out["route"] == "symptom"
    assert out["symptom_summary"] == text
    assert out["emergency"] is False


@pytest.mark.parametrize("text", CHITCHAT)
def test_router_general_chitchat_stays(text: str) -> None:
    assert ChatService._apply_router_overrides(_route("general"), text)["route"] == "general"


def test_router_override_leaves_emergency_and_self_harm() -> None:
    text = "아버지 집에 안 넘어지게 뭘 해 드리면 좋을까요"
    emergency = ChatService._apply_router_overrides(_route("general", emergency=True), text)
    assert emergency["route"] == "general" and emergency["emergency"] is True  # 응급은 _force_emergency 가 맡는다
    self_harm = ChatService._apply_router_overrides(_route("general", self_harm=True), text)
    assert self_harm["route"] == "general"


def test_router_override_keeps_symptom_summary() -> None:
    out = ChatService._apply_router_overrides(_route("general", symptom_summary="요약"), "운동은 일주일에 얼마나 해야 돼요")
    assert out["symptom_summary"] == "요약"


@pytest.mark.parametrize("text", LIFESTYLE)
def test_keyword_fallback_lifestyle_is_medical(text: str) -> None:
    assert MedicalLlmService.classify_intent(text) == "medical_advice"


# "운동 열심히"는 기존 상담 낱말 "열"에 걸린다(이번 변경 밖) - 대체 분류 확인에서는 뺀다.
@pytest.mark.parametrize("text", [t for t in CHITCHAT if "열심히" not in t])
def test_keyword_fallback_chitchat_stays_general(text: str) -> None:
    assert MedicalLlmService.classify_intent(text) == "general_chat"


def test_fall_emergency_still_emergency() -> None:
    # 대조군: 넘어져 못 일어나면 생활 질문이 아니라 응급이다.
    text = "할머니가 화장실에서 넘어지셔서 못 일어나세요 어떻게 해요"
    assert MedicalLlmService.classify_intent(text) == "emergency_guidance"
    forced = ChatService._force_emergency(ChatService._apply_router_overrides(_route("general"), text), text)
    assert forced["route"] == "symptom" and forced["emergency"] is True


def test_router_prompt_unchanged_by_this_work() -> None:
    # 라우터(Opus) 지시문은 API 없이 검증할 수 없어 바꾸지 않았다(2026-11-04) - ⑤는 키워드 덮어쓰기로만 푼다.
    assert "낙상 예방·집 안전" not in ROUTER_PROMPT


# ---- 2026-11-05 기능 점검 F-1·F-2·F-3 --------------------------------------------------------------

NON_HEALTH = [
    "물을 엎질렀는데 어떻게 닦아요",
    "손잡이가 고장났는데 어떻게 고쳐요",
    "보일러가 고장났어요 어떻게 해야 돼요",
    "치매 보험 어떤 게 좋아요",
    "어떤 수면 바지가 좋을까요",
    "걷기 대회 몇 시에 해요",
    "반찬 어떤 거 해 먹을까요",
    "운동 프로그램 몇 번 채널이에요",
    "내일 더위가 어떻게 될까요",
    "기억력 게임 어떻게 해요",
    "난방비가 얼마나 나와요",
    "손주 운동회 뭘 챙겨 가요",
]

SHORT_LIFESTYLE = [
    "낮잠 자도 괜찮을까요",
    "아버지 집에 안 넘어지게 뭘 해 드리면 좋을까요",
    "잠은 몇 시간 자야 돼요",
    "잠은 어떻게 자야 푹 자요",
    "물은 하루에 얼마나 마셔야 돼요",
]


@pytest.mark.parametrize("text", NON_HEALTH)
def test_non_health_question_not_lifted(text: str) -> None:
    # F-1: 라우터가 맞게 general 로 본 비건강 질문(집수리·보험·물건·방송·날씨)을 뒤집지 않는다.
    assert not chat_keywords.is_health_lifestyle_question(text)
    assert ChatService._apply_router_overrides(_route("general"), text)["route"] == "general"


@pytest.mark.parametrize("text", SHORT_LIFESTYLE)
def test_short_lifestyle_question_lifted_without_followup(text: str) -> None:
    # F-2·F-3: 짧은 생활 질문도 상담으로 올리되 "언제부터·얼마나" 되묻기로 빠지지 않는다.
    assert ChatService._apply_router_overrides(_route("general"), text)["route"] == "symptom"
    assert MedicalLlmService.needs_followup_question(text) is False


def test_habit_fix_and_heating_worry_still_lifted() -> None:
    # 제외어를 넓게 잡아 생활 습관 "고쳐요"·난방비를 아끼는 가족 걱정까지 빠지지 않게(재분류 확인 ho.L03·h4.F03).
    assert chat_keywords.is_health_lifestyle_question("낮잠을 오래 자서 밤에 못 자는데 어떻게 고쳐요")
    assert chat_keywords.is_health_lifestyle_question("어머니가 난방비 아낀다고 난방을 안 켜시는데 어떻게 해야 해요")
    assert chat_keywords.is_health_lifestyle_question("와파린 먹는데 나물 반찬 많이 먹어도 되나요")


def test_symptom_followup_unchanged() -> None:
    # 대조: 생활 질문이 아닌 짧은 증상 한마디는 그대로 되묻는다.
    assert MedicalLlmService.needs_followup_question("어지러워요") is True
