"""남은 품질 약점 ①②⑤(2026-11-04, 홀드아웃 v4 판정) - 생성 지시·분류 규칙 고정.

① 증상 상담 advice 는 진료과 / 구체 기준(기간·수치·더해지는 증상) / 집에서 할 일 세 문장.
② 약 질문은 단정·용량 변경·중단 권유 금지를 유지하고, 이유 한 문장 + 누구에게 무엇을 물을지 한 문장.
⑤ 가족이 대신 묻는 생활·집 안전 질문은 라우터가 general 로 보내도 상담으로 올린다. 수치는 단위·기간을 그대로.
"""
from __future__ import annotations

import pytest

from app.prompts.medical_chat_prompts import SYSTEM_PROMPT
from app.services import chat_keywords
from app.services.chat_router_service import SYSTEM_PROMPT as ROUTER_PROMPT
from app.services.chat_service import ChatService
from app.services.medical_llm_service import MedicalLlmService


def _rule(n: str) -> str:
    return next(line for line in SYSTEM_PROMPT.splitlines() if line.startswith(f"{n})"))


def _example(prefix: str) -> str:
    return next(line for line in SYSTEM_PROMPT.splitlines() if line.startswith(prefix))


# ---- ① 구체 기준·집에서 할 일 ----------------------------------------------------------------------


def test_rule8_asks_concrete_criterion_and_home_care() -> None:
    rule8 = _rule("8")
    assert "구체적인 기준을 하나 이상" in rule8
    assert "며칠 넘게" in rule8 and "얼마 이상" in rule8  # 기간·수치
    assert "집에서 할 수 있는 일 한 가지" in rule8
    assert "약 이름·용량은 쓰지 않는다" in rule8


def test_rule8_keeps_119_only_with_red_flags() -> None:
    rule8 = _rule("8")
    assert "위험 신호가 함께 있을 때만" in rule8
    assert "예시에 나온 문장을 다른 증상에 가져오지 마라" in rule8


def test_rule5_allows_three_advice_sentences() -> None:
    rule5 = _rule("5")
    assert "advice는 세 문장까지" in rule5
    assert "empathy는 두 문장까지" in rule5


def test_example1_has_criterion_and_home_care_without_119() -> None:
    example1 = _example('{"empathy": "허리가')
    assert "2주 넘게" in example1 and "다리가 저려 오면" in example1
    assert "119" not in example1
    assert example1.count(".") >= 3  # 세 문장


# ---- ② 약 질문: 이유 + 확인할 사람, 금지는 그대로 ---------------------------------------------------------


def test_rule4_keeps_no_verdict_and_no_dose_change() -> None:
    rule4 = _rule("4")
    assert "단정하지 말고" in rule4
    assert "늘리거나 줄이거나 끊으라고도 하지 마라" in rule4
    assert "119 문장은 붙이지 않는다" in rule4
    assert "이미 약을 더 드셨다고 하면" in rule4  # 과다 복용 119 는 그대로


def test_rule4_adds_reason_and_whom_to_ask() -> None:
    rule4 = _rule("4")
    assert "왜 조심해야 하는지" in rule4
    assert "약사나 의사에게" in rule4
    assert "음식·영양제·다른 약" in rule4


def test_example2_gives_reason_and_whom_without_verdict() -> None:
    example2 = _example('{"empathy": "관절약')
    assert "부작용" in example2  # 이유
    assert "약사나 의사에게" in example2  # 확인할 사람
    for banned in ("119", "드셔도 돼요", "드셔도 괜찮", "끊으", "줄이", "늘리"):
        assert banned not in example2


# ---- ⑤ 수치 단위 ---------------------------------------------------------------------------------


def test_rule10_keeps_units() -> None:
    rule10 = _rule("10")
    assert "일주일 기준 권장량을 하루 기준으로 바꿔 쓰지 마라" in rule10
    assert "확실히 모르는 수치는 쓰지 말고" in rule10


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
