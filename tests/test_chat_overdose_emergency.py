"""숫자·"한꺼번에"·최대 초과·위험 약 중복으로 말한 과다 복용도 응급으로 본다(2026-11-28 전체 점검 H-1).

원인: 응급 키워드의 과다 복용 꼴이 "약을많이먹·약을한꺼번에·과다복용" 세 가지뿐이라 "수면제 5알 먹었는데 괜찮나요?"가
상담으로 남았고, 약 질문 119 빼기(drop_medication_119)가 답의 119 문장까지 지웠다 - 경고 박스·FE 119 버튼·보호자 SOS 없음.
사용자 결정(2026-11-28): 수량은 3 이상 / 중복 복용은 위험 약(당뇨약·인슐린·수면제·안정제·항응고제)만 응급, 그 밖의 약 중복은
응급이 아니고 119 문장만 지우지 않는다 / 과다 복용 응급 답에 행동 한 줄(약 상자·봉지, 억지로 토하게 하지 않기).
모델·유료 API는 부르지 않는다(가짜 로더).
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.config import Settings
from app.services import chat_keywords as kw
from app.services import chat_service as cs
from app.services.medical_llm_service import MedicalLlmService

OVERDOSE_ACTION = "드신 약의 상자나 봉지를 챙겨 119 대원에게 보여 주시고, 억지로 토하게 하지 마세요."
MODEL_119 = "약 이름을 확인해 약사에게 물어보세요. 몸이 이상하면 바로 119에 연락하세요."

EMERGENCY = [
    # 감사 재현 문장
    "수면제 5알 먹었는데 괜찮나요?",
    "감기약 3봉지 한 번에 먹었는데 되나요?",
    "어머니가 당뇨약을 두 번 드신 것 같은데 괜찮을까요?",
    # 수량(3 이상)
    "혈압약 10알을 먹었어요",
    "당뇨약 4알 먹었는데 어지러워요",
    "아스피린 7정 복용했는데 괜찮을까요",
    "남편이 신경안정제 스무 알을 먹었어요",
    "아버지가 혈압약 열 알을 드셨어요",
    "수면제 3알 먹었는데 잠이 안 와요 더 먹어도 되나요",
    # 한꺼번에·한 번에·통째로
    "아버지가 수면제를 한꺼번에 드셨어요",
    "진통제를 한 번에 6알 삼켰어요",
    "어머니가 약통에 있던 약을 통째로 드셨어요",
    "감기약을 한꺼번에 세 봉지 먹었어요",
    "캡슐 약 5캡슐을 한꺼번에 먹었어요",
    "수면유도제 한 통을 다 먹었어요",
    "엄마가 약을 한 움큼 드셨어요",
    "약을 한 번에 다 털어 넣었어요",
    # 최대 초과
    "타이레놀 하루 최대 8알인데 12알 먹었어요",
    "혈압약 최대 1알인데 2알 먹었어요",
    "진통제 하루 최대치를 넘겨서 먹었어요",
    # 위험 약 중복
    "할머니가 수면제를 두 번 드셨어요",
    "인슐린을 두 번 맞은 것 같아요",
    "아버지가 당뇨약을 또 드신 것 같은데 어떡하죠",
    "와파린을 오늘 두 번 먹었어요",
]

NORMAL = [
    "타이레놀 몇 알까지 먹어도 되나요?",
    "혈압약 아침에 먹어도 되나요?",
    "이 약은 하루에 몇 번 먹나요",
    "2알씩 먹으라는데 맞나요",
    "1알 먹었는데 효과가 없어요",
    "약 먹고 3시간 지났는데 또 먹어도 되나요",
    "처방대로 3알 먹었는데 속이 쓰려요",
    "처방대로 혈압약 3알 먹었는데 속이 쓰려요",  # 약 낱말이 있는 용법 - 용법 제외가 막는다
    "진통제를 처방받은 대로 4알 먹었는데 괜찮나요",
    "당뇨약을 아침에 3알씩 먹었는데 괜찮나요",
    "의사가 하루 3번 2알씩 먹으라고 했어요",
    "비타민D 하루 5알씩 먹으려구요 괜찮을까요",
    "유산균 3알 먹었는데 괜찮나요",
    "감기약 1포 먹었는데 졸려요",
    "혈압약이랑 당뇨약 같이 먹어도 되나요",
    "약 먹는 걸 깜빡했어요 지금 먹어도 되나요",
    "진통제 2알 먹었는데 계속 아파요",
    "수면제 반 알 먹었는데 안 졸려요",
    "하루 3번 먹는 약을 2번만 먹었어요",
    "약을 다 먹었어요 더 받아야 하나요",
    "3일 동안 감기약 먹었는데 안 나아요",
    "타이레놀 하루 최대 몇 알인가요",
    "약을 한 번에 먹으면 안 되나요",
    "한꺼번에 먹어도 되는 약인가요",
    "약 봉지 3개를 받았는데 언제 먹나요",
    "오메가3 2캡슐 먹었는데 괜찮나요",
    "과자 3봉지 먹었는데 배가 아파요",
    "당뇨약을 먹었는지 기억이 안 나요",
    "하루 두 번 드시는 당뇨약을 오늘 아침에 드셨어요",
    "진통제를 한 번에 2알 먹었는데 괜찮나요",
    "당뇨약 먹고 또 어지러워요",
]

# 위험 약이 아닌 약의 중복 - 응급은 아니지만 모델의 119 문장을 지우지 않는다(H-1 목표 2).
DUPLICATE_NOT_HIGH_RISK = [
    "혈압약을 깜빡해서 또 먹었는데 괜찮을까요?",
    "감기약을 두 번 먹었는데 괜찮을까요?",
    "할아버지가 혈압약을 아침에 두 번 드셨는데 괜찮을까요?",
    "진통제를 한 번에 2알 먹었는데 한 알 더 먹었어요 괜찮나요",
]


class _Loader:
    def __init__(self, **fields: str) -> None:
        self.fields = {"empathy": "걱정되셨겠어요.", "symptoms": "약 복용 문의", "features": "", "advice": MODEL_119, "reply": "", **fields}
        self.state = SimpleNamespace(model_name="fake-medgemma")

    def ensure_loaded(self) -> None:
        return None

    def generate_structured_text(self, system_prompt: str, user_prompt: str, prefill: str = "", **kwargs) -> str:  # noqa: ARG002
        return json.dumps(self.fields, ensure_ascii=False)


def _gen(message: str, intent: str, proxy: bool = False, **fields: str) -> dict[str, Any]:
    return MedicalLlmService(Settings(), _Loader(**fields)).generate(message, {}, [], intent=intent, proxy=proxy)


def test_lists_are_large_enough() -> None:
    assert len(EMERGENCY) >= 20 and len(NORMAL) >= 20


# ---- 응급 판정 ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("message", EMERGENCY)
def test_overdose_is_emergency_signal(message: str) -> None:
    assert MedicalLlmService.emergency_signals(message), message
    assert MedicalLlmService.classify_intent(message) == "emergency_guidance"


@pytest.mark.parametrize("message", EMERGENCY)
def test_keyword_route_marks_overdose_emergency(message: str) -> None:
    # 라우터 실패·키 없음 턴(키워드 경로)
    route = cs.ChatService._keyword_route(message)
    assert route["emergency"] is True and route["route"] == "symptom"


@pytest.mark.parametrize("router_route", ["symptom", "general", "reservation"])
@pytest.mark.parametrize("message", EMERGENCY)
def test_force_emergency_overrides_router_non_emergency(message: str, router_route: str) -> None:
    # 라우터가 응급 아님으로 낸 턴 - 라우터 뒤 보강
    forced = cs.ChatService._force_emergency({"route": router_route, "emergency": False, "engine": "llm"}, message)
    assert forced["emergency"] is True and forced["route"] == "symptom"


@pytest.mark.parametrize("message", EMERGENCY)
def test_overdose_answer_keeps_119_and_warning(message: str) -> None:
    # 라우터가 응급 아님으로 낸 턴: chat_service 와 같게 _force_emergency 결과로 intent 를 정한다 - 경고 박스·119·행동 한 줄
    forced = cs.ChatService._force_emergency({"route": "symptom", "emergency": False}, message)
    result = _gen(message, intent="emergency_guidance" if forced["emergency"] else "medical_advice")
    assert result["emergencyWarning"], message
    assert "119" in result["reply"] and "119" in result["emergencyWarning"][0]
    assert OVERDOSE_ACTION in result["reply"]
    assert result["riskLevel"] == "high"


def test_medical_advice_intent_still_gets_warning() -> None:
    # intent 가 medical_advice 로 들어와도(경로 보강이 빠진 경우) generate 자체 신호로 경고 박스·119가 남는다
    result = _gen("수면제 5알 먹었는데 괜찮나요?", intent="medical_advice")
    assert result["emergencyWarning"] and "119" in result["reply"] and OVERDOSE_ACTION in result["reply"]


def test_proxy_overdose_uses_carer_warning() -> None:
    result = _gen("어머니가 당뇨약을 두 번 드신 것 같은데 괜찮을까요?", intent="medical_advice", proxy=True)
    assert "곁에 있어 주시고" in result["emergencyWarning"][0]
    assert OVERDOSE_ACTION in result["reply"]


def test_existing_overdose_keyword_also_gets_action_sentence() -> None:
    result = _gen("약을 많이 먹었어요", intent="emergency_guidance")
    assert OVERDOSE_ACTION in result["reply"]
    assert MedicalLlmService.emergency_signals("약을 많이 먹었어요") == ["약을많이먹"]  # 기존 키워드 목록은 그대로


# ---- 과잉 응급 금지 ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("message", NORMAL)
def test_normal_medication_question_is_not_emergency(message: str) -> None:
    assert MedicalLlmService.emergency_signals(message) == [], message
    assert cs.ChatService._keyword_route(message)["emergency"] is False
    assert cs.ChatService._force_emergency({"route": "symptom", "emergency": False}, message)["emergency"] is False


@pytest.mark.parametrize("message", NORMAL)
def test_normal_medication_answer_gets_no_119(message: str) -> None:
    result = _gen(message, intent="medical_advice", advice="약 이름을 확인해 약사에게 물어보세요.")
    assert not result["emergencyWarning"]
    assert "119" not in result["reply"]
    assert OVERDOSE_ACTION not in result["reply"]


@pytest.mark.parametrize("message", [
    "진통제 몇 알까지 먹어도 되나요?", "혈압약 아침에 먹어도 되나요?", "약을 2알씩 먹어도 되나요",
    "비타민D 하루 5알씩 먹으려구요 괜찮을까요", "약 먹고 3시간 지났는데 또 먹어도 되나요",
])
def test_normal_medication_question_still_drops_model_119(message: str) -> None:
    reply, removed = MedicalLlmService.drop_medication_119(message, f"정리했어요.\n{MODEL_119}")
    assert removed == 1 and "119" not in reply


# ---- 위험 약이 아닌 중복: 응급 아님 + 119 유지 ---------------------------------------------------------------

@pytest.mark.parametrize("message", DUPLICATE_NOT_HIGH_RISK)
def test_non_high_risk_duplicate_keeps_119_without_emergency(message: str) -> None:
    assert MedicalLlmService.emergency_signals(message) == [], message
    result = _gen(message, intent="medical_advice")
    assert not result["emergencyWarning"]
    assert "119" in result["reply"]


@pytest.mark.parametrize("message", EMERGENCY + DUPLICATE_NOT_HIGH_RISK)
def test_drop_medication_119_keeps_119_when_overdose_told(message: str) -> None:
    # 응급 판정과 별개로 약 질문 119 빼기의 예외도 넓어졌다(라우터 응급 판정이 빠져도 119가 남는다)
    reply, removed = MedicalLlmService.drop_medication_119(message, f"정리했어요.\n{MODEL_119}")
    assert removed == 0 and "119" in reply


def test_overdose_told_helper_matches_lists() -> None:
    # 위험 약이 아닌 중복 중 "한 알 더"는 기존 _OVERDOSE_TOLD 가 맡는다(drop_medication_119 테스트에서 함께 본다)
    assert all(kw.overdose_told(m) for m in EMERGENCY + DUPLICATE_NOT_HIGH_RISK[:3])
    assert not any(kw.overdose_told(m) for m in NORMAL)


# ---- 로그 ---------------------------------------------------------------------------------------------------

def test_log_has_rule_names_only(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    _gen("수면제 5알 먹었는데 괜찮나요?", intent="medical_advice")
    lines = [r.getMessage() for r in caplog.records]
    assert any("emergency_action=" in line and "overdose" in line for line in lines)
    assert not any("수면제" in line or "5알" in line for line in lines)
