"""챗봇 답변 품질 후속 보완(2026-10-29 E): 저림·소변 진료과 키워드 / 소아청소년과 교체 / 약 문의 답의 119 빼기.
응급 답(키워드·라우터 응급)·과다 복용·약 + 술 답의 119는 지우지 않는다."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.config import Settings
from app.services.medical_llm_service import MedicalLlmService
from app.services.reservation_orchestrator import _symptom_department


class _Loader:
    def __init__(self, **fields: str) -> None:
        self.fields = {"empathy": "걱정되셨겠어요.", "symptoms": "증상", "features": "", "advice": "", "reply": "", **fields}
        self.state = SimpleNamespace(model_name="fake-medgemma")

    def ensure_loaded(self) -> None:
        return None

    def generate_structured_text(self, system_prompt: str, user_prompt: str, prefill: str = "", **kwargs) -> str:  # noqa: ARG002
        return json.dumps(self.fields, ensure_ascii=False)


def _gen(message: str, intent: str = "medical_advice", **fields: str) -> dict[str, Any]:
    return MedicalLlmService(Settings(), _Loader(**fields)).generate(message, {}, [], intent=intent)


# ---- 진료과 키워드 ------------------------------------------------------------------------------------


@pytest.mark.parametrize("message, department", [
    ("손발이 자꾸 저려요", "신경과"),
    ("한쪽 손이 저린 느낌이에요", "신경과"),
    ("손이 저려서 병원 예약하고 싶어요", "신경과"),
    ("허리가 아프고 다리가 저려요", "정형외과"),  # 저림은 정형외과 다음 순서
    ("무릎이 아프고 저려요", "정형외과"),
    ("요즘 소변을 자주 봐요", "내과"),
    ("빈뇨가 심해요", "내과"),
    ("머리가 아프고 손이 저려요", "신경과"),
])
def test_numbness_and_urination_departments(message: str, department: str) -> None:
    assert _symptom_department(message) == department


# ---- 소아청소년과 -------------------------------------------------------------------------------------


def test_pediatrics_replaced_with_hint() -> None:
    out = _gen("손발이 자꾸 저려요", symptoms="손발 저림", advice="소아청소년과 진료를 권해드려요.")
    assert "신경과 진료를 권해드려요" in out["reply"] and "소아청소년과" not in out["reply"]


def test_pediatrics_replaced_with_internal_medicine_without_hint() -> None:
    out = _gen("밤에 다리에 쥐가 자주 나요", symptoms="다리 쥐", advice="소아청소년과 진료를 권해드려요.")
    assert "내과 진료를 권해드려요" in out["reply"] and "소아청소년과" not in out["reply"]


@pytest.mark.parametrize("message", ["손주가 열이 나는데 소아과 가야 하나요?", "소아청소년과에 가 봐야 할까요? 아이가 기침을 해요"])
def test_pediatrics_kept_when_user_asked(message: str) -> None:
    out = _gen(message, symptoms="아이 증상", advice="소아청소년과 진료를 권해드려요.")
    assert "소아청소년과 진료를 권해드려요" in out["reply"]


# ---- 약 문의 답의 119 ---------------------------------------------------------------------------------

WRONG_119 = "혈압이 다시 떨어지거나 다른 증상이 나타나면 즉시 119에 연락하세요."


def test_medication_question_119_sentence_dropped() -> None:
    out = _gen("혈압이 정상으로 나왔으니 약을 끊어도 되죠?", symptoms="약 중단 문의",
               advice=f"약을 끊기 전에 반드시 의사나 약사에게 상담해 주세요. {WRONG_119}")
    assert "119" not in out["reply"]
    assert "약을 끊기 전에 반드시 의사나 약사에게 상담해 주세요." in out["reply"]
    assert "약을 줄이거나 끊거나 거르는 것은 꼭 의사·약사와 상의한 뒤에 하세요." in out["reply"]


@pytest.mark.parametrize("message", ["혈압약을 두 번 먹었는데 또 먹어도 되나요?", "약을 한 알 더 먹었는데 괜찮아요? 진통제 먹어도 돼요?"])
def test_overdose_told_keeps_119(message: str) -> None:
    out = _gen(message, symptoms="약 복용 문의", advice="의사·약사에게 물어봐 주세요. 몸이 이상해지면 바로 119에 연락하세요.")
    assert "몸이 이상해지면 바로 119에 연락하세요." in out["reply"]


def test_keyword_emergency_medication_question_keeps_119() -> None:
    # 키워드 응급("가슴이아") + 약 문의 - 응급 고정 안내의 119가 지워지면 안 된다.
    out = _gen("가슴이 아픈데 진통제 먹어도 돼?", symptoms="가슴 통증", advice="의사에게 물어보세요.")
    assert out["emergencyWarning"] and "119" in out["emergencyWarning"][0]
    assert "바로 119에 전화해 주세요" in out["reply"]


def test_router_emergency_medication_question_keeps_119() -> None:
    # 키워드 없이 라우터가 응급으로 본 턴(intent=emergency_guidance).
    out = _gen("혈압약 끊어도 되나요? 갑자기 너무 어지러워요", intent="emergency_guidance", symptoms="어지러움", advice="x")
    assert out["emergencyWarning"]
    assert "바로 119에 전화해 주세요" in out["reply"]


def test_drug_alcohol_answer_119_untouched() -> None:
    advice = "의사·약사와 상의하세요. 어지럽고 쓰러질 것 같으면 바로 119에 연락하세요."
    out = _gen("혈압약 먹는데 술 마셔도 돼요?", symptoms="약 + 술 문의", advice=advice)
    assert "바로 119에 연락하세요" in out["reply"]


def test_non_medication_question_119_untouched() -> None:
    advice = "내과 진료를 권해드려요. 숨이 많이 차면 바로 119에 연락하세요."
    out = _gen("계단 오르면 숨이 차요", symptoms="숨 가쁨", advice=advice)
    assert "숨이 많이 차면 바로 119에 연락하세요." in out["reply"]


def test_quality_log_counts_only(caplog) -> None:
    import logging
    caplog.set_level(logging.INFO)
    _gen("혈압이 정상으로 나왔으니 약을 끊어도 되죠?", symptoms="약 중단 문의", advice=f"상담해 주세요. {WRONG_119}")
    lines = [r.getMessage() for r in caplog.records if "[CHAT-QUALITY]" in r.getMessage()]
    assert lines and "medication_119_dropped=1" in lines[0] and "혈압" not in lines[0]
