"""챗봇 답변 품질 후속(2026-10-29 E): 약 중단·조절 질문의 상의 문장 보장(E-1) / 규칙 8 경고 신호(E-2)."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.config import Settings
from app.prompts.medical_chat_prompts import SYSTEM_PROMPT
from app.services.medical_llm_service import MedicalLlmService

SELF_ADJUST = "약을 줄이거나 끊거나 거르는 것은 꼭 의사·약사와 상의한 뒤에 하세요."
# 답 본문에는 끊기·줄이기 말이 없다 - 질문만 보고 붙는지 본다(qa.F1에서 공감 문장이 질문을 되풀이하지 않은 경우).
PLAIN_ADVICE = "약을 더 드시거나 바꾸기 전에 꼭 의사나 약사에게 물어봐 주세요."


class _Loader:
    def __init__(self, **fields: str) -> None:
        self.fields = {"empathy": "걱정되셨겠어요.", "symptoms": "약 복용 문의", "features": "", "advice": PLAIN_ADVICE, "reply": "", **fields}
        self.state = SimpleNamespace(model_name="fake-medgemma")

    def ensure_loaded(self) -> None:
        return None

    def generate_structured_text(self, system_prompt: str, user_prompt: str, prefill: str = "", **kwargs) -> str:  # noqa: ARG002
        return json.dumps(self.fields, ensure_ascii=False)


def _gen(message: str, intent: str = "medical_advice", **fields: str) -> dict[str, Any]:
    return MedicalLlmService(Settings(), _Loader(**fields)).generate(message, {}, [], intent=intent)


@pytest.mark.parametrize("message", [
    "혈압이 정상으로 나왔으니 약을 끊어도 되죠?", "혈압약 좀 줄여도 되나요?", "당뇨약 이제 안 먹어도 될까요?",
    "약을 하루 건너뛰어도 돼요?", "콜레스테롤 약 끊으면 안 되나요?", "갑상선약 중단해도 될까요?",
])
def test_stop_or_reduce_question_gets_consult_sentence(message: str) -> None:
    out = _gen(message)
    assert out["reply"].count(SELF_ADJUST) == 1


def test_consult_sentence_not_duplicated() -> None:
    out = _gen("혈압약 끊어도 되나요?", advice=f"의사에게 물어봐 주세요. {SELF_ADJUST}")
    assert out["reply"].count(SELF_ADJUST) == 1


@pytest.mark.parametrize("message", [
    "약속을 끊어도 될까요?", "약을 먹어도 되나요?", "약을 끊었더니 괜찮아요", "감기약이랑 혈압약을 같이 먹어도 되나요",
])
def test_other_questions_unchanged(message: str) -> None:
    assert SELF_ADJUST not in _gen(message)["reply"]


def test_quitting_alcohol_is_not_stopping_medicine() -> None:
    # 술을 끊는 질문은 약을 끊는 것이 아니다 - 약 + 술 규칙만 붙는다.
    out = _gen("혈압약 먹는데 술 끊어도 되나요?")
    assert SELF_ADJUST not in out["reply"]
    assert "혈압약을 드시는 중에 술을 마시면" in out["reply"]


def test_not_applied_on_emergency() -> None:
    out = _gen("약을 끊었는데 가슴이 아파요, 약 끊어도 되나요", intent="emergency_guidance")
    assert out["emergencyWarning"]
    assert SELF_ADJUST not in out["reply"]


def test_needs_self_adjust_keeps_reply_only_callers() -> None:
    assert MedicalLlmService.needs_self_adjust_sentence("필요하면 약을 조절해 보세요.")
    assert not MedicalLlmService.needs_self_adjust_sentence("정형외과 진료를 권해드려요.")


def test_rule8_asks_for_symptom_signals_without_119() -> None:
    rule8 = next(line for line in SYSTEM_PROMPT.splitlines() if line.startswith("8)"))
    assert "신호 한두 가지" in rule8
    assert "계속되거나 심해지면 꼭 진료를 받아 보세요" in rule8  # 신호를 모를 때의 문장
    assert "위험 신호가 함께 있을 때만" in rule8  # 119 규칙은 그대로
    example1 = next(line for line in SYSTEM_PROMPT.splitlines() if line.startswith('{"empathy": "허리가'))
    assert "다리가 저려 오면" in example1 and "119" not in example1
