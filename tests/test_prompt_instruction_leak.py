"""지시문 노출 문장 제거(2026-11-18) - 프롬프트 지시 문장을 20자 이상 그대로 옮긴 문장만 빼고 정상 답·서버 고정 문장은 남긴다."""
from __future__ import annotations

import pytest

from app.services.medical_llm_service import MedicalLlmService

LEAKED = [
    "답을 짧게 써야 할 때도 이 첫 문장은 빼지 않고, 진료과 권유나 진료 기준으로 시작하지 않는다.",
    "이 질문은 증상이 있을 때 무엇을 먹거나 어떻게 하면 좋은지 묻는다.",
    "약을 더 먹어도 되는지, 몇 알인지, 같이 먹어도 되는지(음식·영양제·다른 약), 끊어도 되는지 묻는 질문이면 괜찮다거나 안 된다고 단정하지 말고, 늘리거나 줄이거나 끊으라고도 하지 마라.",
]
KEPT = [
    "기침할 때 피해야 할 것은 흡연과 음주입니다.",
    "무릎 통증이 계속되거나 심해지면 꼭 진료를 받아 보세요.",
    "정형외과 진료를 권해드려요.",
    "며칠이 지나도 나아지지 않거나 통증이 점점 심해지면 꼭 진료를 받아 보세요.",
    "약을 줄이거나 끊거나 거르는 것은 꼭 의사·약사와 상의한 뒤에 하세요.",
    "숨이 차거나 피 섞인 기침이 나오거나 삼키기 어려우면 바로 응급실 진료를 받으세요.",
    "물이나 부드러운 음식을 조금씩 자주 드세요.",
]


@pytest.mark.parametrize("sentence", LEAKED)
def test_leaked_instruction_sentence_is_dropped(sentence: str) -> None:
    out, removed = MedicalLlmService.drop_prompt_leak(f"기침할 때 피해야 할 것은 흡연과 음주입니다. {sentence}")
    assert removed == 1 and sentence not in out and "흡연과 음주" in out


@pytest.mark.parametrize("sentence", KEPT)
def test_normal_sentence_is_kept(sentence: str) -> None:
    assert MedicalLlmService.drop_prompt_leak(sentence) == (sentence, 0)
