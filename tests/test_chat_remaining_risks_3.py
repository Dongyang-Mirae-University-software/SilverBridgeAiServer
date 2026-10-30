"""남은 위험 처리 3차(K) - E2 고정 문장 늘 붙이기(K-1)·수면제 처방 말 섞임(K-2)·약 상투 문장(K-3)·저혈당 재측정·감기약 성분·저림 뇌졸중 신호(K-4) (2026-10-30)."""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.config import Settings
from app.services.medical_llm_service import MedicalLlmService

CALL_119 = "지금은 병원 예약보다 119 연락이 먼저예요. 바로 119에 전화해 주세요."
BP_EX = "혈압이 평소보다 높게 나온 날은 운동을 쉬거나 가볍게 하시고, 가슴 통증·어지럼이 있으면 바로 멈추세요."
SLEEP = "약국에서 살 수 있는 수면유도제(일반의약품)도 있지만, 수면제는 처방이 필요해요. 드시기 전에 약사와 상의하세요."
BOILER = "약 복용이나 다른 대처 방법에 대해 궁금하신 점은 의사나 약사에게 물어봐 주세요."
HYPO = "의식이 있으면 사탕이나 단 음료를 드시고, 의식이 흐리면 억지로 먹이지 마세요."
RECHECK = "15분쯤 뒤 혈당을 다시 재 보고, 계속 낮거나 증상이 이어지면 의사에게 알리세요."
COLD = "어르신은 감기약에도 부작용이 생길 수 있어요. 드시는 약과 같은 성분이 겹치지 않는지 약 상자의 성분표를 약사와 함께 확인하세요."
NUMB = "한쪽 얼굴·팔·다리가 갑자기 저리거나 힘이 빠지면 바로 응급실로 가세요."


class _Loader:
    def __init__(self, **fields: str) -> None:
        self.fields = {"empathy": "걱정되셨겠어요.", "symptoms": "증상", "features": "", "advice": "내과 진료를 권해드려요.", "reply": "", **fields}
        self.state = SimpleNamespace(model_name="fake-medgemma")

    def ensure_loaded(self) -> None:
        return None

    def generate_structured_text(self, system_prompt: str, user_prompt: str, prefill: str = "", **kwargs) -> str:  # noqa: ARG002
        return json.dumps(self.fields, ensure_ascii=False)


def _gen(message: str, intent: str = "medical_advice", **fields: str) -> dict[str, Any]:
    return MedicalLlmService(Settings(), _Loader(**fields)).generate(message, {}, [], intent=intent)


# ---- K-1 혈압 높은 날 운동 ------------------------------------------------------------------------------


def test_bp_exercise_sentence_added_even_with_vague_model_sentence() -> None:
    advice = "운동을 하신 후에도 혈압이 높게 측정된다면 운동을 멈추세요."
    reply = _gen("혈압이 높은 날 운동해도 되나요", advice=advice)["reply"]
    assert reply.count(BP_EX) == 1


def test_bp_exercise_sentence_not_duplicated() -> None:
    assert _gen("혈압이 높은 날 운동해도 되나요", advice=BP_EX)["reply"].count(BP_EX) == 1


# ---- K-2 수면제 처방 말 섞임 ----------------------------------------------------------------------------


def test_sleeping_pill_hedge_dropped() -> None:
    hedge = "수면제는 처방이 필요할 수 있으니 약사와 상의하세요."
    reply = _gen("수면제를 약국에서 그냥 살 수 있나요", advice=f"{hedge} 걱정되시면 진료를 받아 보세요.")["reply"]
    assert hedge not in reply and SLEEP in reply


def test_hedge_kept_for_other_questions() -> None:
    advice = "이 약은 처방이 필요할 수 있으니 약사와 상의하세요."
    assert advice in _gen("소화제 약국에서 사도 돼요?", advice=advice)["reply"]


# ---- K-3 약 상투 문장 -----------------------------------------------------------------------------------


@pytest.mark.parametrize("message", ["요즘 소변을 자주 봐요", "혈압이 높은 날 운동해도 되나요", "눈이 침침하고 자주 피곤해요"])
def test_boilerplate_dropped_without_medication_in_question(message: str) -> None:
    reply = _gen(message, advice=f"내과 진료를 권해드려요. {BOILER}")["reply"]
    assert BOILER not in reply and "진료를 권해드려요." in reply  # 진료과는 힌트로 맞춰질 수 있다(눈 → 안과)


@pytest.mark.parametrize("message", ["혈압약 먹고 어지러워요", "감기약이랑 혈압약을 같이 먹어도 되나요", "당뇨약 먹는데 술을 마셔도 되나요"])
def test_boilerplate_kept_for_medication_questions(message: str) -> None:
    assert BOILER in _gen(message, advice=f"의사와 상의하세요. {BOILER}")["reply"]


def test_boilerplate_only_answer_still_has_clinic() -> None:
    reply = _gen("요즘 소변을 자주 봐요", advice=BOILER)["reply"]
    assert BOILER not in reply and "진료" in reply


# ---- K-4 저혈당 재측정·감기약 성분·저림 뇌졸중 신호 ------------------------------------------------------


def test_hypoglycemia_recheck_after_119_and_action() -> None:
    out = _gen("혈당이 40밖에 안 나오고 몸이 떨려요", intent="emergency_guidance", advice="x")
    lines = out["reply"].split("\n")
    assert CALL_119 in lines and HYPO in lines and RECHECK in lines
    assert lines.index(CALL_119) < lines.index(HYPO) < lines.index(RECHECK)
    assert out["emergencyWarning"] == [MedicalLlmService.EMERGENCY_WARNING]


def test_no_recheck_for_high_glucose() -> None:
    reply = _gen("혈당이 300이에요", intent="emergency_guidance", advice="x")["reply"]
    assert RECHECK not in reply and HYPO not in reply


@pytest.mark.parametrize("message", ["감기약이랑 혈압약을 같이 먹어도 되나요", "혈압약 먹는데 감기약 먹어도 돼요?"])
def test_cold_medicine_combo_sentence(message: str) -> None:
    assert _gen(message, advice="의사와 상의하세요.")["reply"].count(COLD) == 1


def test_cold_medicine_sentence_skipped_when_covered() -> None:
    assert COLD not in _gen("감기약이랑 혈압약 같이 먹어도 되나요", advice="같은 성분이 들어 있는지 약사와 확인하세요.")["reply"]


@pytest.mark.parametrize("message", ["손발이 자꾸 저려요", "다리가 저린 느낌이 있어요"])
def test_numbness_stroke_sign(message: str) -> None:
    reply = _gen(message, advice="신경과 진료를 권해드려요.")["reply"]
    assert reply.count(NUMB) == 1 and "119" not in reply


def test_numbness_sentence_skipped_when_covered_or_other() -> None:
    assert NUMB not in _gen("손이 저려요", advice="한쪽 팔이 저리면 바로 진료를 받으세요.")["reply"]
    assert NUMB not in _gen("무릎이 아파요", advice="정형외과 진료를 권해드려요.")["reply"]


def test_self_harm_fixed_reply_untouched() -> None:
    reply = MedicalLlmService.self_harm_response()["reply"]
    assert not any(s in reply for s in (BP_EX, RECHECK, COLD, NUMB))


def test_k_logs_names_only(caplog) -> None:
    caplog.set_level(logging.INFO)
    _gen("요즘 소변을 자주 봐요", advice=f"내과 진료를 권해드려요. {BOILER}")
    _gen("혈당이 40밖에 안 나와요", intent="emergency_guidance", advice="x")
    msgs = [r.getMessage() for r in caplog.records]
    assert "[CHAT-QUALITY] medication_boilerplate_dropped=1" in msgs
    assert "[CHAT-SAFETY] emergency_action=hypoglycemia,hypoglycemia_recheck" in msgs
    assert not any("소변" in m or "혈당이" in m for m in msgs)
