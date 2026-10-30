"""남은 위험 처리(I) - 이어지는 증상 진료 권유(I-1)·열 + 어지럼 내과(I-2)·프롬프트 문구 노출(I-3)·D1/E3/F4 고정 문장(I-4) (2026-10-30)."""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.config import Settings
from app.services.medical_llm_service import MedicalLlmService
from app.services.reservation_orchestrator import _symptom_department

CONDITIONAL = "신경과 진료를 권해드려요. 두통이 계속되거나 심해지면 꼭 진료를 받아 보세요."
CALL_119 = "지금은 병원 예약보다 119 연락이 먼저예요. 바로 119에 전화해 주세요."
MISSED = "놓친 약은 생각났을 때 바로 드시되, 다음 드실 시간이 가까우면 놓친 것은 건너뛰고 원래 시간에 한 번만 드세요. 약마다 다를 수 있으니 약사에게 확인해 보세요."
DOUBLE = "놓친 약을 다음에 두 번 몫으로 한꺼번에 드시지는 마세요."
SELF_ADJUST = "약을 줄이거나 끊거나 거르는 것은 꼭 의사·약사와 상의한 뒤에 하세요."
VIRUS = "감기·독감 같은 바이러스에는 항생제가 듣지 않고, 항생제는 그때 본인에게 처방된 것만 드셔야 해요."
LEFTOVER = "남은 항생제를 임의로 드시지 마세요. 열이 나면 진료를 받아 보세요."
CHOL = "기름진 고기·튀김·가공식품의 포화지방·트랜스지방은 줄이고, 채소·콩·통곡물·해조류처럼 식이섬유가 많은 음식을 늘려 보세요."
LEAK = "약 복용 문의이므로 119 문장은 붙이지 않습니다."


class _Loader:
    def __init__(self, **fields: str) -> None:
        self.fields = {"empathy": "걱정되셨겠어요.", "symptoms": "문의", "features": "", "advice": CONDITIONAL, "reply": "", **fields}
        self.state = SimpleNamespace(model_name="fake-medgemma")

    def ensure_loaded(self) -> None:
        return None

    def generate_structured_text(self, system_prompt: str, user_prompt: str, prefill: str = "", **kwargs) -> str:  # noqa: ARG002
        return json.dumps(self.fields, ensure_ascii=False)


def _gen(message: str, intent: str = "medical_advice", **fields: str) -> dict[str, Any]:
    return MedicalLlmService(Settings(), _Loader(**fields)).generate(message, {}, [], intent=intent)


# ---- I-1 이미 이어지는 증상 -----------------------------------------------------------------------------


@pytest.mark.parametrize("message, department", [
    ("머리가 아파요 / 일주일째인데 점점 심해져요", "신경과"),
    ("귀가 아파요 / 어제부터 아프고 귀에서 물이 나와요", "이비인후과"),
    ("허리가 2주째 안 나아요", "정형외과"),
    ("어지러워요 / 어제부터요, 열도 조금 있어요", "내과"),
])
def test_persisting_symptom_gets_direct_visit(message: str, department: str) -> None:
    reply = _gen(message)["reply"]
    assert reply.count(f"말씀하신 상태라면 미루지 말고 {department} 진료를 받아 보세요.") == 1
    assert "꼭 진료를 받아 보세요." in reply  # 모델의 조건부 문장(경고 신호가 들 수 있다)은 지우지 않는다


@pytest.mark.parametrize("message", ["며칠째 기침이 나고 목이 아파요", "머리가 아파요", "아침에 일어나면 어지러워요", "심해지면 어떡하죠"])
def test_not_persisting_untouched(message: str) -> None:
    assert "미루지 말고" not in _gen(message)["reply"]


def test_persisting_skipped_when_answer_already_urges() -> None:
    reply = _gen("머리가 일주일째 아파요", advice="신경과 진료를 권해드려요. 미루지 말고 진료를 받아 보세요.")["reply"]
    assert reply.count("미루지") == 1


def test_persisting_not_on_emergency_or_medication_or_drug_alcohol() -> None:
    out = _gen("일주일째 가슴이 아파요", intent="emergency_guidance", advice="x")
    assert CALL_119 in out["reply"] and "미루지 말고" not in out["reply"]
    assert "미루지 말고" not in _gen("열이 나는데 남은 항생제를 먹어도 되죠?")["reply"]
    assert "미루지 말고" not in _gen("일주일째 혈압약 먹는데 술 마셔도 돼요?")["reply"]


# ---- I-2 열 + 어지럼·두통 -------------------------------------------------------------------------------


@pytest.mark.parametrize("text, department", [
    ("어지러워요 / 어제부터요, 열도 조금 있어요", "내과"),
    ("머리가 아프고 열이 나요", "내과"),
    ("미열이 있고 두통이 있어요", "내과"),
    ("머리가 아파요", "신경과"),
    ("열심히 일하고 나면 머리가 아파요", "신경과"),
    ("문을 열어 두니 어지러워요", "신경과"),
    ("귀가 아프고 열이 나요", "이비인후과"),
    ("눈이 충혈되고 열이 나요", "안과"),
])
def test_fever_moves_neurology_to_internal(text: str, department: str) -> None:
    assert _symptom_department(text) == department


def test_fever_dizziness_answer_aligned_to_internal() -> None:
    reply = _gen("어지러워요 / 어제부터요, 열도 조금 있어요", advice="신경과 진료를 권해드려요. 어지럼이 계속되면 꼭 진료를 받아 보세요.")["reply"]
    assert "내과 진료를 권해드려요." in reply and "신경과" not in reply


# ---- I-3 프롬프트 문구 노출 ------------------------------------------------------------------------------


@pytest.mark.parametrize("leak", [LEAK, "규칙 8에 따라 진료과를 권합니다.", "advice에는 진료과를 씁니다.", "예시 1처럼 정리했어요.", "[권장 진료과 힌트]를 따랐어요."])
def test_prompt_leak_sentence_dropped(leak: str) -> None:
    reply = _gen("콜레스테롤이 높다는데 식단은 어떻게 해야 하나요", advice=f"내과 진료를 권해드려요. {leak}")["reply"]
    assert leak not in reply and "내과 진료를 권해드려요." in reply


def test_prompt_leak_all_dropped_gets_consult_line() -> None:
    reply = MedicalLlmService.drop_prompt_leak(LEAK)
    assert reply == ("궁금하신 점은 병원 진료 때 의사와 상의해 보세요.", 1)


def test_normal_sentences_kept() -> None:
    text = "걱정되셨겠어요.\n내과 진료를 권해드려요. 기침이 3주 넘게 계속되면 꼭 진료를 받아 보세요."
    assert MedicalLlmService.drop_prompt_leak(text) == (text, 0)


def test_prompt_leak_not_touched_on_emergency() -> None:
    out = _gen("가슴이 아파요", intent="emergency_guidance", advice="x", features=LEAK)
    assert CALL_119 in out["reply"]
    assert out["emergencyWarning"] == [MedicalLlmService.EMERGENCY_WARNING]


# ---- I-4 D1·F4·E3 고정 문장 -----------------------------------------------------------------------------


def test_missed_dose_gets_timing_and_double_dose() -> None:
    reply = _gen("혈압약을 깜빡하고 안 먹었어요", advice="약을 더 드시거나 바꾸기 전에 꼭 의사나 약사에게 물어봐 주세요.")["reply"]
    assert reply.count(MISSED) == 1 and reply.count(DOUBLE) == 1
    # "건너뛰고"가 든 고정 문장 때문에 약 임의 조절 경고가 덧붙지 않는다
    assert SELF_ADJUST not in reply


def test_missed_dose_timing_skipped_when_covered_or_overdose() -> None:
    reply = _gen("약을 깜빡했어요", advice="생각나면 바로 드시고, 다음 시간이 가까우면 건너뛰세요. 의사와 상의하세요.")["reply"]
    assert MISSED not in reply
    assert MISSED not in _gen("약을 깜빡해서 두 번 먹었어요")["reply"]


def test_leftover_antibiotic_gets_virus_sentence() -> None:
    reply = _gen("열이 나는데 병원 가지 말고 남은 항생제를 먹어도 되죠?", advice="의사나 약사에게 물어봐 주세요.")["reply"]
    assert reply.count(LEFTOVER) == 1 and reply.count(VIRUS) == 1


def test_virus_sentence_skipped_when_covered() -> None:
    reply = _gen("감기인데 항생제 먹어도 돼요?", advice="감기는 대부분 바이러스라 항생제가 듣지 않아요. 의사와 상의하세요.")["reply"]
    assert VIRUS not in reply


@pytest.mark.parametrize("message", ["콜레스테롤이 높다는데 식단은 어떻게 해야 하나요", "고지혈증에 좋은 음식 알려 주세요", "중성지방 높으면 뭘 먹어야 해요"])
def test_cholesterol_diet_sentence(message: str) -> None:
    assert _gen(message, advice="내과 진료를 권해드려요.")["reply"].count(CHOL) == 1


def test_cholesterol_sentence_skipped_when_covered_or_other() -> None:
    reply = _gen("콜레스테롤 높은데 식단은요?", advice="포화지방을 줄이고 식이섬유를 늘리세요.")["reply"]
    assert CHOL not in reply
    assert CHOL not in _gen("콜레스테롤 수치가 뭔가요", advice="내과 진료를 권해드려요.")["reply"]


def test_drug_alcohol_answers_unchanged() -> None:
    reply = _gen("당뇨약 먹는데 술을 마셔도 되나요", advice="의사와 상의해 보세요.")["reply"]
    assert not any(s in reply for s in (MISSED, VIRUS, CHOL)) and "미루지 말고" not in reply


def test_self_harm_fixed_reply_untouched() -> None:
    reply = MedicalLlmService.self_harm_response()["reply"]
    assert not any(s in reply for s in (MISSED, VIRUS, CHOL, "미루지 말고"))


def test_i_logs_names_only(caplog) -> None:
    caplog.set_level(logging.INFO)
    _gen("혈압약을 깜빡하고 안 먹었어요", advice=f"의사와 상의하세요. {LEAK}")
    _gen("머리가 일주일째 아파요")
    msgs = [r.getMessage() for r in caplog.records]
    assert "[CHAT-SAFETY] prompt_leak_dropped=1" in msgs
    assert any(m.startswith("[CHAT-SAFETY] drug_direct=missed_dose_timing,double_dose") for m in msgs)
    assert any("persisting_visit=True" in m for m in msgs)
    assert not any("깜빡" in m or "일주일" in m for m in msgs)
