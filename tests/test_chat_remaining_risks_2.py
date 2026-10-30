"""남은 위험 처리 2차(J) - 수면제 단정 변형(J-1)·증상 칸 분석 문장(J-2)·응급 설명(J-3)·증상 고정 문장(J-4)·약 + 술 답(J-5) (2026-10-30)."""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.config import Settings
from app.services.medical_llm_service import MedicalLlmService

CALL_119 = "지금은 병원 예약보다 119 연락이 먼저예요. 바로 119에 전화해 주세요."
SLEEP = "약국에서 살 수 있는 수면유도제(일반의약품)도 있지만, 수면제는 처방이 필요해요. 드시기 전에 약사와 상의하세요."
NO_DRIVE = "직접 운전해서 병원에 가지 마세요."
CHEST_DRIVE = "직접 운전하거나 혼자 움직이지 마세요."
BLOOD_INFO = "커피색이나 검은색으로 토하는 것은 위나 장의 출혈일 수 있어요."
MI_INFO = "소화불량처럼 느껴져도 심근경색 증상일 수 있어요."
COUGH = "숨이 차거나 피 섞인 기침이 나오거나 삼키기 어려우면 바로 응급실 진료를 받으세요."
ORTHO = "누웠다 일어날 때는 천천히 일어나시고, 물을 충분히 드세요."
VISION = "갑자기 앞이 안 보이거나 시야가 가려지면 바로 응급실이나 안과로 가세요."
SELF_ADJUST = "약을 줄이거나 끊거나 거르는 것은 꼭 의사·약사와 상의한 뒤에 하세요."
NEW_SENTENCES = (NO_DRIVE, BLOOD_INFO, MI_INFO, COUGH, ORTHO, VISION)


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


# ---- J-1 수면제 단정 변형 -------------------------------------------------------------------------------


@pytest.mark.parametrize("wrong", ["수면제는 약국에서 직접 구입할 수 있는 약은 아니에요.", "수면제는 처방 없이 약국에서 살 수 있는 약이 아니에요."])
def test_sleeping_pill_misstatement_variants_dropped(wrong: str) -> None:
    reply = _gen("수면제를 약국에서 그냥 살 수 있나요", advice=f"{wrong} 의사나 약사에게 물어봐 주세요.")["reply"]
    assert wrong not in reply and SLEEP in reply


def test_misstatement_variant_kept_for_other_questions() -> None:
    advice = "이 약은 약국에서 직접 구입할 수 있는 약은 아니에요. 의사와 상의하세요."
    assert advice in _gen("혈압약이 떨어졌는데 어떻게 해요?", advice=advice)["reply"]


# ---- J-2 증상 칸 분석 문장 -------------------------------------------------------------------------------


def test_analysis_note_dropped_from_symptoms() -> None:
    reply = _gen("수면제를 약국에서 그냥 살 수 있나요", symptoms="약 복용 문의 (수면제). 복용 중인 약은 언급되지 않음.")["reply"]
    assert "언급되지" not in reply and "· 증상: 약 복용 문의 (수면제)." in reply


def test_symptoms_all_notes_fall_back_to_question() -> None:
    reply = _gen("무릎이 아파요", symptoms="기간은 명시되지 않음.", advice="정형외과 진료를 권해드려요.")["reply"]
    assert "명시되지" not in reply and "정형외과 진료를 권해드려요." in reply


def test_normal_symptoms_untouched() -> None:
    assert "· 증상: 허리 통증 (일주일째)" in _gen("허리가 일주일째 아파요", symptoms="허리 통증 (일주일째)")["reply"]


# ---- J-3 응급 답 설명 ------------------------------------------------------------------------------------


def _lines(message: str) -> list[str]:
    out = _gen(message, intent="emergency_guidance", advice="x")
    assert out["emergencyWarning"] == [MedicalLlmService.EMERGENCY_WARNING]
    return out["reply"].split("\n")


def test_breathing_gets_no_drive_after_119() -> None:
    lines = _lines("숨이 안 쉬어지고 입술이 파래요")
    assert CALL_119 in lines and NO_DRIVE in lines and lines.index(CALL_119) < lines.index(NO_DRIVE)


def test_breathing_with_chest_keeps_single_drive_line() -> None:
    reply = "\n".join(_lines("가슴이 조이고 숨쉬기 힘들어요"))
    assert CHEST_DRIVE in reply and NO_DRIVE not in reply


def test_coffee_vomit_gets_bleeding_info() -> None:
    lines = _lines("토한 것이 커피색이에요")
    assert CALL_119 in lines and BLOOD_INFO in lines and lines.index(CALL_119) < lines.index(BLOOD_INFO)


def test_chest_indigestion_gets_mi_info() -> None:
    reply = "\n".join(_lines("가슴이 아픈 건 소화불량이니까 그냥 참으면 되죠?"))
    assert CALL_119 in reply and CHEST_DRIVE in reply and MI_INFO in reply
    assert MI_INFO not in "\n".join(_lines("가슴이 아파요"))


def test_emergency_info_not_on_general_answers() -> None:
    reply = _gen("숨이 좀 차요", advice="내과 진료를 권해드려요.")["reply"]
    assert not any(s in reply for s in (NO_DRIVE, BLOOD_INFO, MI_INFO))


# ---- J-4 일반 증상 고정 문장 ----------------------------------------------------------------------------


@pytest.mark.parametrize("message, sentence", [
    ("며칠째 기침이 나고 목이 아파요", COUGH),
    ("목이 따가워요", COUGH),
    ("아침에 일어나면 어지러워요", ORTHO),
    ("앉았다 일어설 때 어지러워요", ORTHO),
    ("눈이 침침하고 자주 피곤해요", VISION),
    ("요즘 시력이 떨어진 것 같아요", VISION),
])
def test_symptom_sentence_once(message: str, sentence: str) -> None:
    reply = _gen(message)["reply"]
    assert reply.count(sentence) == 1 and "119" not in reply


def test_symptom_sentence_skipped_when_covered() -> None:
    assert COUGH not in _gen("기침이 나요", advice="이비인후과 진료를 권해드려요. 피 섞인 가래가 나오면 바로 진료를 받으세요.")["reply"]
    assert ORTHO not in _gen("일어나면 어지러워요", advice="천천히 일어나시고 물을 자주 드세요.")["reply"]
    assert VISION not in _gen("눈이 침침해요", advice="안과 진료를 권해드려요. 갑자기 시력이 떨어지면 바로 오세요.")["reply"]


@pytest.mark.parametrize("message", ["무릎이 아파요", "어지러워요", "속이 더부룩해요"])
def test_symptom_sentences_not_for_other_questions(message: str) -> None:
    reply = _gen(message)["reply"]
    assert not any(s in reply for s in (COUGH, ORTHO, VISION))


def test_cough_sentence_not_on_emergency() -> None:
    reply = "\n".join(_lines("기침하다 피를 토했어요"))
    assert COUGH not in reply


# ---- J-5 약 + 술 답 -------------------------------------------------------------------------------------


def test_drug_alcohol_example_copy_variant_dropped() -> None:
    reply = _gen("당뇨약 먹는데 술을 마셔도 되나요", advice="술을 마시거나 바꾸기 전에 꼭 의사와 상의하세요.")["reply"]
    assert "바꾸기 전에" not in reply


def test_drug_alcohol_skip_dose_sentence_replaced_by_consult() -> None:
    advice = "술을 마실 때는 당뇨약을 복용하지 않거나 의사와 상의하세요."
    reply = _gen("당뇨약 먹는데 술을 마셔도 되나요", advice=advice)["reply"]
    assert advice not in reply and "복용하지 않거나" not in reply
    assert reply.count(SELF_ADJUST) == 1
    assert "저혈당" in reply  # 약 + 술 고정 문장은 그대로


def test_drug_alcohol_without_adjust_untouched() -> None:
    reply = _gen("당뇨약 먹는데 술을 마셔도 되나요", advice="술은 피하시는 게 좋아요. 의사와 상의하세요.")["reply"]
    assert SELF_ADJUST not in reply and "술은 피하시는 게 좋아요." in reply


def test_self_harm_fixed_reply_untouched() -> None:
    reply = MedicalLlmService.self_harm_response()["reply"]
    assert not any(s in reply for s in NEW_SENTENCES + (SELF_ADJUST,))


def test_j_logs_names_only(caplog) -> None:
    caplog.set_level(logging.INFO)
    _gen("토한 것이 커피색이에요", intent="emergency_guidance", advice="x")
    _gen("당뇨약 먹는데 술을 마셔도 되나요", advice="술을 마실 때는 당뇨약을 복용하지 않거나 의사와 상의하세요.")
    _gen("아침에 일어나면 어지러워요")
    msgs = [r.getMessage() for r in caplog.records if "[CHAT-SAFETY]" in r.getMessage()]
    assert "[CHAT-SAFETY] emergency_action=vomiting_blood,vomiting_blood_info" in msgs
    assert any("drug_alcohol dropped={'self_adjust': 1}" in m for m in msgs)
    assert "[CHAT-SAFETY] drug_direct=orthostatic_dizziness" in msgs
    assert not any("커피색" in m or "당뇨약" in m for m in msgs)
