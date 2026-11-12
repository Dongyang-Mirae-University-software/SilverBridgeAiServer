"""칩(꼬리 질문) 답 고정 줄(2026-11-12, 칩 시험지 cp.136/137·cp.155).

DGX 실제 MedGemma 재생에서 "찜질은 따뜻하게 해요, 차갑게 해요?"·"혈압약은 언제 먹어야 해요?"의 답에 질문이 묻는 내용
(온·냉 구분 / 복용 시간)이 없고 진료 권유만 남았다. 정해진 답이 있는 꼴이라 생성에 맡기지 않고 고정 줄을 붙인다
(_LIFESTYLE_RULES.hot_cold_pack · _DRUG_DIRECT_RULES.med_timing). 약 이름별 시간·처방은 말하지 않는다.
"""
from __future__ import annotations

import pytest

from app.services import chat_keywords as kw
from app.services.medical_llm_service import MedicalLlmService as M
from tests.test_chat_sleeping_pill_breath_lifestyle import _gen

PACK = "다친 지 2~3일 안이거나 붓고 열감이 있으면 차갑게, 붓기 없이 뻐근하고 오래된 통증이면 따뜻하게 찜질하세요. 한 번에 20분 안쪽으로, 피부에 바로 대지 말고 수건에 싸서 하세요."
TIMING = "약은 처방받은 시간대로 매일 같은 시간에 드시고, 시간을 바꾸고 싶으면 의사·약사와 먼저 상의하세요."


@pytest.mark.parametrize("message", ["찜질은 따뜻하게 해요, 차갑게 해요?", "무릎에 냉찜질이 좋아요?", "허리에 온찜질해도 돼요?",
                                     "따뜻하게 해야 해요 차갑게 해야 해요?"])
def test_hot_cold_pack_question_gets_rule_line(message: str) -> None:
    assert PACK in _gen(message)["reply"]


@pytest.mark.parametrize("message", ["혈압약은 언제 먹어야 해요?", "당뇨약 몇 시에 먹어요?", "이 약 식후에 먹어야 돼요?",
                                     "혈압약 아침에 드셔야 하나요?"])
def test_med_timing_question_gets_rule_line(message: str) -> None:
    assert TIMING in _gen(message)["reply"]


@pytest.mark.parametrize("message", ["찜질방 가도 돼요?", "약국 언제 열어요?", "혈압약 먹고 술 언제 마셔도 돼요?", "무릎이 아파요",
                                     "언제 병원에 가야 해요?"])
def test_other_questions_get_no_rule_line(message: str) -> None:
    reply = _gen(message)["reply"]
    assert PACK not in reply and TIMING not in reply


def test_rule_line_skipped_when_answer_already_says_it() -> None:
    covered = "다친 지 이틀 안이고 부으면 차갑게, 오래된 뻐근함은 따뜻하게 찜질하세요."
    assert PACK not in _gen("찜질은 따뜻하게 해요, 차갑게 해요?", advice=covered)["reply"]
    covered_med = "혈압약은 매일 같은 시간에 드시고, 바꾸려면 약사와 상의하세요."
    assert TIMING not in _gen("혈압약은 언제 먹어야 해요?", advice=covered_med)["reply"]


def test_proxy_wording() -> None:
    lines = dict(M.drug_direct_sentences("어머니 혈압약은 언제 드셔야 해요?", "", proxy=True))
    assert lines["med_timing"].endswith("상의해 주세요.") and "드시게 해 주시고" in lines["med_timing"]
    lines = dict(M.drug_direct_sentences("찜질은 따뜻하게 해 드려요, 차갑게 해 드려요?", "", proxy=True))
    assert "찜질해 드리세요" in lines["hot_cold_pack"]


def test_hot_cold_pack_is_care_question_but_sauna_is_not() -> None:
    assert M.is_care_question("찜질은 따뜻하게 해요, 차갑게 해요?", "medical_advice", False)
    assert not M.is_care_question("찜질방 가도 돼요?", "medical_advice", False)


@pytest.mark.parametrize("message,department", [("배 아플 때 피해야 할 음식이 있나요?", "내과"), ("머리 아플 때 뭘 하면 좋아요?", "신경과"),
                                                ("허리 아플 때 피할 자세", "정형외과")])
def test_department_hint_reads_apeul(message: str, department: str) -> None:
    # "아플 때" 꼴 칩도 진료과 힌트가 잡힌다(전에는 "아프/아파"만 알아 "가까운 병원"이 됐다).
    assert kw.symptom_department(message) == department


def test_emergency_answer_gets_no_rule_line() -> None:
    reply = _gen("가슴이 조이고 숨이 차요 찜질해도 돼요?", intent="emergency_guidance")["reply"]
    assert PACK not in reply


# ---- #264·#265 기능 점검 반영 (2026-11-12) -------------------------------------------------------------


@pytest.mark.parametrize("message", ["화상에 찜질해도 돼요?", "뜨거운 물에 데었는데 찜질해요?", "배에 찜질해도 돼요?", "다래끼에 온찜질 해요?",
                                     "머리 부딪혔는데 찜질해요?", "가슴이 아픈데 찜질해도 돼요?", "눈에 냉찜질 해도 돼요?"])
def test_pack_line_not_for_burn_belly_eye_chest_head(message: str) -> None:
    # S-1·H-3: 다친 관절·근육 기준 고정 줄이 화상(온찜질 금지)·배·눈·가슴·머리에 붙지 않는다.
    assert "hot_cold_pack" not in dict(M.drug_direct_sentences(message, ""))


def test_burn_is_not_care_question() -> None:
    assert not M.is_care_question("화상에 찜질해도 돼요?", "medical_advice", False)
    assert M.is_care_question("무릎에 냉찜질이 좋아요?", "medical_advice", False)


@pytest.mark.parametrize("message", ["약을 언제까지 먹어요?", "약 먹고 언제 밥 먹어요?", "혈압약 깜빡했는데 언제 먹어요?"])
def test_med_timing_not_for_duration_meal_or_missed_dose(message: str) -> None:
    # M-1: 기간·식사·놓친 약 질문에는 복용 시간 줄을 붙이지 않는다(놓친 약은 놓친 약 규칙).
    assert "med_timing" not in dict(M.drug_direct_sentences(message, ""))


def test_missed_dose_still_gets_missed_dose_line() -> None:
    assert "missed_dose_timing" in dict(M.drug_direct_sentences("혈압약 깜빡했는데 언제 먹어요?", ""))
