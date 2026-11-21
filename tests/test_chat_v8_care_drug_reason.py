"""시험지 v8 holdout-A 3차 판정(2026-11-22) 반영 - 집 관리 질문의 행동 + 횟수·시점(H1), 감기약 같이 먹기의 위험 이유(G1).

시험지 문장을 그대로 넣지 않고 같은 유형의 다른 문장으로 고정한다. 수치는 공인 권고에 있는 것만(근력 운동 주 2일 이상, 냉찜질 20분 안쪽).
"""
from __future__ import annotations

import re

import pytest

from app.services import medical_llm_service as mls
from app.services.medical_llm_service import MedicalLlmService as M


def _names(message: str, reply: str = "모델 답", proxy: bool = False) -> dict[str, str]:
    return dict(M.drug_direct_sentences(message, reply, proxy))


@pytest.mark.parametrize("message,rule", [
    ("허벅지 근육을 키우려면 뭘 하면 좋아요?", "leg_strength_exercise"),
    ("하체 힘이 없는데 운동은 어떻게 해요?", "leg_strength_exercise"),
    ("근력 운동은 어떤 걸 해야 해요?", "leg_strength_exercise"),
    ("넘어지지 않게 욕실을 어떻게 바꾸면 돼요?", "fall_home_safety"),
    ("집에서 낙상을 막으려면 바닥을 어떻게 해요?", "fall_home_safety"),
    ("어깨가 결리는데 집에서 뭘 하면 돼요?", "joint_home_care"),
    ("허리가 쑤시는데 혼자 할 수 있는 게 있나요?", "joint_home_care"),
])
def test_new_care_rules_fire(message, rule) -> None:
    out = _names(message)
    assert rule in out
    text = out[rule]
    assert re.search(r"일주일에 \d번|밤에|한 번에 \d+분", text)  # 행동 + 횟수·시점


@pytest.mark.parametrize("message", [
    "다리가 저려요",                       # 증상 호소(되묻기 대상)
    "어제 넘어졌는데 무릎이 아파요",         # 이미 다침 - 집 정리 질문이 아니다
    "가슴이 아픈데 집에서 뭘 해요?",         # 가슴 - 관절 냉찜질 줄 금지
    "머리가 아픈데 어떻게 해요?",
    "계단 오르는 운동이 무릎에 좋아요?",
])
def test_new_care_rules_do_not_fire_elsewhere(message) -> None:
    assert not set(_names(message)) & {"leg_strength_exercise", "fall_home_safety", "joint_home_care"}


def test_followup_boundary_kept_for_short_complaints() -> None:
    # 짧은 호소는 여전히 되묻는다 - 새 관리 규칙이 되묻기를 막지 않는다.
    assert M.needs_followup_question("무릎이 아파요") and M.needs_followup_question("다리가 저려요")


def test_care_question_accepts_typo_form() -> None:
    assert M.is_care_question("무릎이 시큰한데 집에서 할 수 잇는 건요?", "medical_advice", False)


def test_care_rules_skip_when_answer_already_has_action_and_timing() -> None:
    reply = "의자를 잡고 발뒤꿈치 들기를 일주일에 3번 해 보세요."
    assert "leg_strength_exercise" not in _names("다리 힘을 키우려면 어떤 운동을 해요?", reply)


def test_care_rules_have_proxy_versions_without_119() -> None:
    for r in mls._LIFESTYLE_RULES:
        if r.name in {"leg_strength_exercise", "fall_home_safety", "joint_home_care"}:
            assert r.sentence.proxy and "119" not in str(r.sentence) and "119" not in r.sentence.proxy
    assert "치워 드리고" in _names("넘어지지 않게 방을 어떻게 정리해 드려요?", proxy=True)["fall_home_safety"]


# ---- G1: 감기약 같이 먹기의 위험 이유 -------------------------------------------------------------------------------


def test_cold_medicine_combo_gives_reason() -> None:
    out = _names("감기약이랑 소화제를 같이 먹어도 돼요?")
    assert "겹치면" in out["cold_medicine_combo"] and "많이" in out["cold_medicine_combo"]
    assert "cold_medicine_bp" not in out


@pytest.mark.parametrize("message", ["혈압약 먹는데 종합감기약 먹어도 돼요?", "코감기약이랑 고혈압약 같이 먹어도 되나요"])
def test_cold_medicine_with_bp_gives_bp_reason(message) -> None:
    assert "혈압을 올릴 수 있어" in _names(message)["cold_medicine_bp"]


def test_cold_medicine_bp_skipped_when_reason_already_given() -> None:
    reply = "코막힘 성분이 혈압을 올릴 수 있어요."
    assert "cold_medicine_bp" not in _names("감기약이랑 혈압약 같이 먹어도 돼요?", reply)
