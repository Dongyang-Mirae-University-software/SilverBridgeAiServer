"""수면제 구입 안내(G-1)·숨참/토혈 응급 행동(G-2)·생활관리 고정 문장(G-3) (2026-10-29)."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.config import Settings
from app.services.medical_llm_service import MedicalLlmService

PLAIN = "약을 더 드시거나 바꾸기 전에 꼭 의사나 약사에게 물어봐 주세요."
SLEEP = "약국에서 살 수 있는 수면유도제(일반의약품)도 있지만, 수면제는 처방이 필요해요. 드시기 전에 약사와 상의하세요."
CALL_119 = "지금은 병원 예약보다 119 연락이 먼저예요. 바로 119에 전화해 주세요."
BREATH = "편하게 앉은 자세로 숨을 쉬시고, 꽉 끼는 옷은 풀어 두세요."  # 본인 상담 문장(2026-10-31 - 돌보는 사람 말투는 대리 상담 문장)
BLOOD = "토한 것이 목에 걸리지 않게 옆으로 누우시고, 아무것도 먹거나 마시지 마세요."  # 본인 상담 문장(2026-10-31 - 돌보는 사람 말투는 대리 상담 문장)
FRUIT = "과일은 당도가 낮은 것을 조금씩, 식후에 드시는 것이 좋아요."
BP_EX = "혈압이 평소보다 높게 나온 날은 운동을 쉬거나 가볍게 하시고, 가슴 통증·어지럼이 있으면 바로 멈추세요."
JOINT = "걷기·수영·실내 자전거처럼 관절에 무리가 덜 가는 운동을, 통증이 생기지 않는 범위에서 해 보세요."


class _Loader:
    def __init__(self, **fields: str) -> None:
        self.fields = {"empathy": "걱정되셨겠어요.", "symptoms": "문의", "features": "", "advice": PLAIN, "reply": "", **fields}
        self.state = SimpleNamespace(model_name="fake-medgemma")

    def ensure_loaded(self) -> None:
        return None

    def generate_structured_text(self, system_prompt: str, user_prompt: str, prefill: str = "", **kwargs) -> str:  # noqa: ARG002
        return json.dumps(self.fields, ensure_ascii=False)


def _gen(message: str, intent: str = "medical_advice", **fields: str) -> dict[str, Any]:
    return MedicalLlmService(Settings(), _Loader(**fields)).generate(message, {}, [], intent=intent)


# ---- G-1 수면제 구입 ------------------------------------------------------------------------------------


@pytest.mark.parametrize("message", ["수면제를 약국에서 그냥 살 수 있나요", "잠 오는 약 약국에서 사도 돼요?", "수면유도제 구입하려면 처방 필요해요?"])
def test_sleeping_pill_purchase_gets_sentence(message: str) -> None:
    assert _gen(message)["reply"].count(SLEEP) == 1


def test_sleeping_pill_misstatement_dropped() -> None:
    out = _gen("수면제를 약국에서 그냥 살 수 있나요", advice="수면제는 처방전 없이 약국에서 살 수 없어요. 의사나 약사에게 물어봐 주세요.")
    assert "살 수 없어요" not in out["reply"]
    assert "의사나 약사에게 물어봐 주세요." in out["reply"] and SLEEP in out["reply"]


def test_misstatement_kept_for_other_questions() -> None:
    advice = "처방약은 약국에서 그냥 살 수 없어요. 의사와 상의하세요."
    out = _gen("혈압약이 떨어졌는데 어떻게 해요?", advice=advice)
    assert advice in out["reply"]


def test_sleeping_pill_alcohol_left_to_drug_alcohol_rules() -> None:
    out = _gen("수면제 사서 먹고 술 마셔도 돼?")
    assert SLEEP not in out["reply"]
    assert "수면제나 안정제를 술과 함께 드시면" in out["reply"]


@pytest.mark.parametrize("message", ["수면제 먹고 사과 먹어도 돼요?", "잠이 안 와요"])
def test_non_purchase_questions_untouched(message: str) -> None:
    assert SLEEP not in _gen(message)["reply"]


# ---- G-2 숨참·토혈 응급 행동 ------------------------------------------------------------------------------


@pytest.mark.parametrize("message, intent, action", [
    ("숨이 안 쉬어지고 입술이 파래요", "medical_advice", BREATH),
    ("숨을 못 쉬겠어요", "medical_advice", BREATH),
    ("숨쉬기 힘들어", "emergency_guidance", BREATH),
    ("피를 토했어요", "medical_advice", BLOOD),
    ("토한 것이 커피색이에요", "emergency_guidance", BLOOD),
])
def test_breath_and_blood_actions_after_119(message: str, intent: str, action: str) -> None:
    out = _gen(message, intent=intent, symptoms="응급", advice="x")
    lines = out["reply"].split("\n")
    assert CALL_119 in lines and action in lines and lines.index(CALL_119) < lines.index(action)
    assert out["emergencyWarning"] == [MedicalLlmService.EMERGENCY_WARNING]


def test_choking_gets_no_sitting_advice() -> None:
    out = _gen("목에 떡이 걸려서 질식할 것 같아요", intent="emergency_guidance", symptoms="응급", advice="x")
    assert CALL_119 in out["reply"] and BREATH not in out["reply"]


def test_breath_action_not_on_non_emergency() -> None:
    out = _gen("계단 오르면 숨이 차요", symptoms="숨 가쁨", advice="내과 진료를 권해드려요.")
    assert not out["emergencyWarning"] and BREATH not in out["reply"]


def test_self_harm_fixed_reply_untouched() -> None:
    reply = MedicalLlmService.self_harm_response()["reply"]
    assert BREATH not in reply and BLOOD not in reply


# ---- G-3 생활관리 ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("message, sentence", [
    ("당뇨가 있는데 과일을 먹어도 되나요", FRUIT),
    ("혈당 높은데 과일 주스 마셔도 돼요?", FRUIT),
    ("혈압이 높은 날 운동해도 되나요", BP_EX),
    ("운동하려는데 혈압이 좀 높게 나와요", BP_EX),
    ("관절염이 있는데 어떤 운동이 좋아요", JOINT),
])
def test_lifestyle_sentence_once(message: str, sentence: str) -> None:
    assert _gen(message, advice="의사와 상의해 보세요.")["reply"].count(sentence) == 1


def test_lifestyle_sentence_skipped_when_covered() -> None:
    out = _gen("관절염이 있는데 어떤 운동이 좋아요", advice="수영처럼 통증이 덜한 운동부터 해 보세요.")
    assert JOINT not in out["reply"]
    out = _gen("당뇨가 있는데 과일을 먹어도 되나요", advice="당도가 낮은 과일을 조금씩 드세요.")
    assert FRUIT not in out["reply"]


@pytest.mark.parametrize("message", ["과일 깎는 법 알려 주세요", "운동 끝나고 무릎이 아파요", "혈압이 높아요"])
def test_lifestyle_rules_not_for_other_questions(message: str) -> None:
    reply = _gen(message, advice="진료를 권해드려요.")["reply"]
    assert not any(s in reply for s in (FRUIT, BP_EX, JOINT))


def test_g_log_rule_names_only(caplog) -> None:
    import logging
    caplog.set_level(logging.INFO)
    _gen("수면제를 약국에서 그냥 살 수 있나요", advice="수면제는 약국에서 살 수 없어요.")
    _gen("피를 토했어요", symptoms="응급", advice="x")
    msgs = [r.getMessage() for r in caplog.records if "[CHAT-SAFETY]" in r.getMessage()]
    assert "[CHAT-SAFETY] sleeping_pill_misstatement_dropped=1" in msgs
    assert "[CHAT-SAFETY] drug_direct=sleeping_pill_purchase" in msgs
    assert "[CHAT-SAFETY] emergency_action=vomiting_blood,vomiting_blood_info" in msgs
