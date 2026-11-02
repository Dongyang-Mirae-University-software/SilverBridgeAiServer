"""응급 행동 안내(가슴압박·AED)·과잉 119(알약 "목에 걸려"·짧은 위험 신호 말) 회귀 테스트(2026-11-02).

원인(홀드아웃 v1 진단): ① 응급 키워드 "목에걸려"가 알약 삼키기 질문에 걸려 응급 강제 ② 위험 신호 단어가 있는 짧은 말은
되묻지 않고 119로 감 ③ 반응·호흡 없음 유형의 응급 행동 줄이 없어 가슴압박·AED 안내가 한 번도 안 나감.
"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.config import Settings
from app.services import chat_keywords as kw
from app.services import chat_layout
from app.services.medical_llm_service import DRUG_ALCOHOL_SENTENCES, MedicalLlmService

EMOJI = re.compile(r"[\U0001F000-\U0001FAFF⌀-⏿☀-➿⬀-⯿]️?")
CPR = re.compile(r"가슴압박\(가슴 한가운데를 세게, 빠르게")
AED = re.compile(r"자동심장충격기\(AED\)")


class _Loader:
    def __init__(self, **fields: str) -> None:
        self.fields = {"empathy": "많이 놀라셨겠어요.", "symptoms": "응급 증상", "features": "",
                       "advice": "지금은 병원 예약보다 119 연락이 먼저예요.", "reply": "", **fields}
        self.state = SimpleNamespace(model_name="fake-medgemma")

    def ensure_loaded(self) -> None:
        return None

    def generate_structured_text(self, system_prompt: str, user_prompt: str, prefill: str = "", **kwargs) -> str:  # noqa: ARG002
        return json.dumps(self.fields, ensure_ascii=False)


def _gen(message: str, intent: str, proxy: bool = False, **fields: str) -> dict[str, Any]:
    return MedicalLlmService(Settings(), _Loader(**fields)).generate(message, {}, [], intent=intent, proxy=proxy)


# ---- ① 알약 "목에 걸려" -------------------------------------------------------------------------------------

@pytest.mark.parametrize("message", [
    "알약이 커서 목에 걸려요 쪼개 먹어도 돼요?",
    "캡슐이 자꾸 목에 걸려서 열어서 먹어도 되나",
    "서방정이 목에 걸려요 빻아 먹어도 되나요",
])
def test_pill_stuck_question_is_not_emergency(message: str) -> None:
    assert MedicalLlmService.emergency_signals(message) == []
    assert MedicalLlmService.classify_intent(message) != "emergency_guidance"


@pytest.mark.parametrize("message", [
    "떡이 목에 걸려서 숨을 못 쉬어요",  # 음식 질식 - 다른 키워드도 걸린다
    "할아버지 떡이 목에 걸려요",  # 약 낱말 없음 - 그대로 응급
    "알약이 목에 걸려서 숨이 막혀요",  # 약이어도 숨 이상
    "약이 목에 걸려서 컥컥거려요",
])
def test_real_choking_stays_emergency(message: str) -> None:
    assert MedicalLlmService.emergency_signals(message)


def test_pill_question_answer_has_no_119_or_high() -> None:
    result = _gen("알약이 커서 목에 걸려요 쪼개 먹어도 돼요?", "medical_advice",
                  symptoms="알약 삼킴 어려움", advice="약사에게 물어보세요. 갑자기 심해지면 바로 119에 연락하세요.")
    assert result["riskLevel"] != "high"
    assert not result["emergencyWarning"]
    assert "119" not in result["reply"]  # 약 질문 119 빼기(drop_medication_119)가 이제 적용된다


# ---- ② 짧은 위험 신호 말 → 되묻기 + 119 한 줄 ------------------------------------------------------------------

@pytest.mark.parametrize("message", ["숨이 좀..", "가슴이 좀 답답", "숨차네"])
def test_short_red_flag_message_asks_back_with_119_line(message: str) -> None:
    assert MedicalLlmService.needs_followup_question(message)
    result = MedicalLlmService.followup_question_response(message)
    assert result["reply"].count("119") == 1
    assert "언제부터" in result["reply"]
    assert result["riskLevel"] != "high"


def test_short_message_without_red_flag_keeps_old_closing() -> None:
    result = MedicalLlmService.followup_question_response("배아퍼")
    assert "119" not in result["reply"]
    assert result["meta"]["redFlag119"] is False


@pytest.mark.parametrize("message", [
    "숨이 안 쉬어져요", "숨을 못 쉬겠어요", "가슴이 아파요", "말이 안 나와요", "피가 안 멈춰요", "경련해요",
    "남편이 숨을 안 쉬어요", "엄마가 불러도 대답이 없어요",
])
def test_short_real_emergency_still_emergency_and_not_followup(message: str) -> None:
    assert MedicalLlmService.emergency_signals(message)
    assert not MedicalLlmService.needs_followup_question(message)


@pytest.mark.parametrize("message", [
    "약 먹고 나서 숨쉬기 힘들어요",  # 약 질문 + 호흡곤란 = 응급
    "혈압약 먹었는데 가슴이 아파요",
])
def test_medication_question_with_danger_sign_keeps_119(message: str) -> None:
    assert MedicalLlmService.emergency_signals(message)
    result = _gen(message, "emergency_guidance")
    assert result["riskLevel"] == "high" and "119" in result["reply"] and result["emergencyWarning"]


# ---- ③ 심정지 의심 → 119 첫 줄 + 가슴압박·AED -----------------------------------------------------------------

ARREST = [
    "남편이 갑자기 쓰러졌는데 불러도 대답이 없고 숨을 안쉬어요",
    "의식이 없어요, 불러도 반응이 없어요",
    "아버지가 흔들어도 반응이 없어요",
    "할머니가 숨을 쉬지 않아요",
    "엄마가 쓰러져서 눈을 안 떠요",
]


@pytest.mark.parametrize("message", ARREST)
def test_arrest_answer_has_119_first_then_cpr_and_aed(message: str) -> None:
    assert MedicalLlmService.emergency_signals(message)
    out = chat_layout.apply(dict(_gen(message, "emergency_guidance")), DRUG_ALCOHOL_SENTENCES)
    first = out["reply"].split("\n")[0]
    assert first.startswith("🚨 ") and "119" in first
    assert CPR.search(out["reply"]) and AED.search(out["reply"])
    assert EMOJI.findall(out["reply"]) == ["🚨"]  # 🚨 외 이모지 0


@pytest.mark.parametrize("message", ARREST)
def test_arrest_drops_conflicting_lie_down_and_sit_up_lines(message: str) -> None:
    names = [n for n, _ in MedicalLlmService.emergency_action_sentences(message, True)]
    assert "cardiac_arrest" in names
    assert "collapse_fall" not in names and "breathing" not in names


@pytest.mark.parametrize("message", [
    "숨이 안 쉬어지고 입술이 파래요",  # 본인 호흡곤란 - 앉은 자세
    "숨을 못 쉬겠어요",
    "할머니가 쓰러지셨어요",  # 반응 여부 모름 - 기존 쓰러짐 줄
    "가슴이 조이고 식은땀이 나요",
])
def test_conscious_emergencies_do_not_get_cpr(message: str) -> None:
    names = [n for n, _ in MedicalLlmService.emergency_action_sentences(message)]
    assert "cardiac_arrest" not in names


@pytest.mark.parametrize("message", [
    "집사람이 내가 자다가 숨을 멈췄다가 컥 하고 쉰대",  # 수면 무호흡 - 응급 아님
    "약병 흔들어도 돼요?",
    "아침에 눈이 잘 안 떠져요",
])
def test_arrest_words_do_not_overmatch(message: str) -> None:
    assert not kw.ARREST_SIGNAL.search(kw.compact(message))
    assert MedicalLlmService.emergency_signals(message) == []


def test_arrest_sentence_is_in_fixed_sentence_table_once() -> None:
    from app.services.medical_llm_service import FIXED_SENTENCES

    hits = [s for s in FIXED_SENTENCES if CPR.search(str(s))]
    assert len(hits) == 1 and hits[0].name == "_EMERGENCY_ACTIONS.cardiac_arrest"
