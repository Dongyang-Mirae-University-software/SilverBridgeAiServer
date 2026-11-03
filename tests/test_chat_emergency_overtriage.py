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


# ---- 2026-11-03 기능 점검 반영(H-1·M-1·M-2·L-1·L-3) ---------------------------------------------------------

@pytest.mark.parametrize("message", [
    "알약이 목에 걸려서 침도 못 삼켜요",  # H-1 침을 못 삼킴 - 식도 막힘일 수 있다
    "아버지가 약을 드시다 사탕이 목에 걸려요 침을 흘리세요",  # H-1 약 낱말 + 음식 질식
    "약 먹고 떡이 목에 걸려서 꺽꺽거려요",
    "캡슐이 목에 걸려서 안 넘어가요",
])
def test_pill_exclusion_keeps_real_swallowing_trouble_and_food_choking(message: str) -> None:
    assert "목에걸려" in MedicalLlmService.emergency_signals(message)


@pytest.mark.parametrize("message", [
    "알약이 커서 목에 걸려요 쪼개 먹으면 돼요?",  # "먹으면"의 "면"이 음식으로 잡히지 않게
    "밥 먹고 약 먹는데 알약이 목에 걸려요 가루로 먹어도 되나요",
])
def test_pill_exclusion_still_applies_to_plain_pill_questions(message: str) -> None:
    assert MedicalLlmService.emergency_signals(message) == []


@pytest.mark.parametrize("message", ["숨이 멎을 만큼 놀랐어요", "숨이 멎는 줄 알았어요 너무 예뻐서", "숨이 멎을 것 같이 반가웠어요"])
def test_figurative_breath_stop_is_not_emergency_or_arrest(message: str) -> None:  # M-1
    assert MedicalLlmService.emergency_signals(message) == []
    assert "cardiac_arrest" not in [n for n, _ in MedicalLlmService.emergency_action_sentences(message)]


@pytest.mark.parametrize("message", ["할머니가 갑자기 숨이 멎었어요", "남편 숨이 멎은 것 같아요 불러도 대답이 없어요"])
def test_real_breath_stop_is_still_arrest(message: str) -> None:
    assert MedicalLlmService.emergency_signals(message)
    assert "cardiac_arrest" in [n for n, _ in MedicalLlmService.emergency_action_sentences(message)]


@pytest.mark.parametrize("message", [
    "정신을 잃을 것 같아요",  # M-2 추측 - 의식이 있다
    "아까 정신을 잃었다가 금방 깼어요",  # 회복
    "의식이 없어질 것처럼 어지러워요",
    "엄마가 잠깐 기절했는데 지금은 정신이 돌아왔어요",
])
def test_conscious_speculation_or_recovery_gets_no_cpr_line(message: str) -> None:
    names = [n for n, _ in MedicalLlmService.emergency_action_sentences(message)]
    assert "cardiac_arrest" not in names
    assert MedicalLlmService.emergency_signals(message)  # 응급 판정(119)은 그대로


@pytest.mark.parametrize("message", [
    "엄마가 의식이 없어지셨어요",  # 지금 의식 잃음("없어지셨") - 추측으로 빼지 않는다
    "깨워도 안 깼어요 숨도 안 쉬어요",  # "안 깼어요"는 회복이 아니다
    "정신을 잃고 쓰러졌어요",
    "정신이 안 들어요 숨을 안 쉬어요",
])
def test_current_unresponsiveness_still_gets_cpr_line(message: str) -> None:
    assert "cardiac_arrest" in [n for n, _ in MedicalLlmService.emergency_action_sentences(message)]


# L-1·L-3 라우터 실패(키워드 분류) 경로
@pytest.mark.parametrize("message", ["알약이 커서 목에 걸려요 쪼개 먹어도 돼요?", "서방정 빻아 먹어도 되나요", "캡슐 열어서 먹어도 돼요?"])
def test_router_failure_keeps_pill_question_as_consultation(message: str) -> None:
    assert MedicalLlmService.classify_intent(message) == "medical_advice"


@pytest.mark.parametrize("message", ["남편이 숨을 안 쉬어요", "엄마가 불러도 대답이 없어요", "알약이 목에 걸려서 침도 못 삼켜요"])
def test_router_failure_keeps_arrest_and_swallowing_trouble_as_emergency(message: str) -> None:
    assert MedicalLlmService.classify_intent(message) == "emergency_guidance"


# L-3 되묻기 다음 턴: 앞 질문의 위험 신호가 이어 붙인 문장에 남아 다음 답에도 119 한 줄이 붙는다
def test_followup_next_turn_keeps_red_flag_119() -> None:
    from app.services.chat_service import ChatService

    history = [{"role": "user", "content": "숨이 좀.."}, {"role": "assistant", "content": "몇 가지 여쭤볼게요."}]
    combined = ChatService._followup_answer_text({"route": "symptom", "emergency": False}, "어제부터 그래요", history, True)
    assert combined and combined.startswith("숨이 좀..")
    result = _gen(combined, "medical_advice", symptoms="숨참", advice="내과 진료를 받아 보세요.")
    assert result["reply"].count("119") == 1


def test_followup_next_turn_with_emergency_answer_is_judged_on_this_message() -> None:
    from app.services.chat_service import ChatService

    history = [{"role": "user", "content": "숨이 좀.."}]
    # 이번 문장이 응급이면 라우트가 응급이 되어 이어 붙이지 않는다(판정은 이번 문장 그대로)
    assert ChatService._followup_answer_text({"route": "symptom", "emergency": True}, "숨을 못 쉬겠어요", history, True) is None
    assert MedicalLlmService.emergency_signals("숨을 못 쉬겠어요")


def test_food_word_alone_keeps_choking_emergency() -> None:
    # 거리 낱말(숨·침·꺽꺽) 없이 음식 낱말만으로도 질식으로 본다(H-1)
    assert "목에걸려" in MedicalLlmService.emergency_signals("할머니가 약을 드시다가 떡이 목에 걸려요")


@pytest.mark.parametrize("message", [
    "아버지가 불러도 대답이 없었는데 지금은 깨어나셨어요",
    "남편이 흔들어도 반응이 없더니 방금 깼어요",
])
def test_recovered_person_gets_no_cpr_line(message: str) -> None:
    # 회복 표현만으로 심정지 줄을 뺀다(M-2) - 응급(119)은 그대로
    assert "cardiac_arrest" not in [n for n, _ in MedicalLlmService.emergency_action_sentences(message)]
    assert MedicalLlmService.emergency_signals(message)
