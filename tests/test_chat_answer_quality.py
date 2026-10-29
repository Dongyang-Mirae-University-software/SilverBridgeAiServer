"""챗봇 답변 품질 후처리(2026-10-29 D): 진료과 키워드 패턴(D-6) / 진료과 힌트 맞춤(D-1) / 질문 되풀이 증상 줄(D-2) /
일반 증상 답 병원 권유 보장(D-4) / 규칙 8 119 단계(D-3). 응급·약 + 술·일반 대화 답에는 새 후처리가 닿지 않아야 한다."""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.config import Settings
from app.prompts.medical_chat_prompts import SYSTEM_PROMPT
from app.services.medical_llm_service import MedicalLlmService
from app.services.reservation_orchestrator import _symptom_department


class _Loader:
    def __init__(self, **fields: str) -> None:
        self.fields = {"empathy": "걱정되셨겠어요.", "symptoms": "증상", "features": "", "advice": "", "reply": "", **fields}
        self.state = SimpleNamespace(model_name="fake-medgemma")

    def ensure_loaded(self) -> None:
        return None

    def generate_structured_text(self, system_prompt: str, user_prompt: str, prefill: str = "", **kwargs) -> str:  # noqa: ARG002
        return json.dumps(self.fields, ensure_ascii=False)


def _gen(message: str, intent: str = "medical_advice", **fields: str) -> dict[str, Any]:
    return MedicalLlmService(Settings(), _Loader(**fields)).generate(message, {}, [], intent=intent)


# ---- D-6 진료과 키워드 ----------------------------------------------------------------------------------


@pytest.mark.parametrize("message", [
    "약속 시간에 늦었어요", "속상해서 잠이 안 와요", "배우자가 걱정이에요", "눈치가 보여서요",
    "코로나 백신 맞아도 되나요", "귀찮아서 약을 안 먹었어요", "병원 가려고 해요", "할아버지가 열심히 운동했어요",
])
def test_single_char_words_do_not_trigger_department(message: str) -> None:
    assert _symptom_department(message) is None


@pytest.mark.parametrize("message, department", [
    ("목요일부터 허리가 아파요", "정형외과"),
    ("머리카락이 많이 빠지고 피부가 가려워요", "피부과"),
    ("허리띠를 매면 배가 아파요", "내과"),
    ("기침이 계속 나요", "내과"),
])
def test_false_positive_word_no_longer_wins(message: str, department: str) -> None:
    assert _symptom_department(message) == department


@pytest.mark.parametrize("message, department", [
    ("며칠째 기침이 나고 목이 아파요", "이비인후과"),
    ("아침에 일어나면 어지러워요", "신경과"),
    ("어지러워", "신경과"),
    ("요즘 일어날 때 어지럽고 눈앞이 캄캄해져.", "신경과"),
    ("머리가 너무 아파서 병원 예약하고 싶어요", "신경과"),
    ("계단 오를 때 무릎이 아파요", "정형외과"),
    ("관절염이 있는데 어떤 운동이 좋아요", "정형외과"),
    ("식사하고 나면 속이 더부룩해요", "내과"),
    ("며칠째 변을 못 봤어요. 배가 더부룩해요.", "내과"),
    ("눈이 침침하고 자주 피곤해요", "안과"),
    ("눈이 충혈됐어요", "안과"),
    ("열이 38도예요", "내과"),
    ("감기약이랑 혈압약을 같이 먹어도 되나요", "내과"),
    ("코가 막히고 콧물이 나요", "이비인후과"),
    ("귀에서 소리가 나요", "이비인후과"),
    ("목이 부었어요", "이비인후과"),
    ("팔이 가려워요", "피부과"),
])
def test_symptom_words_keep_department(message: str, department: str) -> None:
    assert _symptom_department(message) == department


# ---- D-1 진료과 힌트 맞춤 --------------------------------------------------------------------------------


def test_answer_department_aligned_to_hint() -> None:
    out = _gen("아침에 일어나면 어지러워요", symptoms="어지러움",
               advice="정신건강의학과 진료를 권해드려요. 어지러움이 계속되면 꼭 진료를 받아 보세요.")
    assert "신경과 진료를 권해드려요" in out["reply"]
    assert "정신건강의학과" not in out["reply"]


def test_same_family_department_kept() -> None:
    advice = "소화기내과 진료를 권해드려요."
    assert MedicalLlmService.align_department(advice, "내과") == (advice, False)


def test_two_departments_kept() -> None:
    advice = "정형외과나 재활의학과 진료를 받아 보세요."
    assert MedicalLlmService.align_department(advice, "신경과") == (advice, False)


def test_no_hint_keeps_answer_department() -> None:
    out = _gen("밤에 다리에 쥐가 자주 나요", symptoms="다리 쥐", advice="재활의학과 진료를 권해드려요.")
    assert "재활의학과 진료를 권해드려요" in out["reply"]


# ---- D-2 질문 되풀이 ------------------------------------------------------------------------------------


def test_echoed_symptoms_line_hidden() -> None:
    out = _gen("혈압약 먹었어요", symptoms="혈압약 먹었어요", advice="내과 진료 때 여쭤보세요.")
    assert "· 증상:" not in out["reply"]
    assert out["summary"] is None
    assert out["reply"].startswith("걱정되셨겠어요.")


def test_server_filled_symptoms_hidden() -> None:
    # 모델이 증상을 비우면 서버가 질문 앞 60자로 채운다 - 그것도 되풀이다.
    out = _gen("당뇨가 있는데 과일을 먹어도 되나요", symptoms="", advice="병원 진료 때 여쭤보세요.")
    assert "· 증상:" not in out["reply"]
    assert out["summary"] is None


def test_long_question_server_fill_hidden() -> None:
    message = "어제 저녁부터 오른쪽 무릎이 시큰거리고 계단을 오를 때마다 너무 아파서 걷기가 힘들고 밤에는 잠도 잘 안 와요 어떻게 하죠"
    assert MedicalLlmService.is_echo(message.strip()[:60], message)


def test_summarized_symptoms_shown() -> None:
    out = _gen("아침에 일어나면 어지러워요", symptoms="어지러움 (아침에 일어날 때)", advice="신경과 진료를 권해드려요.")
    assert "· 증상: 어지러움 (아침에 일어날 때)" in out["reply"]
    assert out["summary"] == "어지러움 (아침에 일어날 때)"


# ---- D-4 병원 권유 보장 ---------------------------------------------------------------------------------


def test_symptom_answer_without_clinic_gets_department_sentence() -> None:
    out = _gen("요즘 무릎이 시려요", symptoms="무릎 시림", advice="따뜻하게 찜질해 보세요.")
    assert out["reply"].endswith("증상이 계속되거나 걱정되시면 정형외과 진료를 받아 보세요.")


def test_symptom_without_hint_gets_clinic_sentence() -> None:
    out = _gen("밤에 다리에 쥐가 자주 나요", symptoms="다리 쥐", advice="스트레칭을 해 보세요.")
    assert out["reply"].endswith("증상이 계속되거나 걱정되시면 병원 진료를 받아 보세요.")


def test_urination_now_gets_internal_medicine_hint() -> None:
    # 2026-10-29 E: "소변"이 내과 힌트가 되어 답의 다른 과(비뇨의학과 포함)를 내과로 맞춘다(예약 표준 진료과와 같은 과).
    out = _gen("요즘 소변을 자주 봐요", symptoms="잦은 소변", advice="비뇨의학과 진료를 권해드려요.")
    assert "내과 진료를 권해드려요" in out["reply"]


def test_lifestyle_question_gets_consult_sentence() -> None:
    out = _gen("혈압이 높은 날 운동해도 되나요", symptoms="고혈압", advice="가벼운 산책부터 해 보세요.")
    assert out["reply"].endswith("궁금하신 점은 병원 진료 때 의사와 상의해 보세요.")


def test_no_extra_sentence_when_clinic_present() -> None:
    out = _gen("계단 오를 때 무릎이 아파요", symptoms="무릎 통증", advice="정형외과 진료를 권해드려요.")
    assert out["reply"].endswith("정형외과 진료를 권해드려요.")


def test_medication_question_left_to_existing_rule() -> None:
    # 약 복용 문의(증상 없음)는 C-2 규칙대로 의사·약사 안내만 - 새 문장을 붙이지 않는다.
    # 수면제 구입 질문은 2026-10-29 G-1 고정 문장이 붙어 test_chat_sleeping_pill_breath_lifestyle 에서 본다.
    out = _gen("감기약이랑 혈압약을 같이 먹어도 되나요", symptoms="약 복용 문의", advice="의사나 약사에게 물어봐 주세요.")
    assert out["reply"].endswith("의사나 약사에게 물어봐 주세요.")


def test_general_chat_untouched() -> None:
    out = _gen("오늘 날씨가 참 좋네요", intent="general_chat", symptoms="", reply="네, 산책하기 좋은 날이에요.")
    assert out["reply"] == "네, 산책하기 좋은 날이에요."


RED_FLAG = "숨쉬기 힘들거나 가슴이 조이고 식은땀이 나는 등 갑자기 심해지면 바로 119에 연락하세요."


def test_red_flag_question_without_119_gets_sentence() -> None:
    out = _gen("계단 오르면 숨이 차요", symptoms="숨 가쁨", advice="내과 진료를 권해드려요. 숨 가쁨이 계속되면 꼭 진료를 받아 보세요.")
    assert out["reply"].endswith(RED_FLAG)
    assert out["riskLevel"] != "high" and not out["emergencyWarning"]  # 응급 판정은 그대로


def test_red_flag_sentence_skipped_when_119_present() -> None:
    advice = "병원 진료를 받아 보세요. 가슴 통증이 함께 있으면 119에 연락하세요."
    out = _gen("가슴이 답답해요", symptoms="가슴 답답함", advice=advice)
    assert RED_FLAG not in out["reply"]


def test_no_red_flag_sentence_for_ordinary_symptom() -> None:
    out = _gen("계단 오를 때 무릎이 아파요", symptoms="무릎 통증", advice="정형외과 진료를 권해드려요.")
    assert "119" not in out["reply"]


# ---- 응급·약 + 술은 새 후처리 대상이 아니다 ----------------------------------------------------------------


@pytest.mark.parametrize("message, intent", [
    ("갑자기 어지럽고 한쪽 팔에 힘이 안 들어가요", "emergency_guidance"),
    ("어지럽고 가슴이 아파요", "medical_advice"),  # 키워드 응급(가슴이아)
])
def test_emergency_answer_untouched(message: str, intent: str) -> None:
    out = _gen(message, intent=intent, symptoms=message, advice="정신건강의학과 진료를 권해드려요.")
    assert out["emergencyWarning"]
    assert f"· 증상: {message}" in out["reply"]  # 되풀이여도 증상 줄을 빼지 않는다
    # 119 안내 줄은 그대로다(2026-10-29 F-2부터 그 뒤에 응급 유형별 행동 한 줄이 붙을 수 있다).
    assert "지금은 병원 예약보다 119 연락이 먼저예요. 바로 119에 전화해 주세요." in out["reply"].split("\n")
    assert "신경과" not in out["reply"]
    assert out["riskLevel"] == "high"


def test_drug_alcohol_answer_untouched_by_quality_rules() -> None:
    message = "고혈압약 먹는데 머리가 아파서 술 마셔도 되나요"  # 힌트 신경과
    out = _gen(message, symptoms=message,
               advice="정신건강의학과 상담을 받아 보세요.")
    assert "· 증상:" in out["reply"]
    assert "정신건강의학과" in out["reply"]
    assert "궁금하신 점은" not in out["reply"]
    assert "혈압약을 드시는 중에 술을 마시면" in out["reply"]  # 기존 약 + 술 규칙은 그대로


def test_quality_log_has_flags_only(caplog) -> None:
    caplog.set_level(logging.INFO)
    _gen("아침에 일어나면 어지러워요", symptoms="아침에 일어나면 어지러워요", advice="정신건강의학과 진료를 권해드려요.")
    lines = [r.getMessage() for r in caplog.records if "[CHAT-QUALITY]" in r.getMessage()]
    assert lines == ["[CHAT-QUALITY] echo_hidden=True department_aligned=True clinic_added=False red_flag_119=False pediatrics=False medication_119_dropped=0"]


# ---- D-3 규칙 8 ----------------------------------------------------------------------------------------


def test_rule8_no_longer_requires_119_for_general_symptoms() -> None:
    rule8 = next(line for line in SYSTEM_PROMPT.splitlines() if line.startswith("8)"))
    assert "계속되거나 심해지면" in rule8
    assert "위험 신호가 함께 있을 때만" in rule8
    example1 = next(line for line in SYSTEM_PROMPT.splitlines() if line.startswith('{"empathy": "허리가'))
    assert "119" not in example1
