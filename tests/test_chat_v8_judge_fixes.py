"""시험지 v8 holdout-A 판정(2026-11-21) 미달 수정 - 우울 질문 돕는 줄·정신건강의학과, 약 하루 최대 횟수·간격, 생성 지시."""
from __future__ import annotations

import pytest

from app.prompts.medical_chat_prompts import SYSTEM_PROMPT, _CARE_QUESTION_LINE
from app.services import medical_llm_service as mls
from app.services.chat_keywords import strip_layout_emoji as _plain
from app.services.medical_llm_service import MedicalLlmService as M
from tests.test_chat_router_followup import _send, env  # noqa: F401 - env 는 픽스처

HELP_SELF = "우울증 검사는 정신건강의학과나 가까운 보건소의 정신건강복지센터에서 받을 수 있어요."
HELP_FAMILY = "가족분 이야기를 판단하지 말고 들어 주시고, 정신건강의학과나 정신건강복지센터 상담을 함께 받아 보시도록 도와 주세요."
LIMIT_PAIN = "약 봉투나 설명서에 적힌 하루 최대 횟수와 복용 간격을 넘기지 마세요. 넘겨 드시면 약에 따라 간이나 위, 콩팥에 무리가 갈 수 있어요."


# ---- B: 위기가 아닌 우울 질문 ----------------------------------------------------------------------------------


def test_depression_help_line_self_and_family() -> None:
    assert M.depression_help_line("우울증 검사는 어디서 받나요?", "답") == HELP_SELF
    assert M.depression_help_line("어머니가 우울해 보이는데 어떻게 도와드려요?", "답") == HELP_FAMILY
    assert M.depression_help_line("우울증 검사는 어디서 받나요?", "답", proxy=True) == HELP_FAMILY


@pytest.mark.parametrize("message", ["우울해서 죽고 싶어요 어떻게 해요", "허리가 아파 죽겠어요", "머리가 아파요"])
def test_depression_help_line_not_for_crisis_idiom_or_unrelated(message) -> None:
    assert M.depression_help_line(message, "답") is None


def test_depression_help_line_not_duplicated() -> None:
    assert M.depression_help_line("우울증 검사는 어디서 받나요?", "가까운 정신건강복지센터에 가 보세요.") is None


def test_depression_question_department_is_psychiatry() -> None:
    assert M.answer_department("요즘 우울해서 잠을 못 자는데 어떻게 해요?") == "정신건강의학과"
    assert M.answer_department("머리가 아파요") == mls._symptom_department("머리가 아파요")  # 다른 질문은 그대로


def test_empty_advice_template_uses_psychiatry_for_depression() -> None:
    out = M._normalize_payload({"symptoms": "", "advice": ""}, "", "우울해서 머리가 무거운데 어디로 가야 하나요?", "medical_advice", False)
    assert out["advice"].startswith("정신건강의학과")


def test_chat_flow_help_line_before_109(env) -> None:
    plain = _plain(_send(env, "우울증 검사는 어디서 받을 수 있나요?")["reply"])
    assert HELP_SELF in plain and plain.index(HELP_SELF) < plain.index("109")
    assert plain.count(HELP_SELF) == 1


def test_new_fixed_sentences_registered() -> None:
    assert mls._DEPRESSION_HELP in mls.FIXED_SENTENCES and mls._DEPRESSION_HELP.proxy == HELP_FAMILY
    names = {r.name for r in mls._DRUG_DIRECT_RULES}
    assert {"painkiller_daily_limit", "daily_limit"} <= names


# ---- C: 약 하루 최대 횟수·간격 ------------------------------------------------------------------------------------


@pytest.mark.parametrize("message", ["진통제를 하루에 몇 번까지 먹어도 돼요?", "타이레놀은 몇 시간마다 먹어요?", "해열제 하루 최대 몇 알이에요?"])
def test_painkiller_daily_limit_fires(message) -> None:
    out = dict(M.drug_direct_sentences(message, "모델 답"))
    assert out.get("painkiller_daily_limit") == LIMIT_PAIN and "daily_limit" not in out
    assert "mg" not in LIMIT_PAIN and "시간마다" not in LIMIT_PAIN  # 수치는 넣지 않는다


def test_other_medicine_daily_limit_fires_without_organ_reason() -> None:
    out = dict(M.drug_direct_sentences("변비약은 하루에 몇 번까지 먹어도 돼요?", "모델 답"))
    assert "daily_limit" in out and "콩팥" not in out["daily_limit"]


@pytest.mark.parametrize("message", ["진통제 먹고 나서 속이 쓰려요", "약은 식후에 먹나요?", "혈압약 먹는데 술 마셔도 돼요?"])
def test_daily_limit_does_not_fire_without_count_question(message) -> None:
    names = {n for n, _ in M.drug_direct_sentences(message, "모델 답")}
    assert not names & {"painkiller_daily_limit", "daily_limit"}


def test_daily_limit_skipped_when_answer_already_says_it() -> None:
    names = {n for n, _ in M.drug_direct_sentences("진통제를 하루에 몇 번까지 먹어도 돼요?", "설명서에 적힌 하루 최대 횟수를 꼭 지키세요.")}
    assert "painkiller_daily_limit" not in names


# ---- D: 생성 지시(최소 보강) ---------------------------------------------------------------------------------------


def test_prompt_asks_for_concrete_visit_criterion_and_action_timing() -> None:
    rule8 = next(line for line in SYSTEM_PROMPT.splitlines() if line.startswith("8)"))
    assert "기간 기준" in rule8 and "계속되거나 심해지면 꼭 진료" not in rule8
    assert "횟수나 시점 하나" in _CARE_QUESTION_LINE
    assert "5) 말투는 존댓말, 짧고 따뜻하게. empathy·advice는 두 문장까지" in SYSTEM_PROMPT  # 길이 상한 그대로
