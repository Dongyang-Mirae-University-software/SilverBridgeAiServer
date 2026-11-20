"""위기가 아닌 우울 질문의 109 한 줄(2026-11-20 팀원 검수)과 고친 고정 문장.

붙어야 하는 것: 본인·가족이 우울(우울증)의 도움·검사·치료를 묻는 말 - 평소 상담 답 끝에 109 한 줄(위험도·위기 답 아님).
붙지 않아야 하는 것: 말버릇("아파 죽겠다"), 우울과 무관한 증상, 자해 신호(위기 고정 답이 먼저), 이미 109가 있는 답.
"""
from __future__ import annotations

import pytest

from app.services import chat_keywords
from app.services import medical_llm_service as mls
from app.services.medical_llm_service import MedicalLlmService

SELF_LINE = "마음이 많이 힘드실 때는 자살예방 상담전화 109(24시간)에서도 이야기를 들어 줘요."
FAMILY_LINE = "가족분이 힘들어 보이시면 자살예방 상담전화 109(24시간)에서 가족도 상담받을 수 있어요."


@pytest.mark.parametrize("message", [
    "우울증 검사는 어디서 받을 수 있나요?",
    "요즘 우울한데 어떻게 하면 좋아질까요?",
    "우울증은 치료하면 나아지나요",
    "우울할 때 병원에 가 봐야 하나요?",
])
def test_self_depression_question_gets_self_109_line(message) -> None:
    assert chat_keywords.depression_help_question(message)
    assert MedicalLlmService.depression_109_line(message, "평소 상담 답이에요.") == SELF_LINE


@pytest.mark.parametrize("message", [
    "어머니가 요즘 우울해 보이는데 어떻게 도와드리면 좋을까요?",
    "아버지가 우울증 같은데 어디서 검사받으면 돼요?",
])
def test_family_depression_question_gets_family_line(message) -> None:
    assert MedicalLlmService.depression_109_line(message, "답") == FAMILY_LINE


def test_guardian_proxy_flag_picks_family_line() -> None:
    assert MedicalLlmService.depression_109_line("우울증 검사는 어디서 받나요?", "답", proxy=True) == FAMILY_LINE


@pytest.mark.parametrize("message", [
    "허리가 아파 죽겠어요",          # 말버릇
    "배고파 죽겠네요 뭐 먹으면 좋을까요",
    "무릎이 아픈데 어떻게 해요?",     # 우울과 무관한 증상
    "요즘 우울해요",                  # 질문 꼴이 아닌 하소연(질문이 아니면 붙이지 않는다)
])
def test_no_line_without_depression_question(message) -> None:
    assert MedicalLlmService.depression_109_line(message, "답") is None


@pytest.mark.parametrize("message", ["우울해서 죽고 싶어요 어떻게 해요", "우울증이라 자살하고 싶어요"])
def test_self_harm_signal_stays_on_crisis_path(message) -> None:
    assert MedicalLlmService.self_harm_signals(message)  # 위기 고정 답 경로가 먼저
    assert MedicalLlmService.depression_109_line(message, "답") is None


def test_no_duplicate_when_reply_already_has_109() -> None:
    assert MedicalLlmService.depression_109_line("우울증 검사는 어디서 받나요?", "자살예방 상담전화 109에 전화하세요.") is None


def test_depression_line_is_a_table_fixed_sentence_without_119() -> None:
    s = mls._DEPRESSION_109
    assert s in mls.FIXED_SENTENCES and s.proxy == FAMILY_LINE
    assert "119" not in s and "119" not in s.proxy


# ---- 채팅 흐름: 상담 답 끝에 붙고 위험도·위기 판정은 그대로 ------------------------------------------------------


from app.services.chat_keywords import strip_layout_emoji as _plain  # noqa: E402
from tests.test_chat_router_followup import _send, env  # noqa: E402,F401 - env 는 픽스처


def test_chat_flow_appends_line_after_answer_and_keeps_risk(env) -> None:
    out = _send(env, "우울증 검사는 어디서 받을 수 있나요?")
    plain = _plain(out["reply"])
    assert "모델 답변" in plain and plain.rstrip().endswith(SELF_LINE)  # 질문에 먼저 답하고 끝에 109
    assert out["riskLevel"] != "high" and not out["emergencyWarning"] and env.llm.calls == 1
    assert plain.count("109") == 1


def test_chat_flow_family_question_gets_family_line(env) -> None:
    out = _send(env, "어머니가 요즘 우울해 보이는데 어떻게 도와드리면 좋을까요?")
    assert _plain(out["reply"]).rstrip().endswith(FAMILY_LINE)


@pytest.mark.parametrize("message", ["허리가 아파 죽겠어요", "무릎이 아픈데 어떻게 해요?"])
def test_chat_flow_no_line_for_idiom_or_unrelated(env, message) -> None:
    assert "109" not in _send(env, message)["reply"]


def test_chat_flow_crisis_sentence_gets_fixed_crisis_reply_not_extra_line(env) -> None:
    out = _send(env, "우울해서 죽고 싶어요 어떻게 해요")
    assert out["riskLevel"] == "high" and env.llm.calls == 0  # 위기 고정 답(모델을 부르지 않는다)
    assert SELF_LINE not in _plain(out["reply"]) and FAMILY_LINE not in _plain(out["reply"])


# ---- 고친 고정 문장(2026-11-20 팀원 검수) ---------------------------------------------------------------------------


def _action(name: str):
    return next(a.sentence for a in mls._EMERGENCY_ACTIONS if a.name == name)


def test_reviewed_fixed_sentences() -> None:
    assert _action("vomiting_blood").proxy.endswith("아무것도 먹이거나 마시게 하지 마세요.")
    hypo = mls._HYPOGLYCEMIA_SENTENCE
    assert "먹이지" not in str(hypo) and "119" in str(hypo) and "혼자 해결하려 하지 말고" in str(hypo)
    seizure = _action("seizure")
    assert str(seizure).startswith("경련이 올 것 같으면 바로 바닥에 누우시고")
    assert seizure.proxy == "억지로 붙잡거나 입에 물건을 넣지 말고, 옆으로 눕힌 뒤 주변의 위험한 물건을 치워 주세요."
    for s in (_action("chest_stroke"), mls._BREATHING_NO_DRIVE):
        for text in (str(s), s.proxy):
            assert "운전" in text and "구급차" in text and "바로 처치" in text and "맞는 병원" in text
    diabetes = next(r.sentence for r in mls._DRUG_ALCOHOL_RULES if r.name == "diabetes")
    for text in (str(diabetes), diabetes.proxy):
        assert "몇 시간 뒤나 밤사이에 저혈당" in text and "늦게 올" not in text
