"""답변 품질 ①② 짧은 판(2026-11-05, 홀드아웃 v5 판정) - 생성 지시 고정.

① 증상 상담 advice 둘째 문장에 구체 기준(기간·수치·더해지는 증상). ② 약 질문 advice 는 이유 + 약사·의사에게 확인할 것.
문장 수는 그대로 두 문장(v4 의 세 문장 판은 응답 시간 +26~46% 로 미반영).
"""
from __future__ import annotations

from app.prompts.medical_chat_prompts import SYSTEM_PROMPT


def _rule(n: str) -> str:
    return next(line for line in SYSTEM_PROMPT.splitlines() if line.startswith(f"{n})"))


def _example(prefix: str) -> str:
    return next(line for line in SYSTEM_PROMPT.splitlines() if line.startswith(prefix))


def test_advice_stays_two_sentences() -> None:
    assert "empathy·advice는 두 문장까지" in _rule("5")
    assert "세 문장" not in SYSTEM_PROMPT
    assert "셋째 문장" not in SYSTEM_PROMPT


def test_rule8_second_sentence_has_concrete_criterion() -> None:
    rule8 = _rule("8")
    assert "구체적인 기준 하나를 넣는다" in rule8
    assert "며칠 넘게" in rule8 and "얼마 이상" in rule8
    assert "위험 신호가 함께 있을 때만" in rule8  # 119 규칙 그대로


def test_rule4_reason_and_whom_without_verdict() -> None:
    rule4 = _rule("4")
    assert "단정하지 말고" in rule4
    assert "늘리거나 줄이거나 끊으라고도 하지 마라" in rule4
    assert "왜 조심해야 하는지" in rule4 and "약사나 의사에게" in rule4
    assert "119 문장은 붙이지 않는다" in rule4
    assert "이미 정해진 양보다 더 먹었다고 직접 말한 경우에만" in rule4


def test_rule10_units() -> None:
    assert "일주일 기준을 하루 기준으로 바꾸지 마라" in _rule("10")


def test_examples_two_sentences_no_119_no_verdict() -> None:
    ex1 = _example('{"empathy": "허리가')
    assert "2주 넘게" in ex1 and "다리가 저려 오면" in ex1 and "119" not in ex1
    ex2 = _example('{"empathy": "관절약')
    assert "부작용" in ex2 and "약사" in ex2
    for banned in ("119", "드셔도 돼요", "드셔도 괜찮", "끊으", "줄이", "늘리"):
        assert banned not in ex2
    for ex in (ex1, ex2):
        advice = ex.split('"advice": "')[1].split('"')[0]
        assert advice.count(".") == 2  # 두 문장


# ---- 약 문의 답 119 빼기 넓힘 · 지시 문구 노출 빼기 (2026-11-05, v1 W04) -------------------------------

from app.services.medical_llm_service import MedicalLlmService  # noqa: E402

_LEAKED_119 = "약을 더 드시면 몸이 이상해지면 바로 119에 연락하라고 쓴다."
_PLAIN_119 = "약을 더 드신 뒤 몸에 이상이 생기면 바로 119에 연락해 주세요."


def test_drop_119_for_supplement_and_drug_name_questions() -> None:
    for q in ("비타민D 하루 5알씩 먹으려구요 괜찮을까요", "와파린 먹는데 나물 매일 먹어도 괜찮은가요"):
        reply, removed = MedicalLlmService.drop_medication_119(q, f"정리했어요.\n{_PLAIN_119}")
        assert removed == 1 and "119" not in reply, q


def test_keep_119_when_overdose_told() -> None:
    q = "혈압약을 두 알 더 먹었는데 괜찮을까요"
    reply, removed = MedicalLlmService.drop_medication_119(q, f"정리했어요.\n{_PLAIN_119}")
    assert removed == 0 and "119" in reply


def test_keep_119_for_non_medication_question() -> None:
    reply, removed = MedicalLlmService.drop_medication_119("계단 오르면 숨이 차요", f"정리했어요.\n{_PLAIN_119}")
    assert removed == 0


def test_prompt_instruction_wording_dropped() -> None:
    reply, removed = MedicalLlmService.drop_prompt_leak(f"성분이 겹칠 수 있어요.\n{_LEAKED_119}")
    assert removed == 1 and "라고 쓴다" not in reply
