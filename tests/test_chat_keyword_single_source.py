"""챗봇 점검 2026-10-30 수정 고정(2026-10-31): 판정 낱말 단일 출처(H-1·M-1·M-3) / 이어서 물어볼 질문(M-2·22자) /
약 + 술 술 끊기 권고(L-1) / 기본 문장 119(L-2) / 죽은 속성(L-5) / 고정 문장 ↔ 빼기 규칙(L-6)."""
from __future__ import annotations

import json
import pathlib
import re
from types import SimpleNamespace
from typing import Any

import pytest

import app.services.medical_llm_service as mls
from app.core.config import Settings
from app.services import chat_keywords, follow_up_questions, reservation_intake, reservation_orchestrator
from app.services.chat_service import ChatService
from app.services.medical_llm_service import MedicalLlmService


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


def _fallback(message: str) -> dict:
    return ChatService()._fallback_reply(message, [])


# ---- 단일 출처 (M-3) ----------------------------------------------------------------------------------------


def test_every_path_uses_the_same_keyword_objects() -> None:
    assert ChatService.HIGH_RISK_KEYWORDS is MedicalLlmService.HIGH_RISK_KEYWORDS is chat_keywords.RED_FLAG_KEYWORDS
    assert ChatService.EMERGENCY_KEYWORDS is MedicalLlmService.EMERGENCY_KEYWORDS is chat_keywords.EMERGENCY_KEYWORDS
    assert MedicalLlmService.SELF_HARM_KEYWORDS is chat_keywords.SELF_HARM_KEYWORDS
    assert ChatService.RESERVATION_KEYWORDS is MedicalLlmService.RESERVATION_KEYWORDS is chat_keywords.RESERVATION_KEYWORDS
    assert MedicalLlmService.MEDICAL_KEYWORDS is chat_keywords.MEDICAL_KEYWORDS
    assert reservation_orchestrator.ReservationOrchestrator.RESERVATION_KEYWORDS is chat_keywords.RESERVATION_CONTEXT_WORDS
    assert mls._symptom_department is reservation_orchestrator._symptom_department is chat_keywords.symptom_department
    assert mls._SYMPTOM_WORDS is chat_keywords.MEDICATION_SYMPTOM_WORDS
    assert mls._GENERAL_SYMPTOM_WORDS is chat_keywords.GENERAL_SYMPTOM_WORDS
    assert mls._FOLLOWUP_SYMPTOM_NOUN is chat_keywords.FOLLOWUP_SYMPTOM_NOUN
    assert reservation_intake._SYMPTOM_WORDS is chat_keywords.SUMMARY_SYMPTOM_WORDS
    assert not hasattr(ChatService, "MEDIUM_RISK_KEYWORDS")


def test_no_keyword_list_literal_outside_the_module() -> None:
    # 판정 목록을 다른 파일에 다시 적으면 다시 갈라진다(H-1). 예약 흐름 안의 조회·검색 낱말은 흐름 내부 값이라 제외.
    allowed = {"LOOKUP_KEYWORDS", "SEARCH_KEYWORDS"}
    services = pathlib.Path(chat_keywords.__file__).parent
    for path in services.glob("*.py"):
        if path.name == "chat_keywords.py":
            continue
        for name in re.findall(r"^\s*(_?[A-Z_]*(?:KEYWORDS|SYMPTOM_WORDS))\s*(?::[^=]*)?=\s*\(", path.read_text(), re.M):
            assert name in allowed, f"{path.name}: {name}"


def test_chat_service_classify_is_the_generation_classifier() -> None:
    for message in ("숨차요", "답답해요", "복통이 있어요", "병원 찾고 싶어요", "오늘 날씨가 좋네요", "가슴이 아파요"):
        assert ChatService.classify_intent(message) == MedicalLlmService.classify_intent(message), message


# ---- H-1 생성 실패 대체 응답 ----------------------------------------------------------------------------------


@pytest.mark.parametrize("message", ["식은땀이 나요", "피가 나요", "얼굴이 파래졌어요", "호흡이 가빠요", "숨차요", "숨 차요"])
def test_fallback_red_flag_is_high_with_119(message: str) -> None:
    result = _fallback(message)
    assert result["riskLevel"] == "high"
    assert "119" in result["reply"]
    assert "안정 상태" not in result["reply"]


@pytest.mark.parametrize("message", ["답답해요", "소변이 자주 마려워요", "어깨가 결려요", "복통이 있어요"])
def test_fallback_symptom_word_is_not_low(message: str) -> None:
    result = _fallback(message)
    assert result["riskLevel"] == "medium"
    assert "안정 상태" not in result["reply"]


def test_fallback_low_only_without_symptom_words() -> None:
    assert _fallback("오늘 날씨가 좋네요")["riskLevel"] == "low"


# ---- M-1 숨차요 -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("message", ["숨차요", "숨 차요", "숨이 차요"])
def test_short_breath_is_not_a_followup_question(message: str) -> None:
    assert MedicalLlmService.needs_followup_question(message) is False
    assert MedicalLlmService.needs_red_flag_sentence(message, "x") is True


def test_short_breath_answer_gets_one_119_line_and_no_suggestions() -> None:
    result = _gen("숨 차요", advice="내과 진료를 받아 보세요.")
    assert result["reply"].count("119") == 1
    assert result["meta"]["redFlag119"] is True
    assert follow_up_questions.should_suggest(result, {"route": "symptom"}, "숨 차요", emergency_text=False, self_harm_text=False) is False


def test_short_breath_is_not_an_emergency_keyword() -> None:
    # "숨이 차요" 경계 결정 유지 - 응급(경고 박스·high)으로 올리지 않는다.
    assert MedicalLlmService.emergency_signals("숨차요") == []


# ---- M-2 이어서 물어볼 질문 -----------------------------------------------------------------------------------


def _plain_result(reply: str = "집에서 쉬어 보세요.", **meta: Any) -> dict:
    return {"reply": reply, "riskLevel": "medium", "intent": "medical_advice", "meta": meta}


def _suggest(result: dict, message: str = "무릎이 아파요") -> bool:
    return follow_up_questions.should_suggest(result, {"route": "symptom"}, message, emergency_text=False, self_harm_text=False)


def test_plain_answer_still_gets_suggestions() -> None:
    assert _suggest(_plain_result()) is True


@pytest.mark.parametrize("meta", [{"redFlag119": True}, {"drugAlcohol": True}])
def test_red_flag_and_drug_alcohol_answers_get_none(meta: dict) -> None:
    assert _suggest(_plain_result(**meta)) is False


def test_any_119_in_reply_gets_none() -> None:
    # 응급 다음 턴: 답에 이미 119가 있어 emergencyFollowup 표시가 없어도 같은 결과.
    assert _suggest(_plain_result("어지러우면 바로 119에 연락하세요.")) is False


def test_drug_alcohol_answer_is_marked() -> None:
    result = _gen("혈압약 먹는데 술 마셔도 돼요?", advice="의사와 상의해 보세요.")
    assert result["meta"]["drugAlcohol"] is True
    assert _suggest(result, "혈압약 먹는데 술 마셔도 돼요?") is False


def test_suggestion_limit_is_22_chars() -> None:
    assert follow_up_questions.MAX_CHARS == 22
    ok, too_long = "가" * 22, "가" * 23
    assert follow_up_questions.sanitize([ok, too_long]) == [ok]
    assert "15자 안팎" in follow_up_questions._SYSTEM_PROMPT
    for questions in (follow_up_questions._RULE_SYMPTOM_SELF, follow_up_questions._RULE_SYMPTOM_PROXY,
                      follow_up_questions._RULE_GENERAL_SELF, follow_up_questions._RULE_GENERAL_PROXY):
        assert follow_up_questions.sanitize(list(questions)) == list(questions)


# ---- L-1 술 끊기 권고 -----------------------------------------------------------------------------------------


def test_quit_alcohol_advice_is_kept() -> None:
    reply = "약 드시는 동안은 술을 끊으세요."
    out, dropped, _ = MedicalLlmService.apply_drug_alcohol_rules("혈압약 먹는데 술 마셔도 돼요?", reply)
    assert dropped == {}
    assert reply in out
    assert MedicalLlmService.needs_self_adjust_sentence(reply, "혈압약 먹는데 술 마셔도 돼요?") is False


def test_skipping_medicine_is_still_dropped() -> None:
    _, dropped, _ = MedicalLlmService.apply_drug_alcohol_rules("혈압약 먹는데 술 마셔도 돼요?", "술을 줄이고 약을 끊으세요.")
    assert dropped.get("self_adjust") == 1


# ---- L-2 기본 문장 ---------------------------------------------------------------------------------------------


def test_empty_advice_default_has_no_119() -> None:
    assert "119" not in MedicalLlmService.format_answer({"symptoms": "무릎 통증", "advice": ""})
    payload = MedicalLlmService._normalize_payload({"symptoms": "어지럼", "advice": ""}, "", "어지러워요", "medical_advice", False)
    assert payload["advice"] and "119" not in payload["advice"]


def test_emergency_advice_keeps_119() -> None:
    assert "119" in MedicalLlmService.format_answer({"symptoms": "가슴 통증", "advice": ""}, emergency=True)


# ---- L-5 ------------------------------------------------------------------------------------------------------


def test_shared_reservation_header_property_is_gone() -> None:
    # 공용 키로 Authorization 을 만들던 손잡이 - 되살리지 않는다(policy ⑧).
    assert not hasattr(Settings, "reservation_headers")


# ---- L-6 고정 문장 ↔ 빼기 규칙 --------------------------------------------------------------------------------


def _fixed_texts() -> dict[str, str]:
    texts: dict[str, str] = {}
    for name, value in vars(mls).items():
        if re.fullmatch(r"_[A-Z0-9_]+", name) and isinstance(value, str) and re.search(r"[가-힣]", value):
            texts[name] = value.format(department="내과") if "{department}" in value else value
    for name in ("_FOLLOWUP_QUESTIONS", "_FOLLOWUP_QUESTIONS_PROXY"):
        for i, value in enumerate(getattr(mls, name)):
            texts[f"{name}[{i}]"] = value
    for table in ("_DRUG_ALCOHOL_RULES", "_DRUG_DIRECT_RULES", "_LIFESTYLE_RULES", "_SYMPTOM_RULES", "_EMERGENCY_ACTIONS"):
        for rule in getattr(mls, table):
            texts[f"{table}.{rule.name}"] = rule.sentence
    return texts


def _drop_hits(sentence: str) -> set[str]:
    hits = set()
    if mls._PROMPT_LEAK.search(sentence):
        hits.add("prompt_leak")
    if mls._MEDICATION_BOILERPLATE.search(sentence):
        hits.add("boilerplate")
    if mls._EXAMPLE_COPY.search(sentence):
        hits.add("example_copy")
    if mls._self_adjust_hit(sentence) and sentence.strip() != mls._SELF_ADJUST_SENTENCE:
        hits.add("self_adjust")
    if (mls._ALCOHOL_WORDS.search(sentence) or "잔" in sentence) and mls._ALCOHOL_ALLOWANCE.search(sentence) \
            and not mls._ALLOWANCE_NEGATED.search(sentence):
        hits.add("allowance")
    for rule in mls._DRUG_ALCOHOL_RULES:
        if rule.wrong and mls._ALCOHOL_WORDS.search(sentence) and rule.wrong.search(sentence) and not rule.covered.search(sentence):
            hits.add(f"wrong:{rule.name}")
    if any(mls._is_unknown_department(m) for m in mls._MEDICINE_DEPT.findall(sentence)):
        hits.add("unknown_department")
    if mls._SLEEPING_PILL_WRONG.search(sentence) or mls._SLEEPING_PILL_HEDGE.search(sentence):
        hits.add("sleeping_pill")
    if "119" in sentence:
        hits.add("119")
    return hits


# 알려진 겹침(점검 ①): 붙이는 단계가 판정·빼기 단계보다 뒤이거나, 그 빼기가 닿지 않는 경로의 문장이다.
_ALLOWED = {
    "_DRUG_DIRECT_RULES.bp_med_stop": {"self_adjust"},  # G10 이 G6·G8 뒤에 붙인다
}
# 119 는 응급·위기·위험 신호 문장에만 - 약 문의 119 빼기(G13)는 응급 답·위기 응답에 닿지 않고, 위험 신호 줄은 G13 뒤에 붙는다.
_ALLOWED_119 = re.compile(r"^(_EMERGENCY_REPLY|_SELF_HARM_\w+|_RED_FLAG_119_SENTENCE|_EMERGENCY_ACTIONS\.\w+)$")


def test_no_fixed_sentence_is_removed_by_a_drop_rule() -> None:
    texts = _fixed_texts()
    assert len(texts) >= 40
    problems = []
    for name, text in texts.items():
        for sentence in mls._SENTENCE_SPLIT.split(text):
            hits = _drop_hits(sentence) - _ALLOWED.get(name, set())
            if _ALLOWED_119.match(name):
                hits.discard("119")
            if hits:
                problems.append((name, sorted(hits)))
    assert problems == []
