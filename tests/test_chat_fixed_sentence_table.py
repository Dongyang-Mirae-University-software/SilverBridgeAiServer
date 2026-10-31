"""고정 문장 표(2026-10-31 구조 정리 B, 점검 L-3·L-6): 답에 붙이는 고정 문장은 모두 표(_SENTENCE_TABLE)에 있고,
본인에게 하는 지시(subject=self)는 대리 상담 문장이 있으며, 대리 상담(wardId·가족 주어)일 때만 그 문장을 고른다.
본인 상담 답의 문장은 표로 옮기기 전과 같다."""
from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any

import pytest

import app.services.medical_llm_service as mls
from app.core.config import Settings
from app.schemas.chat_schema import ChatRequest
from app.services.chat_service import ChatService
from app.services.medical_llm_service import FixedSentence, MedicalLlmService

_RULE_TABLES = ("_DRUG_ALCOHOL_RULES", "_DRUG_DIRECT_RULES", "_LIFESTYLE_RULES", "_SYMPTOM_RULES", "_EMERGENCY_ACTIONS")


class _Loader:
    def __init__(self, **fields: str) -> None:
        self.fields = {"empathy": "걱정되셨겠어요.", "symptoms": "증상", "features": "", "advice": "", "reply": "", **fields}
        self.state = SimpleNamespace(model_name="fake-medgemma")

    def ensure_loaded(self) -> None:
        return None

    def generate_structured_text(self, system_prompt: str, user_prompt: str, prefill: str = "", **kwargs) -> str:  # noqa: ARG002
        return json.dumps(self.fields, ensure_ascii=False)


def _gen(message: str, intent: str = "medical_advice", proxy: bool = False, **fields: str) -> dict[str, Any]:
    return MedicalLlmService(Settings(), _Loader(**fields)).generate(message, {}, [], intent=intent, proxy=proxy)


def _sentence(name: str) -> FixedSentence:
    return next(s for s in mls.FIXED_SENTENCES if s.name == name)


# ---- 표 ---------------------------------------------------------------------------------------------------------


def test_every_rule_sentence_is_in_the_table() -> None:
    names = [s.name for s in mls.FIXED_SENTENCES]
    assert len(names) == len(set(names)) == len(mls._SENTENCE_TABLE)
    for table in _RULE_TABLES:
        for rule in getattr(mls, table):
            assert isinstance(rule.sentence, FixedSentence), f"{table}.{rule.name}"
            assert rule.sentence.name == f"{table}.{rule.name}"


def test_no_sentence_constant_outside_the_table() -> None:
    # 고정 문장처럼 보이는 모듈 상수(존댓말로 끝나는 한글 문장)는 표의 문장과 같은 글이어야 한다.
    texts = {str(s) for s in mls.FIXED_SENTENCES}
    for name, value in vars(mls).items():
        if not re.fullmatch(r"_[A-Z0-9_]+", name) or not isinstance(value, str) or isinstance(value, FixedSentence):
            continue
        if re.search(r"[가-힣]", value) and re.search(r"(요|세요|니다)\.$", value):
            assert value in texts, name


def test_subjects_and_proxy_sentences() -> None:
    for sentence in mls.FIXED_SENTENCES:
        assert sentence.subject in {"self", "carer", "neutral", "own_variant"}, sentence.name
        if sentence.subject == "self":
            assert sentence.proxy and sentence.proxy != str(sentence), sentence.name
        else:
            assert sentence.proxy is None, sentence.name


def test_pick_keeps_own_text_for_self_consult() -> None:
    for sentence in mls.FIXED_SENTENCES:
        assert sentence.pick(False) == str(sentence)
        assert type(sentence.pick(False)) is str
        assert sentence.pick(True) == (sentence.proxy or str(sentence))


def test_proxy_sentences_keep_the_119_rule() -> None:
    # 119 는 응급 경고 박스에만 - 대리 문장이 새로 119 를 들이지 않는다.
    for sentence in mls.FIXED_SENTENCES:
        if sentence.proxy:
            assert ("119" in sentence.proxy) == ("119" in sentence), sentence.name


# ---- 경로별 선택 ---------------------------------------------------------------------------------------------------


def test_drug_alcohol_answer_uses_proxy_sentence_only_for_proxy() -> None:
    message = "혈압약 먹는데 술 마셔도 돼요?"
    rule = _sentence("_DRUG_ALCOHOL_RULES.blood_pressure")
    own = _gen(message, advice="의사와 상의해 보세요.")["reply"]
    assert str(rule) in own and rule.proxy not in own
    proxy = _gen(message, proxy=True, advice="의사와 상의해 보세요.")["reply"]
    assert rule.proxy in proxy and str(rule) not in proxy


def test_drug_direct_and_lifestyle_use_proxy_sentence() -> None:
    names = dict(MedicalLlmService.drug_direct_sentences("진통제를 더 먹어도 되나요?", "", proxy=True))
    assert names["painkiller_limit"] == _sentence("_DRUG_DIRECT_RULES.painkiller_limit").proxy
    assert dict(MedicalLlmService.drug_direct_sentences("진통제를 더 먹어도 되나요?", ""))["painkiller_limit"] == \
        str(_sentence("_DRUG_DIRECT_RULES.painkiller_limit"))


def test_drug_direct_skips_when_proxy_sentence_already_in_reply() -> None:
    sentence = _sentence("_DRUG_DIRECT_RULES.painkiller_limit")
    assert "painkiller_limit" not in dict(MedicalLlmService.drug_direct_sentences("진통제를 더 먹어도 되나요?", sentence.proxy, proxy=True))


def test_emergency_warning_box_and_actions_for_proxy() -> None:
    message = "아버지가 갑자기 쓰러지셨어요"
    own = _gen(message, intent="emergency_guidance")
    proxy = _gen(message, intent="emergency_guidance", proxy=True)
    assert own["emergencyWarning"] == [str(mls._EMERGENCY_REPLY)]
    assert proxy["emergencyWarning"] == [mls._EMERGENCY_REPLY.proxy]
    assert "119" in proxy["emergencyWarning"][0] and proxy["riskLevel"] == "high"
    assert "혼자 계시면" not in proxy["emergencyWarning"][0]
    # 돌보는 사람 말투 행동(쓰러짐)은 두 경우 모두 같은 문장
    collapse = str(_sentence("_EMERGENCY_ACTIONS.collapse_fall"))
    assert collapse in own["reply"] and collapse in proxy["reply"]


def test_hypoglycemia_and_no_drive_for_proxy() -> None:
    names = dict(MedicalLlmService.emergency_action_sentences("어머니 혈당이 50이에요", proxy=True))
    assert names["hypoglycemia"] == mls._HYPOGLYCEMIA_SENTENCE.proxy
    assert names["hypoglycemia_recheck"] == mls._HYPOGLYCEMIA_RECHECK.proxy
    names = dict(MedicalLlmService.emergency_action_sentences("어머니가 숨을 못 쉬세요", proxy=True))
    assert names["breathing_no_drive"] == mls._BREATHING_NO_DRIVE.proxy
    assert dict(MedicalLlmService.emergency_action_sentences("숨을 못 쉬겠어요"))["breathing_no_drive"] == str(mls._BREATHING_NO_DRIVE)


@pytest.mark.parametrize("ward_id, message, expected", [
    ("ward-1", "기침이 나요", True),
    ("", "어머니가 기침을 하세요", True),
    ("", "기침이 나요", False),
    ("  ", "기침이 나요", False),
])
def test_chat_service_proxy_rule(ward_id: str, message: str, expected: bool) -> None:
    payload = ChatRequest(userId="u1", message=message, context={"wardId": ward_id} if ward_id else None)
    assert ChatService._is_proxy(payload, message) is expected


def test_fallback_emergency_warning_for_proxy() -> None:
    proxy = ChatService()._fallback_reply("아버지가 쓰러지셨어요", [], emergency=True, proxy=True)
    own = ChatService()._fallback_reply("쓰러질 것 같아요", [], emergency=True)
    assert proxy["emergencyWarning"] == [mls._EMERGENCY_REPLY.proxy]
    assert own["emergencyWarning"] == [str(mls._EMERGENCY_REPLY)]
