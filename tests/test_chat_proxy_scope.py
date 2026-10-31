"""대리 상담 판정 범위·응급 쪽 돌보는 사람 말투 기본값·대체 응답 오탐(2026-11-01 점검 H-1·M-2).
기대값은 점검 문서(docs/audits/2026-11-01-chat-proxy-table-audit.md 3절) 애매 15문장 표와 사용자 승인 표."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

import app.services.medical_llm_service as mls
from app.core.config import Settings
from app.services import chat_keywords
from app.services.chat_service import ChatService
from app.services.medical_llm_service import MedicalLlmService


@pytest.mark.parametrize("message, expected", [
    ("아들이 열이 나요", True), ("딸이 쓰러졌어요", True), ("손자가 경련해요", True), ("직원이 쓰러졌어요", True),
    ("환자가 의식이 없어요", True), ("엄마 기침이 심해요", True), ("할머니 열나요", True), ("우리 아버지 숨이 차요", True),
    ("어머니가 쓰러지셨는데 제가 어떻게 해요?", True), ("아버지가 숨을 못 쉬는데 저는 뭘 해야 해요", True),
    ("엄마가 죽고 싶다고 저한테 말했어요", False), ("엄마는 괜찮은데 제가 어지러워요", False), ("어머니께 제가 약을 드렸어요", False),
    ("옆집 할머니가 쓰러지셨어요", True), ("저희 어머니가 아파요", True), ("어머니가요 기침을 해요", True),
    # 받아들이지 않는 것: 낱말 속 글자·화자 본인
    ("모형이 부서졌어요", False), ("딸기가 맛있어요", False), ("아이고 허리야", False), ("엄마 나 배 아파", False), ("엄마 저 어지러워요", False),
    ("형이상학 책을 읽었어요", False), ("기침이 나요", False),
])
def test_proxy_subject_scope(message: str, expected: bool) -> None:
    assert MedicalLlmService.proxy_subject(message) is expected


def test_self_harm_family_choice_still_unchanged() -> None:
    # 새 호칭은 대리 판정에만 - 자해 응답 가족용 선택은 넓히지 않는다.
    assert MedicalLlmService.about_family("딸이 죽고 싶대요") is False
    assert MedicalLlmService.about_family("엄마가 죽고 싶다고 저한테 말했어요") is True


@pytest.mark.parametrize("message, expected", [
    ("어머니께 제가 약을 드렸는데 쓰러지셨어요", True), ("딸한테 연락했어요 경련을 해요", True), ("옆 사람 쓰러졌어요", True),
    ("쓰러질 것 같아요", False), ("아이고 숨이 안 쉬어져요", False), ("제가 경련을 해요", False),
])
def test_mentions_other_person(message: str, expected: bool) -> None:
    assert MedicalLlmService.mentions_other_person(message) is expected


class _Loader:
    def __init__(self, **fields: str) -> None:
        self.fields = {"empathy": "걱정되셨겠어요.", "symptoms": "증상", "features": "", "advice": "", "reply": "", **fields}
        self.state = SimpleNamespace(model_name="fake-medgemma")

    def ensure_loaded(self) -> None:
        return None

    def generate_structured_text(self, system_prompt: str, user_prompt: str, prefill: str = "", **kwargs) -> str:  # noqa: ARG002
        return json.dumps(self.fields, ensure_ascii=False)


def _gen(message: str, proxy: bool = False) -> dict[str, Any]:
    return MedicalLlmService(Settings(), _Loader()).generate(message, {}, [], intent="emergency_guidance", proxy=proxy)


def _sentence(name: str):
    return next(s for s in mls.FIXED_SENTENCES if s.name == name)


def test_emergency_uses_carer_wording_when_other_person_is_mentioned() -> None:
    # 화자 본인 말("저한테")로 대리가 아니어도, 다른 사람 낱말이 있으면 응급 행동·경고 박스는 돌보는 사람 말투(H-1 ③).
    collapse = _sentence("_EMERGENCY_ACTIONS.collapse_fall")
    out = _gen("어머니께 제가 약을 드렸는데 쓰러지셨어요")
    assert collapse.proxy in out["reply"] and str(collapse) not in out["reply"]
    assert out["emergencyWarning"] == [mls._EMERGENCY_REPLY.proxy]
    own = _gen("쓰러질 것 같아요")
    assert str(collapse) in own["reply"] and own["emergencyWarning"] == [str(mls._EMERGENCY_REPLY)]


def test_fallback_emergency_box_uses_carer_wording_for_other_person() -> None:
    out = ChatService()._fallback_reply("아버지가 쓰러지셨는데 제가 어떻게 해요", [], emergency=True)
    assert out["emergencyWarning"] == [mls._EMERGENCY_REPLY.proxy]


@pytest.mark.parametrize("message, expected", [
    ("커피 가게 갔어요", False), ("커피가 맛있어요", False), ("커피가 쏟아져서 피가 나요", True), ("코피가 안 멈춰요", True), ("피가 나요", True),
])
def test_coffee_is_not_a_red_flag(message: str, expected: bool) -> None:
    assert chat_keywords.has_red_flag(message) is expected


def test_fallback_coffee_is_not_high() -> None:
    assert ChatService()._fallback_reply("커피 가게 갔어요", [])["riskLevel"] != "high"


def test_diabetes_alcohol_sentence_lives_in_the_table_only() -> None:
    assert not hasattr(mls, "_ALCOHOL_HYPOGLYCEMIA_SENTENCE")
    assert "저혈당" in _sentence("_DRUG_ALCOHOL_RULES.diabetes")
