"""생성 실패 대체 응답의 응급 행동 줄·경계 질환 문장, 지금 의식이 있다는 말(2026-11-05, #174 점검 H-1·M-1·L-1·L-3).

- H-1: MedGemma 대기 초과(응급 3초)·생성 실패·마감 초과여도 응급 대체 응답에 가슴압박·AED 등 응급 행동 줄이 붙는다.
- M-1: "말은 해요"·"숨은 쉬어요"처럼 지금 의식이 있다는 말이면 키워드든 라우터 칸이든 가슴압박 줄을 붙이지 않는다(119는 그대로).
- L-1: 응급이 아닌 대체 응답에도 라우터 경계 질환 문장이 붙는다.
"""
from __future__ import annotations

import re
import tempfile
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.chat_log import ChatLog
from app.schemas.chat_schema import ChatContext, ChatRequest
from app.services import chat_keywords as kw
from app.services import chat_layout
from app.services import chat_service as cs
from app.services.medical_llm_service import DRUG_ALCOHOL_SENTENCES, MedicalLlmService

CPR = re.compile(r"가슴압박\(가슴 한가운데를 세게, 빠르게")
EMOJI = re.compile(r"[\U0001F000-\U0001FAFF⌀-⏿☀-➿⬀-⯿]")


def _fb(message: str, **kwargs: Any) -> dict[str, Any]:
    return cs.ChatService()._fallback_reply(message, [], **kwargs)


# ---- H-1 응급 대체 응답 ------------------------------------------------------------------------------------

@pytest.mark.parametrize("message", ["남편이 쓰러져서 숨을 안 쉬어요", "엄마가 불러도 대답이 없어요"])
def test_emergency_fallback_keeps_cpr_line_for_keyword_arrest(message: str) -> None:
    out = _fb(message, emergency=True)
    assert out["riskLevel"] == "high" and out["emergencyWarning"]
    assert CPR.search(out["reply"]) and "AED" in out["reply"]


def test_emergency_fallback_uses_router_arrest_hint() -> None:
    out = _fb("아버지가 꼼짝을 안 하세요", emergency=True, arrest_hint=True)
    assert CPR.search(out["reply"])


def test_emergency_fallback_without_hint_has_no_cpr_for_non_keyword_text() -> None:
    assert not CPR.search(_fb("아버지가 꼼짝을 안 하세요", emergency=True)["reply"])


def test_emergency_fallback_keeps_other_actions() -> None:
    out = _fb("가슴이 아파요", emergency=True)
    assert "직접 운전" in out["reply"]  # chest_stroke 줄


def test_emergency_fallback_layout_keeps_119_headline_and_only_siren() -> None:
    out = chat_layout.apply(dict(_fb("남편이 숨을 안 쉬어요", emergency=True)), DRUG_ALCOHOL_SENTENCES)
    first = out["reply"].split("\n")[0]
    assert first.startswith("🚨 ") and "119" in first
    assert EMOJI.findall(out["reply"]) == ["🚨"]
    assert CPR.search(out["reply"])


def test_ward_concise_fallback_keeps_cpr_line() -> None:
    out = chat_layout.apply(dict(_fb("남편이 숨을 안 쉬어요", emergency=True)), DRUG_ALCOHOL_SENTENCES, concise=True)
    assert CPR.search(out["reply"]) and "AED" in out["reply"]


def test_self_harm_fallback_is_unchanged() -> None:
    out = _fb("죽고 싶어요", emergency=True, arrest_hint=True)
    assert "109" in out["reply"] and not CPR.search(out["reply"])


# ---- L-1 응급이 아닌 대체 응답의 경계 질환 문장 ------------------------------------------------------------------------

def test_non_emergency_fallback_adds_urgent_condition_sentence() -> None:
    out = _fb("왼쪽 종아리만 부었어요", urgent_condition="dvt")
    assert out["riskLevel"] != "high" and out["reply"].count("심부정맥혈전증") == 1


def test_emergency_fallback_does_not_add_urgent_condition_sentence() -> None:
    assert "심부정맥혈전증" not in _fb("가슴이 아파요", emergency=True, urgent_condition="dvt")["reply"]


def test_unknown_urgent_condition_adds_nothing_in_fallback() -> None:
    assert "오늘 안에 진료" not in _fb("다리가 부었어요", urgent_condition="made_up")["reply"]


# ---- M-1 지금 의식이 있다는 말 ---------------------------------------------------------------------------------

@pytest.mark.parametrize("message", [
    "엄마가 쓰러졌는데 지금 말은 해요",
    "아버지가 쓰러지셨는데 대답은 하세요",
    "할머니가 쓰러지셨는데 숨은 쉬어요",
    "쓰러졌는데 의식은 있어요",
])
def test_awake_person_gets_no_cpr_line_from_keyword_or_router(message: str) -> None:
    assert MedicalLlmService.emergency_signals(message)  # 응급(119)은 그대로
    assert not kw.ARREST_SIGNAL.search(kw.compact(message))
    names = [n for n, _ in MedicalLlmService.emergency_action_sentences(message, True, arrest_hint=True)]
    assert "cardiac_arrest" not in names


@pytest.mark.parametrize("message", [
    "엄마가 불러도 대답을 안 해요",
    "남편이 숨을 안 쉬어요",
    "아버지가 쓰러져서 움직이지 않아요 숨도 안 쉬어요",
    "할머니가 말을 못 하시고 눈을 안 떠요",
    "말을 해도 대답이 없어요 숨을 안 쉬어요",
    "엄마가 쓰러졌는데 눈은 뜨고 있는데 불러도 반응이 없어요",  # 눈을 뜬 것은 의식의 근거가 아니다
])
def test_negated_awake_words_still_count_as_arrest(message: str) -> None:
    assert kw.ARREST_SIGNAL.search(kw.compact(message))
    assert "cardiac_arrest" in [n for n, _ in MedicalLlmService.emergency_action_sentences(message, True)]


# ---- 서비스 경로: MedGemma 대기 초과 ------------------------------------------------------------------------------

class _Router:
    def route(self, message, history, ctx, prev_reservation=None) -> dict[str, Any]:  # noqa: ARG002
        return {"route": "symptom", "normalized_message": message, "symptom_summary": message, "emergency": True,
                "self_harm": False, "cardiac_arrest": True, "urgent_condition": None, "reason": "t", "engine": "fake"}


class _BusyLlm:
    def generate(self, *args, **kwargs):
        raise TimeoutError("busy")


def test_service_busy_generation_still_sends_cpr_line(monkeypatch) -> None:
    engine = create_engine(f"sqlite:///{tempfile.mkdtemp(prefix='sb_fb_')}/t.db")
    ChatLog.__table__.create(bind=engine)
    db = sessionmaker(bind=engine)()
    monkeypatch.setattr(cs, "get_chat_router_service", lambda: _Router())
    monkeypatch.setattr(cs, "get_reservation_credential_service", lambda: object())
    monkeypatch.setattr(cs, "get_reservation_api_client", lambda: object())
    monkeypatch.setattr(cs, "get_medical_llm_service", lambda: _BusyLlm())
    svc = cs.ChatService()
    svc._settings = SimpleNamespace(has_reservation_account_secret=True, claude_model_name="fake")
    try:
        out = svc.process_message(db, ChatRequest(userId="fb1", message="아버지가 꼼짝을 안 하세요", sessionId="s1",
                                                  context=ChatContext()))
    finally:
        db.close()
    assert out["riskLevel"] == "high"
    assert CPR.search(out["reply"]) and "AED" in out["reply"]
    # 2026-11-09 이모지 끔(기본): 🚨 대신 첫 줄이 119 전화 문장, 답 어디에도 이모지 없음.
    first = out["reply"].split("\n")[0]
    assert "119" in first and "전화" in first and EMOJI.findall(out["reply"]) == []


@pytest.mark.parametrize("message", [
    "아버지가 불러도 대답이 없었는데 지금은 말은 하세요",  # 심정지 낱말 + 지금 의식 있음 - 키워드 경로에서도 빼야 한다
    "흔들어도 반응이 없더니 지금은 숨은 쉬어요",
])
def test_keyword_arrest_words_with_awake_now_get_no_cpr(message: str) -> None:
    assert not kw.ARREST_SIGNAL.search(kw.compact(message))
    assert "cardiac_arrest" not in [n for n, _ in MedicalLlmService.emergency_action_sentences(message, True)]
