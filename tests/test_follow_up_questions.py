"""이어서 물어볼 질문(응답 suggestedQuestions, 2026-10-30).
위기·응급·되묻기·예약 선택지 답에는 붙지 않고, 모델이 실패하거나 늦으면 규칙 문장으로 대신하며, 로그에 문장을 남기지 않는다."""
from __future__ import annotations

import logging
import time
from types import SimpleNamespace
from typing import Any

import pytest

from app.schemas.chat_schema import ChatContext, ChatRequest
from app.services import chat_service as cs
from app.services import follow_up_questions as fq
from tests.test_reservation_confirm_chat_service import env  # noqa: F401

ON = SimpleNamespace(follow_up_enabled=True, follow_up_timeout_sec=1.0, follow_up_model="fake",
                     has_reservation_account_secret=True, claude_model_name="fake")
ANSWER = {"reply": "두통은 쉬면 나아지는 경우가 많아요.", "riskLevel": "medium", "intent": "medical_advice", "type": "message", "ui": None}
SYMPTOM = {"route": "symptom", "emergency": False}


def _ok(**kw: Any) -> bool:
    return fq.should_suggest({**ANSWER, **kw}, SYMPTOM, "머리가 아파요", emergency_text=False, self_harm_text=False)


@pytest.mark.parametrize("override", [
    {"riskLevel": "high"},
    {"intent": "emergency_guidance"},
    {"intent": "hospital_reservation"},
    {"meta": {"selfHarm": True}},
    {"meta": {"crisisFollowup": True}},
    {"meta": {"emergencyFollowup": True}},
    {"meta": {"followupAsked": True}},
    {"meta": {"generationFailed": True}},
    {"ui": {"kind": "select", "field": "date"}},
    {"type": "tool_result"},
])
def test_no_questions_on_crisis_emergency_followup_or_reservation_answers(override: dict) -> None:
    assert _ok() is True
    assert _ok(**override) is False


def test_no_questions_when_route_or_text_is_emergency() -> None:
    assert not fq.should_suggest(ANSWER, {"route": "symptom", "emergency": True}, "x", emergency_text=False, self_harm_text=False)
    assert not fq.should_suggest(ANSWER, {"route": "reservation"}, "x", emergency_text=False, self_harm_text=False)
    assert not fq.should_suggest(ANSWER, SYMPTOM, "x", emergency_text=True, self_harm_text=False)
    assert not fq.should_suggest(ANSWER, SYMPTOM, "x", emergency_text=False, self_harm_text=True)


def test_sanitize_keeps_short_plain_questions_only() -> None:
    raw = ["병원에 가야 할까요?", 3, "줄\n바꿈 질문", "아주아주아주아주아주아주 긴 질문 문장이에요", "010-1234-5678로 연락",
           "병원에 가야 할까요?", "머리가 아파요", "집에서 뭘 하면 좋아요?", "어느 진료과로 가요?"]
    assert fq.sanitize(raw, exclude="머리가 아파요") == ["병원에 가야 할까요?", "줄 바꿈 질문", "집에서 뭘 하면 좋아요?"]
    assert fq.sanitize("병원에 가야 할까요?") == []
    assert fq.sanitize(None) == []


def test_rule_questions_follow_route_and_proxy() -> None:
    assert fq.rule_questions("symptom", proxy=False)[0] == "병원에 가야 할까요?"
    assert fq.rule_questions("symptom", proxy=True)[0] == "병원에 모시고 가야 할까요?"
    assert fq.rule_questions("general", proxy=False) != fq.rule_questions("general", proxy=True)
    for proxy in (False, True):
        for route in ("symptom", "general"):
            assert fq.sanitize(fq.rule_questions(route, proxy=proxy)) == fq.rule_questions(route, proxy=proxy)


def test_disabled_or_missing_setting_gives_empty_list() -> None:
    assert fq.suggested_questions(SimpleNamespace(follow_up_enabled=False), ANSWER, SYMPTOM, "머리가 아파요", proxy=False) == []
    assert fq.suggested_questions(SimpleNamespace(), ANSWER, SYMPTOM, "머리가 아파요", proxy=False) == []


def test_model_questions_are_used_and_log_has_no_text(monkeypatch, caplog) -> None:
    monkeypatch.setattr(fq, "_model_questions", lambda s, m, r, p: ["물을 많이 마셔도 되나요?", "언제 병원에 가요?"])
    with caplog.at_level(logging.INFO, logger=fq.__name__):
        out = fq.suggested_questions(ON, ANSWER, SYMPTOM, "머리가 아파요 홍길순", proxy=False)
    assert out == ["물을 많이 마셔도 되나요?", "언제 병원에 가요?"]
    assert "source=model count=2" in caplog.text
    for text in ("물을 많이", "언제 병원", "머리가 아파요", "홍길순", ANSWER["reply"]):
        assert text not in caplog.text


def test_model_failure_falls_back_to_rules(monkeypatch, caplog) -> None:
    def boom(*a: Any) -> list[str]:
        raise ValueError("모델 출력 원문 홍길순")

    monkeypatch.setattr(fq, "_model_questions", boom)
    with caplog.at_level(logging.INFO, logger=fq.__name__):
        out = fq.suggested_questions(ON, ANSWER, SYMPTOM, "머리가 아파요", proxy=True)
    assert out == fq.rule_questions("symptom", proxy=True)
    assert "cause=ValueError" in caplog.text and "홍길순" not in caplog.text


def test_slow_model_does_not_hold_the_reply(monkeypatch) -> None:
    monkeypatch.setattr(fq, "_model_questions", lambda *a: (time.sleep(3), ["늦은 질문이에요?"])[1])
    fast = SimpleNamespace(**{**vars(ON), "follow_up_timeout_sec": 0.5})
    started = time.perf_counter()
    out = fq.suggested_questions(fast, ANSWER, SYMPTOM, "머리가 아파요", proxy=False)
    assert time.perf_counter() - started < 1.5
    assert out == fq.rule_questions("symptom", proxy=False)


# --- 실제 챗 흐름 -------------------------------------------------------------

class _SymptomRouter:
    def route(self, message: str, history, ctx, prev_reservation=None) -> dict[str, Any]:  # noqa: ARG002
        return {"route": "symptom", "normalized_message": message, "symptom_summary": "두통", "emergency": False,
                "reason": "t", "engine": "fake"}


class _CalmLlm:
    def generate(self, *a, **k) -> dict[str, Any]:  # noqa: ARG002
        return {**ANSWER, "recommendedAction": "self_care", "reservationRequired": False, "engine": "fake", "modelName": "fake",
                "tool": None, "data": None}


def _send(env, message: str, ward_id: str | None = None) -> dict[str, Any]:  # noqa: F811
    payload = ChatRequest(userId="user-1", message=message, sessionId="s1", context=ChatContext(wardId=ward_id) if ward_id else None)
    return env.svc.process_message(env.db, payload)


def test_symptom_answer_carries_suggested_questions(env, monkeypatch) -> None:  # noqa: F811
    monkeypatch.setattr(cs, "get_chat_router_service", lambda: _SymptomRouter())
    monkeypatch.setattr(cs, "get_medical_llm_service", lambda: _CalmLlm())
    seen: dict[str, Any] = {}

    def model(settings, message, reply, proxy):  # noqa: ANN001
        seen.update(message=message, reply=reply, proxy=proxy)
        return ["집에서 쉬면 될까요?"]

    monkeypatch.setattr(fq, "_model_questions", model)
    env.svc._settings = ON
    out = _send(env, "어제 저녁부터 머리가 지끈지끈 아프고 열은 없어요", ward_id="W00001")
    assert out["suggestedQuestions"] == ["집에서 쉬면 될까요?"]
    assert seen["proxy"] is True  # 대리 상담이면 가족에 대해 묻는 말투
    assert seen["reply"] == ANSWER["reply"]


def test_self_harm_turn_has_no_suggested_questions(env, monkeypatch) -> None:  # noqa: F811
    called: list[int] = []
    monkeypatch.setattr(fq, "_model_questions", lambda *a: called.append(1) or ["질문이에요?"])
    env.svc._settings = ON
    out = _send(env, "죽고 싶어요")
    assert out["suggestedQuestions"] == []
    assert called == []
