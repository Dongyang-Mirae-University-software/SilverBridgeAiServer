from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from app.services.chat_router_service import ROUTE_SCHEMA, SYSTEM_PROMPT, ChatRouterService
from app.services.chat_service import ChatService
from tests.test_chat_router_followup import _Router, _send, env  # noqa: F401 - env 는 픽스처

# 2026-10-25 라우터 자해 칸: router 근거를 요약 문구가 아니라 라우터 self_harm 칸으로도 판정한다.
# (가벼운 모델은 "다 끝내고 싶어"를 응급으로 보고도 요약에 자해를 쓰지 않아 109 위기 응답이 빠졌다.)


class _FakeGpt:
    def __init__(self, reply: dict) -> None:
        self._reply = reply
        self.state = type("S", (), {"model_name": "fake-gpt"})()

    def generate_structured_text(self, system_prompt: str, user_prompt: str, schema: dict | None = None, **kwargs) -> str:  # noqa: ARG002
        return json.dumps(self._reply, ensure_ascii=False)


def _route(reply: dict) -> dict[str, Any]:
    return ChatRouterService(_FakeGpt(reply), 8).route("다 끝내고 싶어", [], {})


def test_schema_and_prompt_require_self_harm_field() -> None:
    assert ROUTE_SCHEMA["properties"]["self_harm"] == {"type": "boolean"}
    assert "self_harm" in ROUTE_SCHEMA["required"]
    assert '"self_harm":false' in SYSTEM_PROMPT
    # 과장·부정·일을 끝내고 싶다는 뜻은 false 라는 기준이 프롬프트에 있어야 한다.
    assert "아파 죽겠어" in SYSTEM_PROMPT and "그런 생각은 없어요" in SYSTEM_PROMPT


def test_router_self_harm_true_forces_emergency() -> None:
    out = _route({"route": "symptom", "symptom_summary": "무기력감 호소", "emergency": False, "self_harm": True, "reason": "r"})
    assert out["self_harm"] is True
    assert out["emergency"] is True  # 두 칸이 어긋나도 위기 쪽으로


@pytest.mark.parametrize("reply", [
    {"route": "symptom", "symptom_summary": "s", "emergency": False, "reason": "r"},  # 칸 없음(옛 응답 형태)
    {"route": "symptom", "symptom_summary": "s", "emergency": False, "self_harm": "true", "reason": "r"},  # 불리언이 아님
])
def test_router_missing_or_malformed_self_harm_is_false(reply: dict) -> None:
    out = _route(reply)
    assert out["self_harm"] is False and out["emergency"] is False


def test_trigger_router_from_flag_even_without_summary_words() -> None:
    route_info = {"route": "symptom", "emergency": True, "self_harm": True, "symptom_summary": "무기력감과 절망감 호소"}
    assert ChatService._self_harm_trigger(False, route_info, []) == "router"


def test_summary_words_alone_no_longer_trigger() -> None:
    # 2026-10-27 요약 문구 판정 제거(의도된 변경): 칸이 false면 요약에 자살·자해가 있어도 위기 응답 근거가 아니다(일반 응급 안내로).
    route_info = {"route": "symptom", "emergency": True, "self_harm": False, "symptom_summary": "자살 생각을 표현함"}
    assert ChatService._self_harm_trigger(False, route_info, []) is None


def test_summary_helper_removed() -> None:
    from app.services import medical_llm_service as mls
    assert not hasattr(mls.MedicalLlmService, "self_harm_in_summary")
    assert not hasattr(mls, "_SUMMARY_NEGATED") and not hasattr(mls, "_SUMMARY_SELF_HARM")


def test_no_trigger_when_flag_false_and_summary_negated() -> None:
    route_info = {"route": "symptom", "emergency": True, "self_harm": False, "symptom_summary": "흉통 호소, 자해 생각 없음"}
    assert ChatService._self_harm_trigger(False, route_info, []) is None


def test_no_trigger_when_not_emergency_and_no_flag() -> None:
    assert ChatService._self_harm_trigger(False, {"route": "general", "emergency": False}, []) is None


# ---- 라우터 뒤 예약 고정(운영·평가 스크립트 공용) -------------------------------------------------------


def test_override_symptom_plus_reservation_goes_to_reservation() -> None:
    # rs.02: 응급 신호 없는 "증상 + 예약 요청"은 라우터가 증상 상담으로 줘도 예약으로 고정한다.
    message = "머리가 아파서 병원 예약하고 싶어요"
    out = ChatService._apply_router_overrides({"route": "symptom", "emergency": False, "normalized_message": "x"}, message)
    assert out["route"] == "reservation" and out["reason"] == "new-request-override"
    assert out["normalized_message"] == message


@pytest.mark.parametrize("route_info", [
    {"route": "symptom", "emergency": True},  # rs.01 가슴 조임 + 예약 = 응급이면 상담 유지
    {"route": "symptom", "emergency": True, "self_harm": True},  # 위기 문장 + 예약 단어도 상담 유지
])
def test_override_keeps_symptom_when_emergency(route_info: dict) -> None:
    out = ChatService._apply_router_overrides(route_info, "머리가 아파서 병원 예약하고 싶어요")
    assert out["route"] == "symptom"


# ---- 챗 서비스 전체 경로 ------------------------------------------------------------------------------


class _FlagRouter(_Router):
    def __init__(self, self_harm: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.self_harm = self_harm

    def route(self, message: str, history, ctx, prev_reservation=None) -> dict[str, Any]:
        out = super().route(message, history, ctx, prev_reservation)
        return {**out, "self_harm": self.self_harm, "emergency": out["emergency"] or self.self_harm}


def test_chat_gives_crisis_reply_from_router_flag(env, caplog) -> None:  # noqa: F811
    env.router = _FlagRouter(self_harm=True, summary="무기력감 호소")
    with caplog.at_level(logging.INFO):
        out = _send(env, "다 끝내고 싶어")
    assert "109" in out["reply"]
    assert "[CHAT-SAFETY] self_harm=true path=main trigger=router" in "\n".join(r.getMessage() for r in caplog.records)
    assert env.llm.calls == 0


def test_chat_no_crisis_reply_when_flag_false(env, caplog) -> None:  # noqa: F811
    env.router = _FlagRouter(self_harm=False, summary="숙제를 마치고 싶다는 표현")
    with caplog.at_level(logging.INFO):
        out = _send(env, "오늘 숙제 다 끝내고 싶어")
    assert "109" not in out["reply"]
    assert "self_harm=true" not in "\n".join(r.getMessage() for r in caplog.records)
