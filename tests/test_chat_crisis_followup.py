from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from app.models.chat_log import ChatLog
from app.services import chat_service as cs
from app.services.chat_router_service import SYSTEM_PROMPT
from tests.test_chat_router_followup import CRISIS_HISTORY, _Router, _send, env  # noqa: F401 - env 는 픽스처

# 2026-10-26 점검 H-1·M-1·M-2: 위기 고정 응답 뒤 평범한 말까지 같은 응답에 갇히던 것.
# 라우터가 이번 문장 기준으로 자해가 아니라고 하면(self_harm=false) 원래 경로로 보내고 109 한 줄을 한 번 붙인다.

RESPONSE_KEYS = {
    "sessionId", "reply", "riskLevel", "recommendedAction", "reservationRequired", "intent", "engine", "modelName", "type",
    "ui", "tool", "toolData", "summary", "possibleCauses", "homeCare", "visitHospitalIf", "emergencyWarning", "chatLogId", "chatNo",
}


class _FlagRouter(_Router):
    def __init__(self, self_harm: bool, route: str = "symptom", **kwargs) -> None:
        super().__init__(**kwargs)
        self.self_harm, self.route_name = self_harm, route

    def route(self, message: str, history, ctx, prev_reservation=None) -> dict[str, Any]:
        out = super().route(message, history, ctx, prev_reservation)
        return {**out, "route": self.route_name, "self_harm": self.self_harm, "emergency": out["emergency"] or self.self_harm}


def _crisis(env, session: str = "s1") -> None:
    first = _send(env, "죽고 싶어", session)  # 키워드 → 위기 고정 응답(기록에 selfHarm)
    assert "109" in first["reply"] and not first["reply"].startswith(cs._CRISIS_FOLLOWUP)


def _last_meta(env) -> dict[str, Any]:
    log = env.db.query(ChatLog).order_by(ChatLog.id.desc()).first()
    return json.loads(log.context_json)["upstreamMeta"]


def _logs(caplog) -> str:
    return "\n".join(r.getMessage() for r in caplog.records)


def test_prompt_limits_self_harm_to_this_sentence_or_its_continuation() -> None:
    assert "이번 문장이 그 위기를 이어 말하면" in SYSTEM_PROMPT
    assert "직전 대화의 위기와 상관없이 false" in SYSTEM_PROMPT
    assert "직전 대화의 자해·자살 언급만으로는 true로 하지 않는다" in SYSTEM_PROMPT
    assert "문장이나 직전 대화 흐름에 드러나면" not in SYSTEM_PROMPT


@pytest.mark.parametrize("message", ["고마워요. 이야기하니까 좀 나아졌어요", "손주가 놀러 와서 기분이 좋아요", "오늘 날씨가 좋네요"])
def test_ordinary_turn_after_crisis_is_released_with_followup(env, caplog, message: str) -> None:
    _crisis(env)
    # 라우터가 앞 턴 위기 때문에 응급·요약에 자살을 남겨도, 이번 문장 self_harm=false 면 위기 응답에 가두지 않는다.
    env.router = _FlagRouter(self_harm=False, route="general", emergency=True, summary="자살 위기 이후 대화")
    with caplog.at_level(logging.INFO):
        out = _send(env, message, "s1", CRISIS_HISTORY)
    assert out["reply"] == f"{cs._CRISIS_FOLLOWUP}\n모델 답변"
    assert env.llm.calls == 1
    meta = _last_meta(env)
    assert not meta.get("selfHarm") and meta["crisisFollowup"] is True
    assert meta["router"]["reason"] == "release-after-crisis" and meta["router"]["self_harm"] is False
    joined = _logs(caplog)
    assert "[CHAT-SAFETY] crisis_released=true route=symptom trigger=router" in joined
    assert message not in joined


def test_symptom_after_crisis_gets_medical_answer_with_followup(env) -> None:
    _crisis(env)
    env.router = _FlagRouter(self_harm=False, summary="복통 / 앞서부터 / 원인 궁금")
    out = _send(env, "그런데 아까부터 배가 아파요", "s1", CRISIS_HISTORY)
    assert out["reply"].startswith(cs._CRISIS_FOLLOWUP) and env.llm.calls == 1
    assert "selfHarm" not in _last_meta(env)


def test_followup_line_is_added_only_once(env) -> None:
    _crisis(env)
    env.router = _FlagRouter(self_harm=False, route="general")
    second = _send(env, "고마워요", "s1", CRISIS_HISTORY)
    third = _send(env, "손주가 놀러 와서 기분이 좋아요", "s1", CRISIS_HISTORY + [
        {"role": "user", "content": "고마워요"}, {"role": "assistant", "content": second["reply"]}])
    assert second["reply"].startswith(cs._CRISIS_FOLLOWUP)
    assert third["reply"] == "모델 답변"
    assert "crisisFollowup" not in _last_meta(env)


def test_crisis_continuation_keeps_crisis_reply(env) -> None:
    _crisis(env)
    env.router = _FlagRouter(self_harm=True, summary="위기 지속")
    out = _send(env, "아직도 너무 힘들어요", "s1", CRISIS_HISTORY)
    assert "109" in out["reply"] and not out["reply"].startswith(cs._CRISIS_FOLLOWUP)
    assert env.llm.calls == 0 and _last_meta(env)["selfHarmTrigger"] == "router"


def test_self_harm_plus_reservation_after_crisis_keeps_crisis_reply(env) -> None:  # 점검 M-2
    _crisis(env)
    env.router = _FlagRouter(self_harm=True, route="reservation", summary="무기력")
    out = _send(env, "다 끝내고 싶어, 병원 예약해줘", "s1", CRISIS_HISTORY)
    assert env.orch.calls == 0
    assert "109" in out["reply"] and not out["reply"].startswith(cs._CRISIS_FOLLOWUP)


def test_reservation_after_crisis_resumes_with_flag_false(env) -> None:
    _crisis(env)
    env.router = _FlagRouter(self_harm=False, emergency=True, summary="자살 위기 이후 대화")
    out = _send(env, "응 예약해줘", "s1", CRISIS_HISTORY)
    assert env.orch.calls == 1 and out["intent"] == "hospital_reservation"
    assert out["reply"].startswith(cs._CRISIS_FOLLOWUP)


def test_emergency_keyword_after_crisis_keeps_crisis_reply(env) -> None:
    _crisis(env)
    env.router = _FlagRouter(self_harm=False, emergency=True, summary="자살 위기 이후 실신")
    out = _send(env, "아까 잠깐 기절했어요", "s1", CRISIS_HISTORY)
    assert "109" in out["reply"] and env.llm.calls == 0


def test_clean_router_after_crisis_still_gets_followup_once(env) -> None:
    _crisis(env)
    env.router = _FlagRouter(self_harm=False, route="general")
    out = _send(env, "고마워요", "s1", CRISIS_HISTORY)
    assert out["reply"] == f"{cs._CRISIS_FOLLOWUP}\n모델 답변"


def test_no_release_without_server_crisis_record(env) -> None:
    # 서버 기록상 직전 턴이 위기가 아니면 클라이언트 history만으로는 풀지 않는다(sh.11 형태 - 라우터·history 근거 그대로).
    env.router = _FlagRouter(self_harm=False, emergency=True, summary="응급")
    out = _send(env, "어떻게 해야 할지 모르겠어", "s1", CRISIS_HISTORY)
    assert "109" in out["reply"] and env.llm.calls == 0


# ---- 점검 M-1: 라우터 실패 폴백 ------------------------------------------------------------------------------


def test_router_failure_short_continuation_gets_crisis_reply(env, caplog) -> None:
    _crisis(env)
    env.router = _Router(fail="api_529")
    with caplog.at_level(logging.INFO):
        out = _send(env, "어떻게 해야 할지 모르겠어", "s1", CRISIS_HISTORY)
    assert "109" in out["reply"] and not out["reply"].startswith(cs._CRISIS_FOLLOWUP)
    joined = _logs(caplog)
    assert "[CHAT-SAFETY] crisis_continues_fallback=true text_len=" in joined
    assert "모르겠어" not in joined


def test_router_failure_without_server_record_uses_history(env) -> None:
    env.router = _Router(fail="api_529")
    out = _send(env, "어떻게 해야 할지 모르겠어", "s1", CRISIS_HISTORY)
    assert "109" in out["reply"] and env.llm.calls == 0


@pytest.mark.parametrize("message", ["그런데 아까부터 배가 아파요", "고마워요. 이야기하니까 정말 많이 나아졌어요", "응 예약해줘"])
def test_router_failure_other_turns_go_normal_with_followup(env, message: str) -> None:
    _crisis(env)
    env.router = _Router(fail="api_529")
    out = _send(env, message, "s1", CRISIS_HISTORY)
    assert out["reply"].startswith(cs._CRISIS_FOLLOWUP)
    assert "selfHarm" not in _last_meta(env)


def test_router_failure_without_crisis_history_is_unchanged(env) -> None:
    env.router = _Router(fail="api_529")
    out = _send(env, "어떻게 해야 할지 모르겠어")
    assert not out["reply"].startswith(cs._CRISIS_FOLLOWUP) and "109" not in out["reply"]


# ---- 점검 L-1: meta.router.self_harm ------------------------------------------------------------------------


@pytest.mark.parametrize("flag", [True, False])
def test_router_self_harm_is_recorded(env, flag: bool) -> None:
    env.router = _FlagRouter(self_harm=flag)
    _send(env, "요즘 잠이 안 와요")
    assert _last_meta(env)["router"]["self_harm"] is flag


def test_router_self_harm_is_none_when_router_not_used(env) -> None:
    _send(env, "죽고 싶어")  # 키워드 선분기 - 라우터를 부르지 않는다
    assert _last_meta(env)["router"]["self_harm"] is None
    env.router = _Router(fail="parse")
    _send(env, "요즘 잠이 안 와요", "s2")
    assert _last_meta(env)["router"]["self_harm"] is None


def test_response_keys_unchanged_on_release(env) -> None:
    _crisis(env)
    env.router = _FlagRouter(self_harm=False, route="general")
    out = _send(env, "고마워요", "s1", CRISIS_HISTORY)
    assert set(out) == RESPONSE_KEYS
