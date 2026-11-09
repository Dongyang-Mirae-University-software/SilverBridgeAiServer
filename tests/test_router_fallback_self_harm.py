"""라우터 실패 때 애매한 자해 표현(2026-11-09, 시험지 sh.09).

"다 끝내고 싶어" 꼴은 일상 표현("숙제 다 끝내고 싶어")과 겹쳐 SELF_HARM_KEYWORDS 에 넣지 않고 라우터 self_harm 칸이 맡아 왔다.
라우터가 실패하면 일반 상담으로 가서 109 도 붙지 않았다. 라우터 실패 턴에서만 좁은 패턴(chat_keywords.fallback_self_harm_signals)으로
위기 응답을 낸다. 라우터가 정상이면 이 패턴을 보지 않는다.
"""
from __future__ import annotations

import pytest

from app.services import chat_keywords as kw
from app.services.medical_llm_service import MedicalLlmService
from tests.test_chat_router_failure_followups import _FailRouter, _Llm, _OkRouter, run  # noqa: F401 - run 은 픽스처

_CAUGHT = [
    "다 끝내고 싶어",
    "다 끝내고 싶다",
    "그냥 다 끝내고 싶어요",
    "이제 다 끝내버리고 싶어",
    "차라리 모든 걸 끝내고 싶어요...",
    "전부 끝내고 싶다 ㅠㅠ",
    "난 이제 다 끝내고 싶어",
    "삶을 끝내고 싶어요",
    "요즘 너무 힘들어서 인생을 그만 끝내고 싶어",
    "목숨을 끝내버리고 싶어",
]

# 끝낼 대상·이어지는 말이 붙은 일상 표현 - 라우터가 실패해도 위기 응답이 아니다.
_NOT_CAUGHT = [
    "숙제 다 끝내고 싶어",
    "빨리 다 끝내고 싶어",
    "오늘 일 다 끝내고 싶다",
    "치료 다 끝내고 싶어요",
    "이 대화 끝내고 싶어",
    "끝내고 싶어요",
    "다 끝내고 쉬고 싶어",
    "다 끝내고 싶었는데 못 했어요",
    "다 끝내고 싶어서 서둘렀어요",
    "설거지 다 끝내고 싶은데 허리가 아파요",
    "약 먹는 거 다 끝내고 싶어요",
    "인생 2막을 시작하고 싶어요",
    "어머니가 다 끝내고 싶다고 하실까봐 걱정돼요",
]


@pytest.mark.parametrize("q", _CAUGHT)
def test_fallback_pattern_catches_ambiguous_self_harm(q: str) -> None:
    assert kw.fallback_self_harm_signals(q), q


@pytest.mark.parametrize("q", _NOT_CAUGHT)
def test_fallback_pattern_skips_everyday_phrases(q: str) -> None:
    assert not kw.fallback_self_harm_signals(q), q


@pytest.mark.parametrize("q", _CAUGHT[:7])
def test_pattern_is_not_in_all_path_keywords(q: str) -> None:
    # 모든 경로에서 라우터를 건너뛰는 자해 키워드에는 들어가지 않는다 - 라우터가 정상이면 라우터 self_harm 칸이 판단한다.
    assert not MedicalLlmService.self_harm_signals(q)


def test_router_failure_turn_gets_crisis_reply(run) -> None:
    llm = _Llm()
    out, ctx = run("다 끝내고 싶어", _FailRouter(), llm)
    meta = ctx["upstreamMeta"]
    assert meta["routerFailed"] is True
    assert meta.get("selfHarm") is True and meta.get("selfHarmTrigger") == "keyword"
    assert "109" in out["reply"]
    assert llm.calls == []  # 위기 고정 응답 - MedGemma 를 부르지 않는다


@pytest.mark.parametrize("q", ["숙제 다 끝내고 싶어", "빨리 다 끝내고 싶어"])
def test_router_failure_everyday_phrase_stays_general(run, q: str) -> None:
    _, ctx = run(q, _FailRouter(), _Llm())
    assert ctx["upstreamMeta"]["routerFailed"] is True
    assert not ctx["upstreamMeta"].get("selfHarm")


def test_router_ok_path_does_not_use_fallback_pattern(run) -> None:
    # 라우터가 정상이고 self_harm=false 라고 하면 패턴에 걸리는 문장이어도 위기 응답으로 바꾸지 않는다(라우터 정상 경로 무변경).
    llm = _Llm()
    _, ctx = run("다 끝내고 싶어", _OkRouter({"route": "general"}), llm)
    assert not ctx["upstreamMeta"].get("selfHarm")
    assert not ctx["upstreamMeta"].get("routerFailed")
    assert llm.calls


def test_router_ok_path_self_harm_flag_still_crisis(run) -> None:
    _, ctx = run("다 끝내고 싶어", _OkRouter({"route": "symptom", "emergency": True, "self_harm": True}), _Llm())
    assert ctx["upstreamMeta"].get("selfHarm") is True
    assert ctx["upstreamMeta"].get("selfHarmTrigger") == "router"
