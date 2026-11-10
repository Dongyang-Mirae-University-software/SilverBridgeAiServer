"""전체 점검 1차(2026-11-10, docs/audits/2026-11-10-full-baseline-changes-security-audit.md) 판정 경계 고정.

H-3 깼다가 다시 반응·호흡이 없다는 문장은 회복이 아니다(가슴압박 줄) / H-4 라우터 실패 때 "다 끝내고 싶은데요" 말끝 변형 /
H-5 단계 되묻기 위험 신호 질문의 부정 답·"예전에도"·"응급실" / M-9 수용한 경계. 모든 문장은 합성 문장이다.
"""
from __future__ import annotations

import pytest

from app.services import chat_keywords as kw
from app.services import chat_service as cs
from app.services import followup_steps as fs
from app.services.medical_llm_service import MedicalLlmService
from tests.test_chat_router_failure_followups import _FailRouter, _Llm, _OkRouter, run  # noqa: F401 - run 은 픽스처


def _cpr(message: str, hint: bool) -> bool:
    return any(name == "cardiac_arrest" for name, _ in MedicalLlmService.emergency_action_sentences(message, arrest_hint=hint))


# ---- H-3 깼다가 다시 ---------------------------------------------------------------------------------------------

_RELAPSED = [
    "아까 깨셨는데 지금 다시 숨을 안 쉬어요",
    "잠깐 깼다가 다시 쓰러져서 의식이 없어요",
    "정신이 돌아왔다가 또 불러도 반응이 없어요",
    "금방 깨어났는데 지금은 숨을 안 쉬어요",
]


@pytest.mark.parametrize("q", _RELAPSED)
def test_relapse_after_recovery_is_not_recovery(q: str) -> None:
    assert not kw.arrest_conscious(kw.compact(q))
    # 키워드만(라우터 실패 경로의 응급 답)으로도, 라우터 심정지 힌트가 있어도 가슴압박 줄이 붙는다.
    assert _cpr(q, hint=False) and _cpr(q, hint=True)


@pytest.mark.parametrize("q", [
    "정신을 잃었다가 금방 깼어요",
    "쓰러졌다가 지금은 멀쩡해요 숨을 안 쉬었었대요",
    "아까 의식이 없었는데 지금은 깨셨어요",
    "깨어나셨어요 아까는 불러도 반응이 없었어요",
])
def test_recovery_stays_recovery(q: str) -> None:
    assert kw.arrest_conscious(kw.compact(q))
    assert not _cpr(q, hint=True)


@pytest.mark.parametrize("q", _RELAPSED)
def test_relapse_router_failure_route_is_emergency(q: str) -> None:
    # 라우터 실패 턴: 키워드 경로가 응급으로 보내고, 응급 답 후처리(emergency_action_sentences, 힌트 없음)가 가슴압박 줄을 붙인다.
    route = cs.ChatService._keyword_route(q)
    assert route["emergency"] is True and _cpr(q, hint=False)


# ---- H-4 라우터 실패 자해 말끝 ------------------------------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "이제 정말 다 끝내고 싶은데요",
    "다 끝내고 싶은데",
    "그냥 다 끝내고 싶다고요",
    "너무 지쳐서 다 끝내고 싶다는 생각이 들어요",
    "다 끝내고 싶다는 생각만 나요",
])
def test_fallback_self_harm_ending_variants(q: str) -> None:
    assert kw.fallback_self_harm_signals(q)


@pytest.mark.parametrize("q", [
    "숙제 다 끝내고 싶은데요",
    "다 끝내고 싶었는데 못 했어요",
    "다 끝내고 싶어서 서둘렀어요",
    "설거지 다 끝내고 싶은데 허리가 아파요",
    "다 끝내고 싶다는 생각은 없어요",
    "일을 다 끝내고 싶다고요",
])
def test_fallback_self_harm_everyday_stays_out(q: str) -> None:
    assert not kw.fallback_self_harm_signals(q)


def test_ending_variant_router_failure_turn_gets_crisis_reply(run) -> None:
    llm = _Llm()
    out, ctx = run("이제 정말 다 끝내고 싶은데요", _FailRouter(), llm)
    assert ctx["upstreamMeta"]["routerFailed"] is True and ctx["upstreamMeta"].get("selfHarm") is True
    assert "109" in out["reply"] and llm.calls == []


def test_ending_variant_router_ok_turn_follows_router(run) -> None:
    # 라우터가 정상이면 실패 전용 패턴을 보지 않는다 - 라우터 self_harm 칸이 판정한다.
    llm = _Llm()
    _, ctx = run("이제 정말 다 끝내고 싶은데요", _OkRouter({"route": "general"}), llm)
    assert not ctx["upstreamMeta"].get("selfHarm") and llm.calls
    _, ctx = run(
        "이제 정말 다 끝내고 싶은데요", _OkRouter({"route": "symptom", "emergency": True, "self_harm": True}), _Llm(), user="b",
    )
    assert ctx["upstreamMeta"].get("selfHarm") is True and ctx["upstreamMeta"].get("selfHarmTrigger") == "router"


# ---- H-5 단계 되묻기 위험 신호 질문 --------------------------------------------------------------------------------

def _first(symptom: str, answer: str) -> str:
    return fs.interpret(fs.new_state(symptom, False), answer)[0]


@pytest.mark.parametrize("symptom", ["가슴이 답답해요", "기침이 나요"])
@pytest.mark.parametrize("answer", ["안 답답해요", "숨이 안 가빠요", "숨은 안 차요", "안 힘들어요", "아니요", "그런 건 없는데요"])
def test_negated_answer_to_red_flag_question_is_no(symptom: str, answer: str) -> None:
    assert fs.red_flag_action(fs.new_state(symptom, False)) == "119"
    assert _first(symptom, answer) == "no"


@pytest.mark.parametrize("answer", ["예전에도 가끔 그랬어요", "응급실 가야 하나요?", "그래도 좀 걱정돼요"])
def test_yes_lookalike_is_not_yes(answer: str) -> None:
    assert _first("가슴이 답답해요", answer) != "yes"


@pytest.mark.parametrize("answer", ["네", "예", "응", "예 그래요", "그래요", "네 숨이 차요", "숨이 차요", "가슴이 조여요"])
def test_plain_yes_still_yes(answer: str) -> None:
    assert _first("가슴이 답답해요", answer) == "yes"


# ---- M-9 수용한 경계(바꾸지 않았음을 고정) --------------------------------------------------------------------------

def test_breathing_but_unresponsive_gets_no_cpr() -> None:
    # 숨을 쉬면 가슴압박 대상이 아니다. 119 안내는 응급 키워드로 나간다.
    q = "숨은 쉬는데 불러도 반응이 없어요"
    assert kw.emergency_signals(q) and not _cpr(q, hint=True)


def test_fallback_how_to_question_keeps_cpr_line() -> None:
    # 라우터 실패 때 "숨 안 쉬는 사람은 어떻게…"에는 가슴압박 줄을 붙인다(안전 쪽, 2026-11-09 사용자 결정과 같은 기준).
    assert kw.fallback_arrest_signal("숨 안 쉬는 사람은 어떻게 해야 해요?")
