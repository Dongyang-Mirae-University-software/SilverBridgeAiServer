"""#188 기능 점검(2026-11-09, docs/audits/2026-11-09-router-failure-followups-audit.md) 후속 고정.

C-1 고정 안내가 응급·위기·부상 문장을 막지 않는다 / H-1 조사 생략 꼴 가슴압박 줄 / H-2 "안 움직여요"는 의식 있음이 아니다 /
M-1 지난 일·회복 꼴은 가슴압박 줄 아님 / M-2 크레딧 소진 중 BURST 중복 생략 / L-1·L-2 실패 분류. 모든 문장은 합성 문장이다.
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta

import anthropic
import httpx
import pytest
from sqlalchemy import select

from app.models.chat_log import ChatLog
from app.services import chat_keywords as kw
from app.services import chat_service as cs
from app.services import router_failure_monitor as rfm
from app.services.chat_stats_service import ChatStatsService
from app.services.claude_structured_client import StructuredCallError
from app.services.medical_llm_service import MedicalLlmService
from tests.test_chat_router_failure_followups import _FailRouter, _Llm, _plain_reply, run  # noqa: F401 - run 은 픽스처


def _cpr(message: str, hint: bool) -> bool:
    return any(name == "cardiac_arrest" for name, _ in MedicalLlmService.emergency_action_sentences(message, arrest_hint=hint))


# ---- C-1 엉뚱한 질문 고정 안내 ----------------------------------------------------------------------------------

_NOT_OFF_TOPIC = [
    "드라마 보다 정신 잃음 어떡해?",
    "드라마보다 엄마 꼼짝안해 어떡해?",
    "드라마 보다 깜빡 졸았는데 엄마가 안 깨나 알려줘",
    "주식 다 날려서 다 끝내고 싶어 뭐야 이게",
    "코인 다 잃어서 사라지고 싶어 알려줘",
    "축구하다 발목 삐끗했는데 어케해?",
    "날씨땜시 무릅 시린디 어째야 쓰까?",
    "여행지 추천해줘 다리 불편한 엄마랑",
    "운세 알려줘 요즘 너무 지쳐",
    "야구 보다 기절했어 왜?",
    "게임하다 팔이 저려 왜 그래?",
    "영화 보다가 아빠가 반응이 없어 어떡해?",
]


@pytest.mark.parametrize("q", _NOT_OFF_TOPIC)
def test_off_topic_does_not_block_consciousness_crisis_injury(q: str) -> None:
    assert not kw.is_off_topic_question(q)


@pytest.mark.parametrize("q", ["주식 팔까?", "드라마 몇 시에 해?", "야구 몇대몇이야?", "맛집 추천해줘", "수학 숙제 좀 해줘"])
def test_off_topic_still_fixed_reply(q: str) -> None:
    assert kw.is_off_topic_question(q)


@pytest.mark.parametrize("q", _NOT_OFF_TOPIC[:5])
def test_service_failure_path_does_not_send_fixed_reply(run, q: str) -> None:
    out, ctx = run(q, _FailRouter(), _Llm())
    assert _plain_reply(out["reply"]) != cs._OFF_TOPIC_REPLY
    assert not ctx["upstreamMeta"].get("offTopic")


def _only_new_guards(monkeypatch) -> None:
    """건강 낱말·증상·응급 키워드 가드를 끄고 이번에 더한 가드(호칭·실패 경로 응급·반응 없음)만 남긴다."""
    monkeypatch.setattr(kw, "_OFF_TOPIC_HEALTH", kw.re.compile("(?!)"))
    monkeypatch.setattr(kw, "has_symptom_word", lambda text: False)
    monkeypatch.setattr(kw, "has_red_flag", lambda text: False)
    monkeypatch.setattr(kw, "emergency_signals", lambda text: [])


def test_off_topic_guard_uses_fallback_emergency_and_arrest(monkeypatch) -> None:
    # 주제 낱말 + 묻는 꼴인데 건강 낱말 목록에 없는 응급 표현 - 실패 경로 응급 판정·반응 없음이 막는다(점검 C-1 ②).
    _only_new_guards(monkeypatch)
    assert not kw.is_off_topic_question("야구 보다 갑자기 식은땀 나 왜?")
    assert not kw.is_off_topic_question("드라마 보다 숨을 안 쉬어 어떡해?")


def test_off_topic_person_guard(monkeypatch) -> None:
    # 점검 C-1 ①: 가족·옆 사람의 일을 묻는 말은 고정 안내로 보내지 않는다.
    _only_new_guards(monkeypatch)
    assert not kw.is_off_topic_question("드라마 보다가 할머니가 이상해 어떡해?")
    assert kw.is_off_topic_question("드라마 몇 시에 해?")


def test_off_topic_red_flag_guard(monkeypatch) -> None:
    # 점검 결함 주입 F6: has_red_flag 가드가 빠지면 잡혀야 한다.
    monkeypatch.setattr(kw, "has_red_flag", lambda text: True)
    assert not kw.is_off_topic_question("오늘 날씨 어때?")


def test_off_topic_turn_self_harm_guard(monkeypatch) -> None:
    # 점검 결함 주입 F2: 문장 판정이 틀려도 자해 신호 가드가 고정 안내를 막는다(겹침 방어).
    monkeypatch.setattr(kw, "is_off_topic_question", lambda text: True)
    route = {"route": "general", "emergency": False}
    assert not cs.ChatService._is_off_topic_turn(route, "주식 때문에 죽고 싶어 알려줘", None, False, False, False)


# ---- H-1·H-2·M-1 가슴압박 줄 -------------------------------------------------------------------------------------

_ARREST_FALLBACK_POSITIVE = [
    "아빠가 갑자기 쓰러졌는데 숨 안 쉬어요",
    "엄마가 쓰러졌는데 호흡 없어요",
    "아빠 쓰러짐 숨 안쉼",
    "할아버지 쓰러짐 숨 안 쉼 눈 안 뜸",
    "남편이 쓰러졌는데 숨 안 쉬는 것 같아요",  # 목격자 추측 - 붙인다(2026-11-09 사용자 결정)
    "할아버지가 쓰러졌는데 가슴이 안 움직여요",
    "숨소리가 안 들리고 가슴도 안 움직여요",
    "할아버지가 안 움직이시고 숨도 없음",
]


@pytest.mark.parametrize("q", _ARREST_FALLBACK_POSITIVE)
def test_fallback_arrest_dropped_particle_and_negated_move(q: str) -> None:
    assert kw.fallback_arrest_signal(q)
    assert cs.ChatService._keyword_route(q).get("fallback_arrest") is True


_ARREST_FALLBACK_NEGATIVE = [
    "숨이 안 쉬어져요",                   # 본인 호흡곤란 - 앉은 자세 줄
    "숨 안 쉬어져서 답답해요",
    "숨을 못 쉬겠어요",
    "숨이 없어질 것 같이 답답해요",        # 본인 느낌
    "예전에 흔들어도 꼼짝 안 하던 때가 있었는데 지금은 괜찮아요",
    "남편이 아침에 흔들어도 꼼짝 안 하다가 금방 깼어요",
    "쓰러지셨을 때 흔들어도 꼼짝 안 하던 게 지금은 숨 잘 쉬세요",
    "어제 계단 오르다 숨 없을 뻔했어요",
    "아버지가 의식이 없어요 숨은 쉬어요",
    "쓰러졌다가 지금은 움직여요",
    "예전에 숨 안 쉬던 적이 있었어요",     # 지난 일(옛 '지금 아님' 검사 - 결함 주입 F1)
    "아까 숨 안 쉬다가 다시 쉬었어요",
]


@pytest.mark.parametrize("q", _ARREST_FALLBACK_NEGATIVE)
def test_fallback_arrest_not_for_self_breathing_past_or_recovered(q: str) -> None:
    assert not kw.fallback_arrest_signal(q)


@pytest.mark.parametrize("q", [
    "할아버지가 쓰러졌는데 가슴이 안 움직여요",
    "엄마가 불러도 반응이 없고 몸이 안 움직여요",
    "아빠가 흔들어도 꼼짝도 안 하시고 전혀 안 움직이세요",
])
def test_negated_move_is_not_conscious_on_both_paths(q: str) -> None:
    # 점검 H-2: 라우터 성공(cardiac_arrest=true) 경로에서도 가슴압박 줄이 붙는다.
    assert not kw.arrest_conscious(kw.compact(q))
    assert _cpr(q, hint=True)


@pytest.mark.parametrize("q", [
    "쓰러지셨을 때 흔들어도 꼼짝 안 하던 게 지금은 숨 잘 쉬세요",
    "쓰러졌다가 지금은 멀쩡해요 숨을 안 쉬었었대요",
    "쓰러졌다가 지금은 움직여요",
])
def test_recovered_gets_no_cpr_even_with_router_hint(q: str) -> None:
    assert kw.arrest_conscious(kw.compact(q))
    assert not _cpr(q, hint=True)


def test_recovery_question_form_is_not_recovery() -> None:
    # "지금은 괜찮을까요"는 묻는 꼴 - 회복으로 읽지 않는다.
    q = "엄마가 쓰러졌는데 지금은 괜찮을까요 숨을 안 쉬어요"
    assert not kw.arrest_conscious(kw.compact(q)) and _cpr(q, hint=False)


# ---- M-2 크레딧 소진 중 BURST 중복 --------------------------------------------------------------------------------

class _T:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _errors(caplog, tag: str) -> int:
    return sum(1 for r in caplog.records if r.levelno == logging.ERROR and tag in r.getMessage())


def test_sustained_credit_exhaustion_logs_credit_only(caplog) -> None:
    clock = _T()
    monitor = rfm.RouterFailureMonitor(600, 5, 3600, clock=clock)
    with caplog.at_level(logging.WARNING):
        for _ in range(70):  # 70분 동안 1분마다 크레딧 실패
            monitor.record("api_400", "credit")
            clock.now += 60
    assert _errors(caplog, "[CHAT-ROUTER-CREDIT-EXHAUSTED]") == 2
    assert _errors(caplog, "[CHAT-ROUTER-FAILED-BURST]") == 0


def test_burst_still_alerts_when_other_kind_joins_credit_window(caplog) -> None:
    clock = _T()
    monitor = rfm.RouterFailureMonitor(600, 5, 3600, clock=clock)
    with caplog.at_level(logging.WARNING):
        for _ in range(6):
            monitor.record("api_400", "credit")
        monitor.record("api_529", "other")
        monitor.record("api_529", "other")
    assert _errors(caplog, "[CHAT-ROUTER-FAILED-BURST]") == 1


def test_burst_alerts_for_credit_when_not_in_credit_state(caplog) -> None:
    # 회복 뒤(record_success) 창에 남은 credit 실패는 BURST 로 알린다 - 소진 알림이 꺼진 상태다.
    clock = _T()
    monitor = rfm.RouterFailureMonitor(600, 5, 3600, clock=clock)
    with caplog.at_level(logging.WARNING):
        for _ in range(4):
            monitor.record("api_400", "credit")
        monitor.record_success()
        monitor.record("api_500", "other")
    assert _errors(caplog, "[CHAT-ROUTER-FAILED-BURST]") == 1


# ---- L-1·L-2 실패 분류 ---------------------------------------------------------------------------------------------

_REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _wrapped(cls, status: int, err_type: str, message: str) -> StructuredCallError:
    body = {"type": "error", "error": {"type": err_type, "message": message}}
    origin = cls(message="x", response=httpx.Response(status, request=_REQ, json=body), body=body)
    try:
        raise StructuredCallError(f"api_{status}") from origin
    except StructuredCallError as exc:
        return exc


def test_skipped_credit_call_counts_as_credit() -> None:
    assert rfm.failure_kind(StructuredCallError("credit_exhausted")) == "credit"


def test_billing_word_alone_is_not_credit() -> None:
    assert rfm.failure_kind(_wrapped(anthropic.BadRequestError, 400, "invalid_request_error",
                                     "model not available on your billing plan tier")) == "other"
    assert rfm.failure_kind(_wrapped(anthropic.PermissionDeniedError, 403, "permission_error",
                                     "Please visit Plans & Billing to enable access")) == "auth"
    assert rfm.failure_kind(_wrapped(anthropic.BadRequestError, 400, "invalid_request_error",
                                     "Your credit balance is too low to access the Anthropic API.")) == "credit"


# ---- 집계: 모르는 종류는 other (결함 주입 F11) -----------------------------------------------------------------------

def test_stats_unknown_failure_kind_counts_as_other(run) -> None:
    out, _ = run("머리가 아파요", _FailRouter(), _Llm(), user="k")
    row = run.db.execute(select(ChatLog).where(ChatLog.id == out["chatLogId"])).scalar_one()
    ctx = json.loads(row.context_json)
    ctx["upstreamMeta"]["routerFailureKind"] = "something_new"
    row.context_json = json.dumps(ctx, ensure_ascii=False)
    run.db.commit()
    today = date.today()
    stats = ChatStatsService.stats(run.db, today - timedelta(days=1), today + timedelta(days=1))
    assert stats["routerFailures"] == 1
    assert stats["routerFailuresByKind"] == {"credit": 0, "auth": 0, "timeout": 0, "other": 1}
