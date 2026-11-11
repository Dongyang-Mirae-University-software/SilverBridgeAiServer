"""라우터 실패 보완 3건(2026-11-08, #187 점검 L-1 후속).

A. 라우터 실패 누적 알림 - 창 안에서 기준 횟수에 닿으면 ERROR 한 번(창마다 1회), 원인 코드만. 기록 meta.routerFailed + 집계 routerFailures.
B. 라우터 실패 경로의 심정지 가슴압박 줄 - 키워드 판정(fallback_arrest)이 라우터 cardiac_arrest 칸과 같은 줄을 붙인다. 일반 221건 오탐 0.
C. 의료·예약과 무관한 질문 고정 안내 - MedGemma·이어서 물어볼 질문 모델을 부르지 않는다. 애매하면(건강 낱말·응급) 상담 그대로.
"""
from __future__ import annotations

import glob
import json
import logging
import pathlib
import re
import tempfile
from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.models.chat_log import ChatLog
from app.schemas.chat_schema import ChatContext, ChatRequest
from app.services import chat_keywords as kw
from app.services import chat_service as cs
from app.services import follow_up_questions
from app.services import router_failure_monitor as rfm
from app.services.chat_stats_service import ChatStatsService
from app.services.claude_structured_client import StructuredCallError

def _plain_reply(reply: str) -> str:
    """chat_layout 이 붙인 줄머리 그림 문자·줄바꿈을 걷어 문장만 비교한다."""
    return " ".join(line.lstrip("🏥💡🚨 ").strip() for line in reply.split("\n"))


CPR = re.compile(r"가슴압박\(가슴 한가운데를 세게, 빠르게")
_EVAL = pathlib.Path(__file__).parent / "eval"
SECRET = "sk-ant-SECRET-key"
BODY = "할머니를 아무리 깨워도 안 일어나세요"


def _unique_items():
    seen: set[str] = set()
    for f in sorted(glob.glob(str(_EVAL / "*.json"))):
        if pathlib.Path(f).name.startswith("chip_set"):
            continue  # 칩 시험지(2026-11-11)는 앞 턴 다음에 누르는 짧은 칩 문장이라 일반·응급 문항 수에 넣지 않는다
        for it in json.loads(pathlib.Path(f).read_text(encoding="utf-8")).get("items", []):
            q = it.get("q") or ""
            if q and q not in seen:
                seen.add(q)
                yield it


# ---- A. 누적 알림 ---------------------------------------------------------------------------------------------

class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _bursts(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if "[CHAT-ROUTER-FAILED-BURST]" in r.getMessage()]


def test_four_failures_do_not_alert_fifth_alerts_once(caplog) -> None:
    clock = _Clock()
    m = rfm.RouterFailureMonitor(600, 5, clock=clock)
    with caplog.at_level(logging.INFO):
        assert [m.record("api_400") for _ in range(4)] == [False] * 4
        assert _bursts(caplog) == []
        assert m.record("parse") is True
        assert [m.record("api_400") for _ in range(10)] == [False] * 10  # 같은 창에서는 다시 남기지 않는다
    recs = _bursts(caplog)
    assert len(recs) == 1 and recs[0].levelno == logging.ERROR
    assert "count=5" in recs[0].getMessage() and "api_400:4" in recs[0].getMessage() and "parse:1" in recs[0].getMessage()


def test_window_resets_after_it_passes(caplog) -> None:
    clock = _Clock()
    m = rfm.RouterFailureMonitor(600, 5, clock=clock)
    with caplog.at_level(logging.INFO):
        for _ in range(4):
            m.record("api_400")
        clock.t += 600  # 창이 지났다 - 앞의 4건은 버리고 새로 센다
        assert [m.record("api_400") for _ in range(4)] == [False] * 4
        assert m.record("api_400") is True
        clock.t += 599
        assert m.record("api_400") is False  # 같은 창
        clock.t += 1
        assert [m.record("x") for _ in range(5)] == [False] * 4 + [True]  # 다음 창에서 다시 1회
    assert len(_bursts(caplog)) == 2


def test_alert_log_has_only_codes(caplog) -> None:
    m = rfm.RouterFailureMonitor(600, 1, clock=_Clock())
    with caplog.at_level(logging.INFO):
        m.record(f"{BODY} {SECRET}\n가짜 줄")
    msg = _bursts(caplog)[0].getMessage()
    assert SECRET not in msg and "깨워도" not in msg and "\n" not in msg and "causes=other/other:1" in msg


def test_settings_defaults() -> None:
    from app.core.config import Settings

    s = Settings()
    assert (s.chat_router_failure_alert_window_sec, s.chat_router_failure_alert_threshold) == (600, 5)


# ---- 서비스 경로 공통 -------------------------------------------------------------------------------------------

class _FailRouter:
    calls = 0

    def route(self, message, history, ctx, prev_reservation=None) -> dict[str, Any]:  # noqa: ARG002
        _FailRouter.calls += 1
        raise StructuredCallError("api_400")


class _OkRouter:
    def __init__(self, out: dict[str, Any]) -> None:
        self.out = out

    def route(self, message, history, ctx, prev_reservation=None) -> dict[str, Any]:  # noqa: ARG002
        return {"normalized_message": message, "symptom_summary": "", "emergency": False, "self_harm": False,
                "cardiac_arrest": False, "urgent_condition": "", "reason": "t", "engine": "fake", **self.out}


class _Llm:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def generate(self, message, ctx, history, **kwargs):  # noqa: ARG002
        self.calls.append(kwargs)
        if self.fail:
            raise TimeoutError("busy")
        return {"reply": "모델 답", "riskLevel": "low", "recommendedAction": "home_monitoring",
                "reservationRequired": False, "intent": kwargs.get("intent") or "general_chat", "engine": "medgemma",
                "modelName": "m", "summary": None, "possibleCauses": [], "homeCare": [], "visitHospitalIf": [],
                "emergencyWarning": []}


@pytest.fixture()
def run(monkeypatch):
    engine = create_engine(f"sqlite:///{tempfile.mkdtemp(prefix='sb_rf_')}/t.db")
    ChatLog.__table__.create(bind=engine)
    db = sessionmaker(bind=engine)()
    monitor = rfm.RouterFailureMonitor(600, 5, clock=_Clock())
    monkeypatch.setattr(cs, "get_router_failure_monitor", lambda: monitor)
    monkeypatch.setattr(cs, "get_reservation_credential_service", lambda: object())
    monkeypatch.setattr(cs, "get_reservation_api_client", lambda: object())
    suggest_calls: list[str] = []
    monkeypatch.setattr(follow_up_questions, "_model_questions", lambda *a, **k: suggest_calls.append("x") or [])

    def _run(message: str, router: Any, llm: _Llm, user: str = "u1") -> tuple[dict, dict]:
        monkeypatch.setattr(cs, "get_chat_router_service", lambda: router)
        monkeypatch.setattr(cs, "get_medical_llm_service", lambda: llm)
        svc = cs.ChatService()
        svc._settings = SimpleNamespace(has_reservation_account_secret=True, claude_model_name="fake",
                                        follow_up_enabled=True, follow_up_timeout_sec=2)
        out = svc.process_message(db, ChatRequest(userId=user, message=message, sessionId=f"s-{message[:4]}",
                                                  context=ChatContext()))
        row = db.execute(select(ChatLog).where(ChatLog.id == out["chatLogId"])).scalar_one()
        return out, json.loads(row.context_json)

    _run.db = db
    _run.monitor = monitor
    _run.suggest_calls = suggest_calls
    yield _run
    db.close()


def test_router_failure_marks_meta_and_feeds_monitor(run, caplog) -> None:
    with caplog.at_level(logging.INFO):
        for i in range(5):
            _, ctx = run(f"머리가 아파요 {i}", _FailRouter(), _Llm(), user=f"u{i}")
            assert ctx["upstreamMeta"]["routerFailed"] is True
            assert ctx["upstreamMeta"]["router"]["engine"] == "keyword"
    assert len(_bursts(caplog)) == 1
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "머리가" not in text.split("[CHAT-ROUTER-FAILED-BURST]")[1].split("\n")[0]


def test_router_success_has_no_router_failed(run) -> None:
    _, ctx = run("머리가 아파요", _OkRouter({"route": "symptom"}), _Llm())
    assert "routerFailed" not in ctx["upstreamMeta"]


def test_stats_counts_router_failures(run) -> None:
    run("머리가 아파요", _FailRouter(), _Llm(), user="a")
    run("배가 아파요", _FailRouter(), _Llm(), user="b")
    run("허리가 아파요", _OkRouter({"route": "symptom"}), _Llm(), user="c")
    today = date.today()
    # 테스트 날짜가 KST 경계에 걸려도 세 건이 같은 범위 안에 들도록 앞뒤 하루를 넉넉히 본다.
    out = ChatStatsService.stats(run.db, today - timedelta(days=1), today + timedelta(days=1))
    assert out["routerFailures"] == 2 and out["totalChats"] == 3


# ---- B. 실패 경로 가슴압박 줄 -----------------------------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "할머니를 아무리 깨워도 안 일어나세요",
    "엄마 숨소리가 전혀 안 들려요",
    "아버지가 가슴이 오르락내리락을 안 해요",
    "남편이 갑자기 축 늘어져서 미동도 없어요",
    "경로당 회원분이 쓰러짐 눈안뜸 숨없음",
    "아버지가 쓰러지셨는데 흔들어도 꼼짝을 안하세요",
])
def test_fallback_arrest_signal_positive(q: str) -> None:
    assert kw.fallback_arrest_signal(q)
    route = cs.ChatService._keyword_route(q)
    assert route["fallback_arrest"] is True
    assert cs.ChatService._router_hints(route).get("arrest_hint") is True


@pytest.mark.parametrize("q", [
    "깨워도 안 일어나셨는데 지금은 깨셨어요",
    "흔들어도 꼼짝 안 하더니 지금은 말은 하세요",
    "요즘 몸이 축 처지고 기운이 없어요",
    "숨이 없을까봐 걱정돼요",
    "숨이 차요",
    "할아버지 입술이 퍼렇게 변했어요",  # 청색증만으로는 심정지 줄 아님(119·응급은 그대로)
    "넘어지면 못 일어나실까봐 걱정이에요",
    "숨 쉴 틈이 없이 바빠요",
])
def test_fallback_arrest_signal_negative(q: str) -> None:
    assert not kw.fallback_arrest_signal(q)
    assert "fallback_arrest" not in cs.ChatService._keyword_route(q)


def test_eval_general_221_no_arrest_false_positive() -> None:
    general = [it for it in _unique_items() if (it.get("expect") or {}).get("emergency") is False]
    assert len(general) == 309  # 2026-11-11 holdout v6(일반 증상 88문항) 추가 전 221
    assert [it["id"] for it in general if kw.fallback_arrest_signal(it["q"])] == []
    not_cpr = [it["id"] for it in _unique_items()
               if (it.get("expect") or {}).get("cpr_aed") is False and kw.fallback_arrest_signal(it["q"])]
    assert not_cpr == []


def test_eval_cpr_items_get_cpr_line_without_router() -> None:
    cpr = [it for it in _unique_items() if (it.get("expect") or {}).get("cpr_aed") is True]
    got = [it["id"] for it in cpr
           if kw.ARREST_SIGNAL.search(kw.compact(it["q"])) or kw.fallback_arrest_signal(it["q"])]
    assert len(cpr) == 18 and len(got) >= 17, sorted({it["id"] for it in cpr} - set(got))


def test_service_failure_path_sends_cpr_line_via_generation_hint(run) -> None:
    llm = _Llm()
    _, ctx = run(BODY, _FailRouter(), llm)
    assert llm.calls and llm.calls[0].get("arrest_hint") is True
    assert ctx["upstreamMeta"]["router"]["fallback_arrest"] is True
    assert ctx["upstreamMeta"]["router"]["cardiac_arrest"] is None


def test_service_failure_path_busy_generation_keeps_cpr_line(run) -> None:
    out, _ = run(BODY, _FailRouter(), _Llm(fail=True))
    assert out["riskLevel"] == "high" and CPR.search(out["reply"]) and "AED" in out["reply"]


def test_router_success_path_ignores_fallback_arrest(run) -> None:
    llm = _Llm()
    _, ctx = run(BODY, _OkRouter({"route": "symptom", "emergency": True, "cardiac_arrest": False}), llm)
    assert "arrest_hint" not in llm.calls[0]
    assert ctx["upstreamMeta"]["router"]["fallback_arrest"] is None


# ---- C. 엉뚱한 질문 고정 안내 -----------------------------------------------------------------------------------

@pytest.mark.parametrize("q", [
    "오늘 날씨 어때?", "삼성전자 주식 살까요", "야구 결과 알려줘", "김치찌개 레시피 알려줘",
    "영어로 번역해줘 사랑해", "로또 번호 추천해줘", "오늘 운세 봐줘", "파이썬 코딩 알려줘",
])
def test_off_topic_positive(q: str) -> None:
    assert kw.is_off_topic_question(q)


@pytest.mark.parametrize("q", [
    "오늘 날씨가 참 좋네요",            # 맞장구 - 묻는 꼴 아님
    "날씨가 추운데 산책해도 될까요",      # 생활 관리
    "당뇨인데 김치찌개 먹어도 돼?",
    "날씨 때문에 무릎이 쑤셔요",
    "드라마 보다가 가슴이 아파요",
    "야구 보다가 쓰러지셨어요 어떡해",     # 응급
    "날씨 알려줘 근데 숨이 차요",
    "주식 때문에 죽고 싶어요",           # 자해
    "로또 사러 가다가 넘어졌는데 괜찮을까요",
    "병원 예약하고 날씨도 알려줘",
    "안녕하세요", "고마워요",
])
def test_off_topic_negative(q: str) -> None:
    assert not kw.is_off_topic_question(q)


def test_eval_sets_have_no_off_topic_hits() -> None:
    assert [it["id"] for it in _unique_items() if kw.is_off_topic_question(it["q"])] == []


def test_service_off_topic_fixed_reply_no_model_calls(run) -> None:
    llm = _Llm()
    out, ctx = run("오늘 날씨 어때?", _OkRouter({"route": "general"}), llm)
    assert _plain_reply(out["reply"]) == cs._OFF_TOPIC_REPLY and out["engine"] == "rule" and out["riskLevel"] == "low"
    assert llm.calls == [] and out["suggestedQuestions"] == [] and run.suggest_calls == []
    assert ctx["upstreamMeta"]["offTopic"] is True


def test_service_off_topic_also_on_router_failure(run) -> None:
    llm = _Llm()
    out, ctx = run("로또 번호 추천해줘", _FailRouter(), llm)
    assert _plain_reply(out["reply"]) == cs._OFF_TOPIC_REPLY and llm.calls == []
    assert ctx["upstreamMeta"]["routerFailed"] is True


def test_service_off_topic_text_with_emergency_goes_medical(run) -> None:
    llm = _Llm()
    out, ctx = run("야구 보다가 아버지가 쓰러지셨어요 어떡해", _FailRouter(), llm)
    assert _plain_reply(out["reply"]) != cs._OFF_TOPIC_REPLY and llm.calls
    assert ctx["upstreamMeta"]["router"]["emergency"] is True and "offTopic" not in ctx["upstreamMeta"]


def test_service_router_symptom_route_not_overridden(run) -> None:
    llm = _Llm()
    out, _ = run("오늘 날씨 어때?", _OkRouter({"route": "symptom"}), llm)
    assert _plain_reply(out["reply"]) != cs._OFF_TOPIC_REPLY and llm.calls




@pytest.mark.parametrize("route, kwargs, expected", [
    ({"route": "general"}, {}, True),
    ({"route": "general", "emergency": True}, {}, False),
    ({"route": "general", "self_harm": True}, {}, False),
    ({"route": "symptom"}, {}, False),
    ({"route": "general"}, {"crisis_followup": True}, False),
    ({"route": "general"}, {"after_emergency": True}, False),
    ({"route": "general"}, {"after_followup": True}, False),
])
def test_off_topic_turn_guards(route, kwargs, expected) -> None:
    args = {"crisis_followup": False, "after_emergency": False, "after_followup": False, **kwargs}
    assert cs.ChatService._is_off_topic_turn(route, "오늘 날씨 어때?", None, **args) is expected


# ---- A-2. 크레딧 소진 구분(2026-11-08 추가 지시) - 목 예외만(실제 호출 없음) ------------------------------------------

import anthropic  # noqa: E402
import httpx  # noqa: E402


def _status_error(cls, status: int, err_type: str | None, message: str = "x", code: str | None = None):
    error: dict[str, Any] = {"message": message}
    if err_type:
        error["type"] = err_type
    if code:
        error["details"] = {"error_code": code}
    r = httpx.Response(status, request=httpx.Request("POST", "https://api.example/v1/messages"))
    return cls("m", response=r, body={"type": "error", "error": error})


def _wrapped(origin: BaseException, cause: str) -> StructuredCallError:
    try:
        try:
            raise origin
        except BaseException as inner:
            raise StructuredCallError(cause) from inner
    except StructuredCallError as outer:
        return outer


@pytest.mark.parametrize("origin, cause, kind", [
    # 11/04 실제 형태: 400 invalid_request_error + "credit balance is too low"
    (_status_error(anthropic.BadRequestError, 400, "invalid_request_error",
                   "Your credit balance is too low to access the Anthropic API."), "api_400", "credit"),
    (_status_error(anthropic.APIStatusError, 402, "billing_error"), "api_402", "credit"),
    (_status_error(anthropic.PermissionDeniedError, 403, "permission_error", code="billing_frozen"), "api_403", "credit"),
    (_status_error(anthropic.BadRequestError, 400, "invalid_request_error", code="enforced_spend_limit_reached"), "api_400", "credit"),
    (_status_error(anthropic.AuthenticationError, 401, "authentication_error", "invalid x-api-key"), "api_401", "auth"),
    (_status_error(anthropic.PermissionDeniedError, 403, "permission_error"), "api_403", "auth"),
    (_status_error(anthropic.BadRequestError, 400, "invalid_request_error", "max_tokens too large"), "api_400", "other"),
    (_status_error(anthropic.InternalServerError, 504, "timeout_error"), "api_504", "timeout"),
    (anthropic.APITimeoutError(request=httpx.Request("POST", "https://api.example")), "connection", "timeout"),
    (anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.example")), "connection", "other"),
    (_status_error(anthropic.RateLimitError, 429, "rate_limit_error"), "rate_limited", "other"),
])
def test_failure_kind_classifies_sdk_errors(origin, cause, kind) -> None:
    assert rfm.failure_kind(_wrapped(origin, cause)) == kind


def test_failure_kind_fixed_codes() -> None:
    assert rfm.failure_kind(StructuredCallError("not_ready")) == "off"
    assert rfm.failure_kind(StructuredCallError("deadline")) == "timeout"
    assert rfm.failure_kind(StructuredCallError("parse")) == "other"
    assert rfm.failure_kind(ValueError("boom")) == "other"


def _credit_logs(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "[CHAT-ROUTER-CREDIT-EXHAUSTED]" in r.getMessage()]


def test_credit_first_failure_alerts_immediately_then_every_interval(caplog) -> None:
    clock = _Clock()
    m = rfm.RouterFailureMonitor(600, 5, credit_repeat_sec=3600, clock=clock)
    with caplog.at_level(logging.INFO):
        assert m.record("api_400", "credit") is True  # 1건째에 바로
        assert m.record("api_400", "credit") is False
        clock.t += 3599
        m.record("api_400", "credit")
        assert len(_credit_logs(caplog)) == 1
        clock.t += 1
        m.record("api_400", "credit")
    logs = _credit_logs(caplog)
    assert len(logs) == 2 and "alert=first" in logs[0] and "alert=repeat" in logs[1]
    assert all(r.levelno == logging.ERROR for r in caplog.records if "CREDIT-EXHAUSTED" in r.getMessage())
    assert "credit balance" not in "\n".join(logs)


def test_credit_recovers_on_success_and_alerts_again_next_time(caplog) -> None:
    clock = _Clock()
    m = rfm.RouterFailureMonitor(600, 5, clock=clock)
    with caplog.at_level(logging.INFO):
        m.record("api_400", "credit")
        assert m.status(True)["state"] == "credit_exhausted"
        m.record_success()
        assert m.status(True)["state"] == "failing"  # 지금 창의 실패 1회는 남는다
        clock.t += 600
        assert m.status(True) == {"state": "ok", "recentFailures": 0, "windowSec": 600}
        m.record("api_400", "credit")  # 다시 떨어지면 다시 바로 알린다
    assert len(_credit_logs(caplog)) == 2
    assert any("[CHAT-ROUTER-CREDIT-RECOVERED]" in r.getMessage() for r in caplog.records)


def test_status_states() -> None:
    m = rfm.RouterFailureMonitor(600, 5, clock=_Clock())
    assert m.status(True)["state"] == "ok"
    m.record("api_500", "other")
    m.record("connection", "timeout")
    assert m.status(True) == {"state": "failing", "recentFailures": 2, "windowSec": 600}
    assert m.status(False) == {"state": "off", "recentFailures": 0, "windowSec": 600}


def test_off_is_not_counted(caplog) -> None:
    m = rfm.RouterFailureMonitor(600, 1, clock=_Clock())
    with caplog.at_level(logging.INFO):
        assert m.record("not_ready", "off") is False
    assert m.status(True)["state"] == "ok" and _bursts(caplog) == []


class _NoKeyRouter:
    def route(self, *a, **k):  # noqa: ARG002
        raise StructuredCallError("not_ready")


class _CreditRouter:
    def route(self, *a, **k):  # noqa: ARG002
        raise _wrapped(_status_error(anthropic.BadRequestError, 400, "invalid_request_error",
                                     "Your credit balance is too low to access the Anthropic API."), "api_400")


def test_service_no_key_is_not_router_failure(run) -> None:
    _, ctx = run("머리가 아파요", _NoKeyRouter(), _Llm())
    assert "routerFailed" not in ctx["upstreamMeta"] and run.monitor.status(True)["state"] == "ok"


def test_service_credit_failure_alerts_and_records_kind(run, caplog) -> None:
    with caplog.at_level(logging.INFO):
        _, ctx = run("머리가 아파요", _CreditRouter(), _Llm())
    assert ctx["upstreamMeta"]["routerFailed"] is True and ctx["upstreamMeta"]["routerFailureKind"] == "credit"
    assert len(_credit_logs(caplog)) == 1 and run.monitor.status(True)["state"] == "credit_exhausted"
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "credit balance" not in joined and "머리가" not in joined
    run("배가 아파요", _OkRouter({"route": "symptom"}), _Llm(), user="u2")
    assert run.monitor.status(True)["state"] == "failing"  # 성공하면 소진 상태가 풀린다


def test_stats_router_failures_by_kind_and_status(run) -> None:
    run("머리가 아파요", _CreditRouter(), _Llm(), user="a")
    run("배가 아파요", _FailRouter(), _Llm(), user="b")
    run("허리가 아파요", _NoKeyRouter(), _Llm(), user="c")
    today = date.today()
    out = ChatStatsService.stats(run.db, today - timedelta(days=1), today + timedelta(days=1),
                                 router_status=run.monitor.status(True))
    assert out["routerFailures"] == 2
    assert out["routerFailuresByKind"] == {"credit": 1, "auth": 0, "timeout": 0, "other": 1}
    assert out["routerStatus"]["state"] == "credit_exhausted" and out["routerStatus"]["recentFailures"] == 2
