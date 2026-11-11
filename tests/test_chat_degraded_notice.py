"""라우터(Claude) 호출 실패 턴의 안내 한 줄(2026-11-11 사용자 승인).

- 실패 턴의 일반·상담 답 끝(layout 본문 끝)에 한 줄: "지금 일부 기능이 원활하지 않아 간단히 안내드려요. 잠시 뒤 다시 물어봐 주세요."
- 응급(119)·위기(109)·위기 뒤 해제·응급 다음 턴·위험 신호 "네" 끝맺음에는 붙이지 않는다.
- 예약 흐름이 글 입력을 읽지 못해 같은 칸을 버튼·달력으로 다시 물으면 버튼 안내로 바꾼다. 버튼 턴·정상 진행에는 없다.
- 되묻기 질문 턴·엉뚱한 질문 고정 안내·인사에는 붙이지 않는다. 라우터 정상 턴은 그대로. CHAT_DEGRADED_NOTICE=false 로 끈다.
실제 라우터·MedGemma·예약 서비스는 부르지 않는다."""
from __future__ import annotations

import logging
from typing import Any

import pytest

from app.schemas.chat_schema import UiOption, UiPayload
from app.services import chat_layout
from app.services import medical_llm_service as mls
from tests.reservation_flow_helpers import FUTURE, FakeReservation, make_orchestrator, make_request
from tests.test_audit2_chat_flows import (  # noqa: F401 - env 는 픽스처
    CRISIS_109, EMERGENCY, _ctx, _FailingLoader, _flow_state, _seed_flow, _send, env,
)
from tests.test_audit2_chat_flows import _FailRouter as _EnvFailRouter
from tests.test_audit2_chat_flows import _OkRouter as _EnvOkRouter
from tests.test_chat_router_failure_followups import _FailRouter, _Llm, _OkRouter, run  # noqa: F401 - run 은 픽스처

NOTICE = "지금 일부 기능이 원활하지 않아 간단히 안내드려요. 잠시 뒤 다시 물어봐 주세요."
INPUT_NOTICE = "지금은 글로 쓴 내용을 읽기 어려워요. 아래 버튼으로 골라 주세요."
NOTICE_LINES = ["지금 일부 기능이 원활하지 않아 간단히 안내드려요.", "잠시 뒤 다시 물어봐 주세요."]
INPUT_LINES = ["지금은 글로 쓴 내용을 읽기 어려워요.", "아래 버튼으로 골라 주세요."]
# 되묻지 않고 바로 상담 답(MedGemma)으로 가는 문장.
CONSULT = "어제부터 허리가 아프고 다리가 저려서 걷기가 힘들어요"


def _has(out: dict[str, Any], lines: list[str]) -> bool:
    return any(line in out["reply"] for line in lines)


def _no_notice(env_or_none: Any, out: dict[str, Any], ctx: dict[str, Any]) -> None:
    assert not _has(out, NOTICE_LINES) and not _has(out, INPUT_LINES)
    assert "degradedNotice" not in ctx["upstreamMeta"]
    for key in ("body", "homeCare", "visitHospitalIf", "questions"):
        assert not any(n in line for n in NOTICE_LINES + INPUT_LINES for line in (out.get("layout") or {}).get(key) or [])


# ---- 고정 문장 표·스위치 ---------------------------------------------------------------------------------------


def test_notice_sentences_are_in_the_fixed_sentence_table() -> None:
    names = {s.name: str(s) for s in mls.FIXED_SENTENCES}
    assert names["_DEGRADED_NOTICE"] == NOTICE and names["_DEGRADED_INPUT_NOTICE"] == INPUT_NOTICE
    assert mls.DEGRADED_NOTICES == {"router": NOTICE, "reservation-input": INPUT_NOTICE}


def test_switch_default_is_on(monkeypatch) -> None:
    from app.core import config

    monkeypatch.delenv("CHAT_DEGRADED_NOTICE", raising=False)
    assert config._as_bool("CHAT_DEGRADED_NOTICE", True) is True
    assert config.Settings().chat_degraded_notice is True
    monkeypatch.setenv("CHAT_DEGRADED_NOTICE", "false")
    assert config._as_bool("CHAT_DEGRADED_NOTICE", True) is False


# ---- 라우터 정상 / 실패 ----------------------------------------------------------------------------------------


def test_router_ok_turn_has_no_notice(run) -> None:
    llm = _Llm()
    out, ctx = run(CONSULT, _OkRouter({"route": "symptom"}), llm)
    assert len(llm.calls) == 1
    _no_notice(None, out, ctx)


def test_router_failure_turn_gets_the_notice_at_the_end_of_the_body(run, caplog) -> None:
    llm = _Llm()
    with caplog.at_level(logging.INFO):
        out, ctx = run(CONSULT, _FailRouter(), llm)
    assert len(llm.calls) == 1 and ctx["upstreamMeta"]["routerFailed"] is True
    assert ctx["upstreamMeta"]["degradedNotice"] == "router"
    layout = out["layout"]
    assert layout["body"][-2:] == NOTICE_LINES and "urgent" not in layout
    assert out["reply"] == chat_layout.flatten(layout)  # 화면(layout)과 글(reply)이 같은 문장·같은 자리
    assert [out["reply"].count(line) for line in NOTICE_LINES] == [1, 1]
    assert all(line.count(".") <= 1 for line in out["reply"].split("\n") if line in NOTICE_LINES)  # 한 줄 한 문장
    logs = [r.getMessage() for r in caplog.records if "degraded_notice" in r.getMessage()]
    assert logs == ["[CHAT-QUALITY] degraded_notice=router"]  # 이름·값만, 본문 없음
    assert NOTICE not in str(ctx["upstreamMeta"])  # 기록 meta 에는 종류만


def test_notice_survives_when_generation_also_fails(run) -> None:
    out, ctx = run(CONSULT, _FailRouter(), _Llm(fail=True))
    assert ctx["upstreamMeta"]["generationFailed"] is True and ctx["upstreamMeta"]["degradedNotice"] == "router"
    assert out["reply"] == chat_layout.flatten(out["layout"]) and out["layout"]["body"][-2:] == NOTICE_LINES


def test_switch_off_keeps_the_old_answer(env) -> None:
    env.router = _EnvFailRouter()
    env.svc._settings.chat_degraded_notice = False
    out = _send(env, CONSULT)
    ctx = _ctx(env, out)
    assert ctx["upstreamMeta"]["routerFailed"] is True and len(env.llm.calls) == 1
    _no_notice(env, out, ctx)


def test_emoji_on_keeps_one_emoji_per_notice_line(env) -> None:
    env.router = _EnvFailRouter()
    env.svc._settings.chat_layout_emoji = True
    out = _send(env, CONSULT, role=None)
    tail = out["layout"]["body"][-2:]
    assert [chat_layout.kw.strip_layout_emoji(x).strip() for x in tail] == NOTICE_LINES
    assert all(x != plain and x.endswith(plain) for x, plain in zip(tail, NOTICE_LINES))
    assert out["reply"] == chat_layout.flatten(out["layout"])


# ---- 응급·위기에는 붙이지 않는다 ---------------------------------------------------------------------------------


def test_router_failure_emergency_answer_has_no_notice(env) -> None:
    env.router = _EnvFailRouter()
    out = _send(env, EMERGENCY)
    ctx = _ctx(env, out)
    assert out["riskLevel"] == "high" and ctx["upstreamMeta"]["routerFailed"] is True
    _no_notice(env, out, ctx)


def test_router_failure_self_harm_answer_has_no_notice(env) -> None:
    env.router = _EnvFailRouter()
    out = _send(env, "다 끝내고 싶어")  # 라우터를 부르고 실패한 뒤 좁은 패턴으로 위기 응답
    ctx = _ctx(env, out)
    assert ctx["upstreamMeta"]["selfHarm"] is True and ctx["upstreamMeta"]["routerFailed"] is True
    _no_notice(env, out, ctx)


def test_router_failure_turn_after_crisis_keeps_only_the_109_line(env) -> None:
    _send(env, "죽고 싶어요")
    env.router = _EnvFailRouter()
    out = _send(env, CONSULT)
    ctx = _ctx(env, out)
    assert ctx["upstreamMeta"]["crisisFollowup"] is True and ctx["upstreamMeta"]["routerFailed"] is True
    assert "109" in out["reply"]
    _no_notice(env, out, ctx)


def test_router_failure_turn_after_emergency_has_no_notice(env) -> None:
    env.router = _EnvFailRouter()
    assert _send(env, EMERGENCY)["riskLevel"] == "high"
    out = _send(env, "이제 좀 괜찮아졌어요")
    ctx = _ctx(env, out)
    assert ctx["upstreamMeta"]["routerFailed"] is True
    _no_notice(env, out, ctx)


def test_router_failure_red_flag_yes_ending_has_no_notice(env) -> None:
    env.router = _EnvFailRouter()
    first = _send(env, "기침이 나요")
    ctx = _ctx(env, first)
    assert ctx["upstreamMeta"].get("followupAsked") is True
    _no_notice(env, first, ctx)  # 되묻기 질문 턴: 질문 하나만 보이는 화면
    out = _send(env, "네")
    ctx = _ctx(env, out)
    assert ctx["upstreamMeta"].get("followupRedFlag") in {"119", "visit"} and ctx["upstreamMeta"]["routerFailed"] is True
    _no_notice(env, out, ctx)


# ---- 되묻기·고정 안내·인사 --------------------------------------------------------------------------------------


@pytest.mark.parametrize("message", ["오늘 날씨 어때요?", "고마워요"])
def test_router_failure_off_topic_and_closing_words_have_no_notice(run, message: str) -> None:
    out, ctx = run(message, _FailRouter(), _Llm())
    assert ctx["upstreamMeta"]["routerFailed"] is True
    _no_notice(None, out, ctx)


# ---- 피보호자 간결 답변 -----------------------------------------------------------------------------------------


class _LongLlm:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.histories: list[list[dict[str, str]]] = []

    def generate(self, message, ctx, history, intent=None, **kwargs):  # noqa: ARG002
        self.histories.append(list(history))
        return {"reply": self.reply, "riskLevel": "medium", "recommendedAction": "guardian_contact", "reservationRequired": False,
                "intent": intent, "engine": "medgemma", "modelName": "m", "summary": None, "possibleCauses": [], "homeCare": [],
                "visitHospitalIf": [], "emergencyWarning": []}


LONG = " ".join(f"허리를 따뜻하게 하고 무리한 동작은 피하면서 충분히 쉬어 주시는 것이 좋습니다 {i}번." for i in range(8))


def test_ward_concise_answer_does_not_cut_the_notice(env) -> None:
    env.router = _EnvFailRouter()
    env.llm = _LongLlm(LONG)
    out = _send(env, CONSULT, role="WARD")
    ctx = _ctx(env, out)
    assert ctx["upstreamMeta"]["wardConcise"]["dropped"] > 0  # 모델 문장은 줄었다
    assert out["layout"]["body"][-2:] == NOTICE_LINES and ctx["upstreamMeta"]["degradedNotice"] == "router"
    assert out["reply"] == chat_layout.flatten(out["layout"])


def test_notice_is_not_repeated_and_is_kept_out_of_the_next_history(env) -> None:
    env.router = _EnvFailRouter()
    env.llm = _LongLlm(f"따뜻하게 찜질해 보세요. {NOTICE}")  # 모델이 같은 문장을 써도 한 번만
    out = _send(env, CONSULT, role=None)
    assert [out["reply"].count(line) for line in NOTICE_LINES] == [1, 1]
    env.router = _EnvOkRouter()
    nxt = _send(env, "찜질은 하루에 몇 번 하면 좋을까요 궁금해요", role=None)
    said = " ".join(item["content"] for item in env.llm.histories[-1] if item["role"] == "assistant")
    assert "찜질" in said and not any(line in said for line in NOTICE_LINES)
    assert "degradedNotice" not in _ctx(env, nxt)["upstreamMeta"]


# ---- 예약 흐름 --------------------------------------------------------------------------------------------------


class _UiOrch:
    """다음 턴에 묻는 칸(ui)을 정해 두는 예약 오케스트레이터 자리."""

    def __init__(self, ui: UiPayload | None) -> None:
        self.ui = ui
        self.calls: list[str] = []

    def handle(self, req, meta):
        self.calls.append(req.message)
        return {"reply": "원하시는 시간을 골라 주세요.", "riskLevel": "medium", "recommendedAction": "guardian_contact",
                "reservationRequired": True, "intent": "hospital_reservation", "engine": "claude", "modelName": "fake",
                "type": "message", "ui": self.ui, "tool": None, "data": None, "meta": {**meta, "flowState": _flow_state()}}


def _time_ui(kind: str = "select", field: str = "time") -> UiPayload:
    options = [UiOption(value="10:00", label="오전 10시")] if kind == "select" else []
    return UiPayload(kind=kind, field=field, label="시간", options=options)


def test_reservation_typed_value_not_read_gets_the_button_notice(env) -> None:
    _seed_flow(env, ui_field="time")
    env.router = _EnvFailRouter()
    env.orch = _UiOrch(_time_ui())
    out = _send(env, "열한 시로 해줘")
    ctx = _ctx(env, out)
    assert env.orch.calls == ["병원 예약: 내과 / 열한 시로 해줘"]
    assert out["reply"] == "원하시는 시간을 골라 주세요.\n" + "\n".join(INPUT_LINES)
    assert out["layout"] is None and out["ui"].field == "time"  # 예약 답 모양은 그대로(버튼이 안내 바로 아래)
    assert ctx["upstreamMeta"]["degradedNotice"] == "reservation-input" and not _has(out, NOTICE_LINES)


@pytest.mark.parametrize("ui", [_time_ui(field="date", kind="date"), _time_ui(kind="text"), None],
                         ids=["moved-on", "text-input-step", "no-ui"])
def test_reservation_turn_that_moved_on_or_has_no_buttons_has_no_notice(env, ui) -> None:
    _seed_flow(env, ui_field="time")
    env.router = _EnvFailRouter()
    env.orch = _UiOrch(ui)
    out = _send(env, "열한 시로 해줘")
    ctx = _ctx(env, out)
    assert ctx["upstreamMeta"]["routerFailed"] is True and len(env.orch.calls) == 1
    _no_notice(env, out, ctx)


def test_reservation_button_turn_has_no_notice(env) -> None:
    _seed_flow(env, ui_field="time")
    env.router = _EnvFailRouter()
    env.orch = _UiOrch(_time_ui())
    out = _send(env, ui=("time", "10:00"))
    assert len(env.orch.calls) == 1
    _no_notice(env, out, _ctx(env, out))


def test_reservation_same_step_with_router_ok_has_no_notice(env) -> None:
    _seed_flow(env, ui_field="time")
    env.router = _EnvOkRouter(route="reservation", normalized_message="병원 예약: 내과")
    env.orch = _UiOrch(_time_ui())
    out = _send(env, "열한 시로 해줘")
    assert len(env.orch.calls) == 1
    _no_notice(env, out, _ctx(env, out))


def test_new_reservation_request_on_a_failed_router_turn_has_no_notice(env) -> None:
    env.router = _EnvFailRouter()
    env.orch = _UiOrch(_time_ui(field="department"))
    out = _send(env, "내과 병원 예약하고 싶어요")
    assert len(env.orch.calls) == 1
    _no_notice(env, out, _ctx(env, out))


def test_orchestrator_asks_the_time_step_again_for_korean_number_time() -> None:
    """글로 쓴 한글 숫자 시간("열한 시")은 읽지 않는다 - 같은 시간 칸을 버튼으로 다시 묻는다(버튼 안내가 붙는 근거)."""
    fake = FakeReservation()
    orch = make_orchestrator(fake)
    orch._loader = _FailingLoader()
    state = _flow_state()
    state["draft"].update({"hospital_id": 1, "hospital_name": "강남메디내과", "location": "서울 강남구", "reservation_date": FUTURE})
    out = orch.handle(make_request("병원 예약: 내과 / 열한 시로 해줘"),
                      {"storedFlow": {"flowState": state}, "rawMessage": "열한 시로 해줘", "awaitingField": "time"})
    assert out is not None and out["ui"] is not None
    assert (out["ui"].field, out["ui"].kind) == ("time", "select")
    assert not out["meta"]["flowState"]["draft"].get("reservation_time")
