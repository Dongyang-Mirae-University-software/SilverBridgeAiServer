"""챗봇 개인정보 최소화(S-6·S-8)와 요청 전체 마감(O-3) - 2026-10-28.

- Claude(라우터·예약 추출)로 나가는 입력에 이름·전화·이메일·상세주소가 없고 주소는 시·구까지다.
- 병원에 넘기는 증상 요약은 이번 턴 사용자 문장의 증상만, 개인정보 형태 제거, 60자, 없으면 빈 값.
- 요청 하나에 전체 마감을 두고 각 호출은 남은 시간만큼만 기다린다. 예약 생성·취소는 남은 시간이 충분할 때만 시작한다.
시간은 가짜 시계로 흘린다(실제로 기다리지 않는다).
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest

from app.core.config import Settings
from app.schemas.chat_schema import ChatContext, ChatRequest, UiSelection
from app.services import chat_deadline as dl
from app.services import chat_service as cs
from app.services import claude_structured_client as csc
from app.services.chat_deadline import ChatDeadlineExceeded, chat_deadline
from app.services.chat_router_service import ChatRouterService
from app.services.claude_structured_client import ClaudeStructuredClient, StructuredCallError
from app.services.medical_llm_service import MedicalLlmService
from app.services.reservation_api_client import ReservationApiClient
from app.services.reservation_intake import _extract_symptom_summary, merge_symptom_summary
from app.services.reservation_orchestrator import coarse_region
from tests.reservation_flow_helpers import FUTURE, FakeLoader, FakeReservation, make_orchestrator, make_request
from tests.test_reservation_confirm_chat_service import env  # noqa: F401

PROFILE = {
    "name": "홍길순", "phone": "010-1234-5678", "email": "gil@example.com", "birthDate": "1950-01-01",
    "address": "서울특별시 강남구 테헤란로 123", "addressDetail": "101동 202호", "postcode": "06234", "gender": "F",
    "location": None,
}
SECRETS = ("홍길순", "010-1234-5678", "gil@example.com", "1950-01-01", "테헤란로", "101동", "06234")


class _Clock:
    """chat_deadline 이 보는 시계. advance 로만 흐른다."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, sec: float) -> None:
        self.now += sec


@pytest.fixture()
def clock(monkeypatch) -> _Clock:
    c = _Clock()
    monkeypatch.setattr(dl, "time", SimpleNamespace(perf_counter=c))
    return c


# ---- S-6: Claude 입력 줄이기 ---------------------------------------------------------------------------------


class _RecordingLoader(FakeLoader):
    def __init__(self, extracted: dict[str, Any] | None = None) -> None:
        super().__init__(extracted)
        self.prompts: list[str] = []
        self.kwargs: list[dict[str, Any]] = []

    def generate_structured_text(self, system: str, prompt: str, schema: Any = None, **kwargs) -> str:
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        return super().generate_structured_text(system, prompt, schema)


def _orch_with(loader: _RecordingLoader, fake: FakeReservation):
    orch = make_orchestrator(fake)
    orch._loader = loader
    return orch


def test_extraction_prompt_has_region_only_and_no_contact_info() -> None:
    loader = _RecordingLoader({"reservation_date": FUTURE})
    history = [{"role": "user", "content": f"예전 이야기 {i}"} for i in range(10)]
    history[-1] = {"role": "user", "content": "제 번호는 010-9999-8888 이고 메일은 me@test.kr 이에요"}
    history[0] = {"role": "user", "content": "아주 오래된 이야기 마커"}
    req = make_request("근처 내과 예약해줘", history=history, context=PROFILE)
    _orch_with(loader, FakeReservation()).handle(req, {"rawMessage": req.message})
    prompt = loader.prompts[0]
    for secret in SECRETS + ("010-9999-8888", "me@test.kr", "아주 오래된 이야기 마커"):
        assert secret not in prompt, secret
    assert "서울특별시 강남구" in prompt
    assert "[전화]" in prompt and "[이메일]" in prompt
    assert loader.kwargs[0].get("stage") == "reservation_extract"


def test_profile_still_fills_patient_on_server_side() -> None:
    """프로필을 Claude에 보내지 않아도 예약 초안의 환자명·전화는 서버가 채운다(확인 카드에 그대로)."""
    loader = _RecordingLoader({"reservation_date": FUTURE})
    req = make_request("강남메디내과 내과 14:00 예약해줘", context=PROFILE)
    out = _orch_with(loader, FakeReservation()).handle(req, {"rawMessage": req.message})
    assert out["ui"].field == "reservation_confirm"
    draft = out["meta"]["pendingAction"]["draft"]
    assert draft["patient_name"] == "홍길순" and draft["phone"] == "010-1234-5678"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("서울특별시 강남구 테헤란로 123", "서울특별시 강남구"),
        ("서울 강남구 역삼동 1-2", "서울 강남구"),
        ("경기도 성남시 분당구 판교역로 235", "경기도 성남시 분당구"),
        ("서울 강남구 강남역입구 3", "서울 강남구"),
        ("테헤란로 123", None),
        ("", None),
    ],
)
def test_coarse_region(text: str, expected: str | None) -> None:
    assert coarse_region(text) == expected


class _RouterLoader:
    def __init__(self) -> None:
        self.state = SimpleNamespace(model_name="fake")
        self.prompts: list[str] = []
        self.kwargs: list[dict[str, Any]] = []

    def generate_structured_text(self, system: str, prompt: str, schema: Any = None, **kwargs) -> str:  # noqa: ARG002
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        return json.dumps({"route": "general", "normalized_message": "", "symptom_summary": "", "emergency": False,
                           "self_harm": False, "reason": "t"})


def test_router_history_masks_contact_patterns_but_keeps_turns() -> None:
    loader = _RouterLoader()
    history = [{"role": "user", "content": f"턴{i}"} for i in range(7)] + [
        {"role": "user", "content": "전화 010-2222-3333, 주민번호 500101-2345678, 메일 a@b.co"}
    ]
    ChatRouterService(loader, keep_turns=8).route("안녕하세요", history, PROFILE)  # type: ignore[arg-type]
    prompt = loader.prompts[0]
    assert "010-2222-3333" not in prompt and "500101-2345678" not in prompt and "a@b.co" not in prompt
    assert "[전화]" in prompt and "[주민번호]" in prompt and "[이메일]" in prompt
    assert all(f"턴{i}" in prompt for i in range(7))  # 턴 수는 그대로(안전 판단 근거)
    assert "홍길순" not in prompt  # 라우터는 프로필을 원래도 넣지 않는다
    assert loader.kwargs[0] == {"stage": "router", "reserve_sec": 30.0}


# ---- S-8: 증상 요약 ------------------------------------------------------------------------------------------


def test_symptom_summary_rules() -> None:
    assert _extract_symptom_summary("근처 내과 찾아줘") is None
    assert _extract_symptom_summary("내일 열 시에 예약해 줘") is None  # "열 시"는 증상이 아니다
    got = _extract_symptom_summary("어제 손주가 왔어요. 배가 아프고 열이 나요 010-1234-5678. 내과 예약해줘")
    assert got == "배가 아프고 열이 나요."
    long = _extract_symptom_summary("허리가 " + "너무 " * 40 + "아파요")
    assert long is not None and len(long) <= 60
    assert merge_symptom_summary("배가 아파요", "내일로 바꿔줘") == "배가 아파요"
    assert merge_symptom_summary("배가 아파요", "기침도 나요") == "배가 아파요 기침도 나요"


def test_symptom_summary_ignores_history_and_llm_value() -> None:
    loader = _RecordingLoader({"reservation_date": FUTURE, "symptom_summary": "이력에서 지어낸 요약"})
    history = [{"role": "user", "content": "허리가 아파요"}, {"role": "assistant", "content": "정형외과를 권해요"}]
    req = make_request("강남메디내과 내과 14:00 예약해줘", history=history)
    out = _orch_with(loader, FakeReservation()).handle(req, {"rawMessage": req.message})
    assert "symptom_summary" not in out["meta"]["pendingAction"]["draft"]


def test_symptom_summary_reaches_reservation_without_phone(env) -> None:  # noqa: F811
    def send(message: str = "", ui: tuple[str, str] | None = None, history=None):
        payload = ChatRequest(
            userId="user-1", message=message, sessionId="s1", history=history or [],
            context=ChatContext(name="홍길동", phone="010-1234-5678", location="서울 강남구"),
            uiSelection=UiSelection(field=ui[0], value=ui[1]) if ui else None,
        )
        return env.svc.process_message(env.db, payload)

    card = send("머리가 아파요 010-7777-8888. 강남메디내과 내과 14:00 예약해줘",
                history=[{"role": "user", "content": "어제 손주 결혼식에 다녀왔어요"}])
    token = next(o.value for o in card["ui"].options if o.value.startswith("confirm:"))
    send(ui=("reservation_confirm", token))
    created = [c[1] for c in env.fake.calls if c[0] == "create"]
    assert len(created) == 1
    summary = created[0]["symptom_summary"]
    assert summary and "머리가 아파요" in summary and "010" not in summary and "결혼식" not in summary


# ---- O-3: 마감 - Claude 호출 ---------------------------------------------------------------------------------


def _claude(timeout_sec: int = 60) -> ClaudeStructuredClient:
    settings = SimpleNamespace(anthropic_api_key="k", reservation_extract_model="claude-opus-5-5",
                               claude_timeout_sec=timeout_sec, claude_max_tokens=512)
    return ClaudeStructuredClient(settings)  # type: ignore[arg-type]


class _SlowAnthropic:
    """SDK처럼 timeout 을 지킨다: slow 초가 걸리는 호출은 timeout 만큼 흐른 뒤 시간 초과."""

    def __init__(self, clock: _Clock, slow: float) -> None:
        self.clock = clock
        self.slow = slow
        self.timeouts: list[float] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.timeouts.append(kwargs["timeout"])
        if self.slow > kwargs["timeout"]:
            self.clock.advance(kwargs["timeout"])
            raise anthropic.APITimeoutError(request=httpx.Request("POST", "https://example.invalid"))
        self.clock.advance(self.slow)
        block = SimpleNamespace(type="text", text='{"route":"general"}')
        return SimpleNamespace(stop_reason="end_turn", content=[block])


def test_claude_sdk_retries_are_off() -> None:
    assert _claude()._client.max_retries == 0


def test_claude_call_gets_only_remaining_time_and_one_retry(clock, monkeypatch) -> None:
    monkeypatch.setattr(csc, "time", SimpleNamespace(sleep=clock.advance))
    client = _claude()
    slow = _SlowAnthropic(clock, slow=500)
    client._client = slow
    with chat_deadline(110):
        with pytest.raises(StructuredCallError) as err:
            client.generate_structured_text("s", "u", stage="router", reserve_sec=30)
    assert err.value.cause == "connection"
    assert slow.timeouts == [60, pytest.approx(19.5)]  # 110-30=80 → 60 / 재시도: 남은 49.5-30
    assert clock.now - 1000 <= 110


def test_claude_call_not_started_when_no_time_left(clock) -> None:
    client = _claude()
    slow = _SlowAnthropic(clock, slow=1)
    client._client = slow
    with chat_deadline(20):
        with pytest.raises(StructuredCallError) as err:
            client.generate_structured_text("s", "u", stage="router", reserve_sec=30)
    assert err.value.cause == "deadline" and slow.timeouts == []


def test_without_deadline_claude_keeps_configured_timeout(clock) -> None:
    client = _claude()
    slow = _SlowAnthropic(clock, slow=1)
    client._client = slow
    client.generate_structured_text("s", "u", stage="router", reserve_sec=30)
    assert slow.timeouts == [60]


# ---- O-3: 마감 - 예약 API · MedGemma ----------------------------------------------------------------------------


def test_reservation_api_timeout_follows_remaining_time(clock, monkeypatch) -> None:
    seen: list[float] = []

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"[]"

    def fake_urlopen(req, timeout):  # noqa: ARG001
        seen.append(timeout)
        return _Resp()

    import app.services.reservation_api_client as rac

    monkeypatch.setattr(rac.urllib.request, "urlopen", fake_urlopen)
    client = ReservationApiClient(SimpleNamespace(
        reservation_api_base_url="http://x/api/v1", reservation_api_timeout_sec=120, reservation_user_jwt="",
        reservation_api_key="k",
    ))  # type: ignore[arg-type]
    with chat_deadline(50):
        client.list_my_appointments(api_key_override="k")
    with chat_deadline(4):
        with pytest.raises(ChatDeadlineExceeded):
            client.list_my_appointments(api_key_override="k")
    client.list_my_appointments(api_key_override="k")  # 챗 밖은 설정값 그대로
    assert seen == [48, 120]


class _SpyGen:
    def __init__(self) -> None:
        self.state = SimpleNamespace(model_name="fake")
        self.timeouts: list[float | None] = []

    def ensure_loaded(self) -> None:
        return None

    def generate_structured_text(self, system_prompt: str, user_prompt: str, prefill: str = "", *, wait_timeout_sec=None) -> str:  # noqa: ARG002
        self.timeouts.append(wait_timeout_sec)
        return '{"reply":"쉬어 보세요.","symptoms":""}'


def test_medgemma_wait_shrinks_and_skips_without_generation_time(clock) -> None:
    spy = _SpyGen()
    svc = MedicalLlmService(Settings(), spy)  # type: ignore[arg-type]
    with chat_deadline(70):
        svc.generate("어제부터 기침이 나요", {}, [], intent="medical_advice")
    with chat_deadline(100):
        svc.generate("가슴이 너무 아파서 숨을 못 쉬겠어요", {}, [], intent="medical_advice")  # 응급 3초 그대로
    with chat_deadline(20), pytest.raises(ChatDeadlineExceeded):
        svc.generate("어제부터 기침이 나요", {}, [], intent="medical_advice")
    assert spy.timeouts == [40, 3]


# ---- O-3: 마감 - chat_service 전체 경로 ------------------------------------------------------------------------


def _payload(message: str = "", ui: tuple[str, str] | None = None) -> ChatRequest:
    return ChatRequest(
        userId="user-1", message=message, sessionId="s1",
        context=ChatContext(name="홍길동", phone="010-1234-5678", location="서울 강남구"),
        uiSelection=UiSelection(field=ui[0], value=ui[1]) if ui else None,
    )


def _deadline(monkeypatch, seconds: float) -> None:
    monkeypatch.setattr(cs, "chat_deadline", lambda: chat_deadline(seconds))


def test_emergency_reply_survives_deadline(env, monkeypatch) -> None:  # noqa: F811
    def no_time(*a, **k):  # noqa: ARG001
        raise ChatDeadlineExceeded("medgemma_wait")

    monkeypatch.setattr(env.llm, "generate", no_time)
    out = env.svc.process_message(env.db, _payload("숨을 못 쉬겠어요"))
    assert out["riskLevel"] == "high" and "119" in out["reply"] and out["emergencyWarning"]


def test_self_harm_fixed_reply_unchanged_even_when_deadline_is_gone(env, monkeypatch, clock) -> None:  # noqa: F811
    normal = env.svc.process_message(env.db, _payload("죽고 싶어요"))
    _deadline(monkeypatch, 0.001)
    clock.advance(5)
    late = env.svc.process_message(env.db, _payload("죽고 싶어요"))
    assert late["reply"] == normal["reply"] and late["riskLevel"] == "high"


def test_reservation_deadline_falls_back_and_keeps_draft(env, monkeypatch) -> None:  # noqa: F811
    first = env.svc.process_message(env.db, _payload("근처 내과 예약해줘"))  # 병원 고르기 단계 - 초안 저장
    assert first["tool"] == "search_hospital"

    def no_time(*a, **k):  # noqa: ARG001
        raise ChatDeadlineExceeded("reservation_api:GET:/hospitals")

    monkeypatch.setattr(env.fake, "resolve_hospital", no_time)
    out = env.svc.process_message(env.db, _payload(ui=("hospital", "강남메디내과")))
    assert out["reply"] == "예약 정보를 처리하지 못했습니다. 잠시 후 다시 시도해 주세요."
    assert env.fake.names() == []
    flow = env.svc._last_flow_state(env.db, "user-1", "s1")
    assert flow["flowState"]["draft"]["department"] == "내과"  # 다음 턴이 이어 쓸 초안이 남는다


def _card(env) -> dict[str, Any]:  # noqa: F811
    return env.svc.process_message(env.db, _payload("강남메디내과 내과 14:00 예약해줘"))


def _tok(out: dict[str, Any], prefix: str) -> str:
    return next(o.value for o in out["ui"].options if o.value.startswith(prefix))


def test_create_not_started_without_enough_time_and_card_comes_back(env, monkeypatch, clock) -> None:  # noqa: F811
    card = _card(env)
    _deadline(monkeypatch, 10)  # 25초 미만
    again = env.svc.process_message(env.db, _payload(ui=("reservation_confirm", _tok(card, "confirm:"))))
    assert env.fake.names() == []
    assert again["ui"].field == "reservation_confirm" and again["reply"].startswith("처리가 늦어져")
    new_token = _tok(again, "confirm:")
    assert new_token != _tok(card, "confirm:")
    _deadline(monkeypatch, 110)
    done = env.svc.process_message(env.db, _payload(ui=("reservation_confirm", new_token)))
    assert env.fake.names() == ["create"] and done["tool"] == "make_appointment"


def test_create_rechecked_after_slow_lookup(env, monkeypatch, clock) -> None:  # noqa: F811
    card = _card(env)
    _deadline(monkeypatch, 40)
    original = env.fake.list_my_appointments

    def slow_list(api_key_override=None):
        clock.advance(20)  # 생성 직전 남은 시간 20초 < 25초
        return original(api_key_override)

    monkeypatch.setattr(env.fake, "list_my_appointments", slow_list)
    again = env.svc.process_message(env.db, _payload(ui=("reservation_confirm", _tok(card, "confirm:"))))
    assert env.fake.names() == [] and again["ui"].field == "reservation_confirm"


def test_cancel_not_started_without_enough_time(env, monkeypatch, clock) -> None:  # noqa: F811
    env.fake.appointments = {5: {"reservationId": 5, "hospitalName": "강남메디내과", "department": "내과",
                                 "reservationDate": FUTURE, "reservationTime": "09:00", "status": "BOOKED"}}
    env.svc.process_message(env.db, _payload("예약 취소할래"))
    confirm = env.svc.process_message(env.db, _payload(ui=("reservation_cancel_pick", "5")))
    _deadline(monkeypatch, 10)
    again = env.svc.process_message(env.db, _payload(ui=("reservation_cancel_confirm", _tok(confirm, "yes:"))))
    assert env.fake.names() == [] and again["reply"].startswith("처리가 늦어져")
    _deadline(monkeypatch, 110)
    env.svc.process_message(env.db, _payload(ui=("reservation_cancel_confirm", _tok(again, "yes:"))))
    assert env.fake.names() == ["cancel"]
