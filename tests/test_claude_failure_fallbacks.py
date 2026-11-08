"""Claude 실패 대비(2026-11-08): 예약 추출·이어서 물어볼 질문·약 봉투 인식이 크레딧 소진·시간 초과·키 없음에도 끊기지 않는다.

실제 Anthropic 호출은 하지 않는다 - SDK 예외를 직접 만들어 가짜 클라이언트가 던진다.
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest

from app.core.config import get_settings
from app.services import claude_availability as ca
from app.services import claude_structured_client as csc
from app.services import follow_up_questions as fq
from app.services.claude_structured_client import ClaudeStructuredClient, StructuredCallError
from app.services.medication_ocr_service import (
    MedicationOcrService,
    OcrNotConfiguredError,
    OcrUnavailableError,
    OcrUpstreamError,
)

_REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
_CREDIT_TEXT = "Your credit balance is too low to access the Anthropic API. 비밀문구"


def credit_error() -> anthropic.BadRequestError:
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": _CREDIT_TEXT}}
    return anthropic.BadRequestError(_CREDIT_TEXT, response=httpx.Response(400, request=_REQ), body=body)


def other_400() -> anthropic.BadRequestError:
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": "messages: field required"}}
    return anthropic.BadRequestError("bad", response=httpx.Response(400, request=_REQ), body=body)


def timeout_error() -> anthropic.APITimeoutError:
    return anthropic.APITimeoutError(request=_REQ)


class _Block:
    def __init__(self, text: str) -> None:
        self.type, self.text = "text", text


class _Resp:
    def __init__(self, text: str) -> None:
        self.content, self.stop_reason, self.usage = [_Block(text)], "end_turn", None


class FakeMessages:
    """errors 를 차례로 던지고, 다 쓰면 response 를 돌려준다. 호출 횟수를 센다."""

    def __init__(self, *errors: Exception, response: Any = None) -> None:
        self.errors = list(errors)
        self.response = response
        self.calls = 0

    def create(self, **kwargs: Any) -> Any:  # noqa: ARG002
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.response


def _attach(target: Any, messages: FakeMessages) -> None:
    target._client = SimpleNamespace(beta=SimpleNamespace(messages=messages))


def structured_client(messages: FakeMessages | None, *, key: str = "sk-test") -> ClaudeStructuredClient:
    client = ClaudeStructuredClient(get_settings().model_copy(update={"anthropic_api_key": key}), "fake-model")
    if messages is not None:
        _attach(client, messages)
    return client


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    ca.reset_for_tests()
    monkeypatch.setattr(csc, "_RETRY_BACKOFF_SEC", 0.0)
    yield
    ca.reset_for_tests()


# ---- 공용 클라이언트 ------------------------------------------------------------------------


def test_credit_error_marks_and_skips_only_for_opted_in_callers(caplog) -> None:
    msgs = FakeMessages(credit_error(), credit_error())
    client = structured_client(msgs)
    with caplog.at_level(logging.INFO):
        with pytest.raises(StructuredCallError) as first:
            client.generate_structured_text("s", "u", skip_when_exhausted=True)
        assert first.value.cause == "credit_exhausted" and ca.skipping()
        assert msgs.calls == 1  # 400 은 다시 시도하지 않는다

        with pytest.raises(StructuredCallError) as skipped:
            client.generate_structured_text("s", "u", skip_when_exhausted=True)
        assert skipped.value.cause == "credit_exhausted" and msgs.calls == 1  # 건너뛰는 동안 부르지 않는다

        # 건너뛰기를 켜지 않은 호출자(라우터)는 예전처럼 부르고 예전 코드(api_400)를 받는다.
        with pytest.raises(StructuredCallError) as router:
            client.generate_structured_text("s", "u")
        assert router.value.cause == "api_400" and msgs.calls == 2
    assert "비밀문구" not in caplog.text and "credit balance" not in caplog.text
    assert "[CLAUDE-CREDIT] cause=credit_exhausted" in caplog.text


def test_other_400_is_not_credit_exhaustion() -> None:
    client = structured_client(FakeMessages(other_400()))
    with pytest.raises(StructuredCallError) as exc:
        client.generate_structured_text("s", "u", skip_when_exhausted=True)
    assert exc.value.cause == "api_400" and not ca.skipping()


def test_skip_expires_then_one_retry_and_success_clears(monkeypatch) -> None:
    msgs = FakeMessages(credit_error(), response=_Resp('{"ok": true}'))
    client = structured_client(msgs)
    monkeypatch.setattr(ca, "SKIP_SEC", 0.0)  # 바로 다시 시도할 수 있게
    with pytest.raises(StructuredCallError):
        client.generate_structured_text("s", "u", skip_when_exhausted=True)
    assert not ca.skipping()
    assert client.generate_structured_text("s", "u", skip_when_exhausted=True) == '{"ok": true}'
    assert msgs.calls == 2 and not ca.skipping()


def test_success_clears_skip() -> None:
    ca.mark_exhausted()
    client = structured_client(FakeMessages(response=_Resp("{}")))
    assert client.generate_structured_text("s", "u") == "{}"  # 라우터처럼 건너뛰기를 안 켠 호출이 성공하면
    assert not ca.skipping()


def test_timeout_without_retry_and_default_retry() -> None:
    once = FakeMessages(timeout_error(), timeout_error())
    with pytest.raises(StructuredCallError) as exc:
        structured_client(once).generate_structured_text("s", "u", retry_on_timeout=False)
    assert exc.value.cause == "timeout" and once.calls == 1

    twice = FakeMessages(timeout_error(), timeout_error())
    with pytest.raises(StructuredCallError) as exc:
        structured_client(twice).generate_structured_text("s", "u")
    assert exc.value.cause == "connection" and twice.calls == 2  # 기본은 예전 그대로
    assert not ca.skipping()


def test_no_key_is_not_ready() -> None:
    with pytest.raises(StructuredCallError) as exc:
        structured_client(None, key="").generate_structured_text("s", "u", skip_when_exhausted=True)
    assert exc.value.cause == "not_ready"


# ---- 이어서 물어볼 질문 --------------------------------------------------------------------

ON = SimpleNamespace(follow_up_enabled=True, follow_up_timeout_sec=1.0, follow_up_model="fake")
ANSWER = {"reply": "두통은 쉬면 나아지는 경우가 많아요.", "riskLevel": "medium", "intent": "medical_advice", "type": "message", "ui": None}
SYMPTOM = {"route": "symptom", "emergency": False}
_MODEL_QUESTIONS = json.dumps({"questions": ["언제 병원에 가요?", "물 마셔도 되나요?"]}, ensure_ascii=False)


def _suggest() -> list[str]:
    return fq.suggested_questions(ON, ANSWER, SYMPTOM, "머리가 아파요", proxy=False)


@pytest.mark.parametrize(
    "messages, key, cause",
    [
        (FakeMessages(credit_error()), "sk-test", "credit_exhausted"),
        (FakeMessages(timeout_error(), timeout_error()), "sk-test", "connection"),
        (None, "", "not_ready"),
    ],
    ids=["credit", "timeout", "no_key"],
)
def test_follow_up_failures_give_rule_questions(monkeypatch, caplog, messages, key, cause) -> None:
    client = structured_client(messages, key=key)
    monkeypatch.setattr(fq, "_get_client", lambda settings: client)
    with caplog.at_level(logging.INFO, logger=fq.__name__):
        assert _suggest() == fq.rule_questions("symptom", proxy=False)
    assert f"source=rule count=3 cause={cause}" in caplog.text


def test_follow_up_skips_model_while_exhausted(monkeypatch, caplog) -> None:
    called: list[int] = []
    monkeypatch.setattr(fq, "_model_questions", lambda *a: called.append(1) or ["질문이에요?"])
    ca.mark_exhausted()
    with caplog.at_level(logging.INFO, logger=fq.__name__):
        assert _suggest() == fq.rule_questions("symptom", proxy=False)
    assert called == [] and "cause=credit_exhausted" in caplog.text


def test_follow_up_success_uses_model(monkeypatch) -> None:
    client = structured_client(FakeMessages(response=_Resp(_MODEL_QUESTIONS)))
    monkeypatch.setattr(fq, "_get_client", lambda settings: client)
    assert _suggest() == ["언제 병원에 가요?", "물 마셔도 되나요?"]


# ---- 약 봉투 인식(서비스) -------------------------------------------------------------------

_OCR_OK = json.dumps({"readable": True, "reason": None, "drugs": [{
    "name": "슈다페드정", "perDose": 1, "timesPerDay": 3, "days": 3, "slots": ["MORNING"], "timing": None, "confidence": "high",
}]}, ensure_ascii=False)


def ocr_service(messages: FakeMessages | None, *, key: str = "sk-test") -> MedicationOcrService:
    service = MedicationOcrService(get_settings().model_copy(update={"anthropic_api_key": key}))
    if messages is not None:
        _attach(service, messages)
    return service


def test_ocr_credit_error_is_unavailable_and_then_skipped() -> None:
    msgs = FakeMessages(credit_error())
    service = ocr_service(msgs)
    with pytest.raises(OcrUnavailableError):
        service.read(b"x")
    with pytest.raises(OcrUnavailableError):
        service.read(b"x")
    assert msgs.calls == 1  # 두 번째는 부르지 않았다


def test_ocr_timeout_stays_upstream_error() -> None:
    with pytest.raises(OcrUpstreamError):
        ocr_service(FakeMessages(timeout_error())).read(b"x")
    assert not ca.skipping()


def test_ocr_no_key_is_not_configured() -> None:
    with pytest.raises(OcrNotConfiguredError):
        ocr_service(None, key="").read(b"x")


def test_ocr_success_reads_and_clears_skip(monkeypatch) -> None:
    monkeypatch.setattr(ca, "SKIP_SEC", 0.0)
    ca.mark_exhausted()
    result = ocr_service(FakeMessages(response=_Resp(_OCR_OK))).read(b"x")
    assert result.readable and result.drugs[0].name == "슈다페드정" and not ca.skipping()


# ---- 약 봉투 인식(API 응답) -----------------------------------------------------------------


def test_ocr_route_answers_503_unavailable_on_credit_exhaustion(caplog) -> None:
    from fastapi.testclient import TestClient

    from tests.test_medication_ocr import FakeService, _make_app, _post

    with caplog.at_level(logging.INFO):
        with TestClient(_make_app(FakeService(error=OcrUnavailableError("credit_exhausted")))) as client:
            res = _post(client)
    assert res.status_code == 503  # 400 이면 BE 가 "사진 형식" 안내로 바꾼다 - 쓰지 않는다
    body = res.json()
    assert body["errorCode"] == "OCR_UNAVAILABLE"
    assert body["message"] == "사진 인식을 지금 쓸 수 없습니다. 직접 입력해 주세요."
    assert "result=unavailable_credit_exhausted" in caplog.text


# ---- 예약: Claude 가 실패해도 버튼만으로 예약 완료까지 --------------------------------------


def _book_by_choices(loader: Any) -> tuple[Any, list[dict[str, Any]]]:
    from tests.test_reservation_choice_only import SELF_NO_ADDRESS, _fake, _Session, _values
    from tests.reservation_flow_helpers import FUTURE

    fake = _fake()
    s = _Session(fake, SELF_NO_ADDRESS)
    s.orch._loader = loader
    first = s.say("신경과 예약은 어떻게 하면 되나요?")
    assert first["ui"].field == "location_sido"
    s.pick("location_sido", "서울")
    s.pick("location", "서울 서초구")
    s.pick("hospital_confirm", "서초내과")
    time_ask = s.pick("date", FUTURE)
    card = s.pick("time", _values(time_ask)[0])
    done = s.pick("reservation_confirm", next(v for v in _values(card) if v.startswith("confirm:")))
    assert fake.names() == ["create"] and done["ui"].field == "reservation_followup"
    return fake, s.outs


def _extract_failures(outs: list[dict[str, Any]]) -> list[str]:
    return [
        str(step.get("error"))
        for out in outs
        for step in (out.get("meta") or {}).get("decisionTrace") or []
        if step.get("event") == "reservation_extraction_failed"
    ]


def test_reservation_completes_when_credit_exhausted() -> None:
    msgs = FakeMessages(*[credit_error() for _ in range(5)])
    _, outs = _book_by_choices(structured_client(msgs))
    assert msgs.calls == 1  # 첫 문장에서 한 번 실패한 뒤로는 부르지 않는다
    assert _extract_failures(outs) and set(_extract_failures(outs)) == {"credit_exhausted"}


def test_reservation_completes_on_timeout_without_waiting_twice() -> None:
    msgs = FakeMessages(*[timeout_error() for _ in range(5)])
    _, outs = _book_by_choices(structured_client(msgs))
    assert msgs.calls == 1  # 첫 문장 한 번, 시간 초과를 다시 기다리지 않았다(버튼 턴은 추출을 부르지 않는다)
    assert set(_extract_failures(outs)) == {"timeout"}


def test_reservation_completes_without_key() -> None:
    _, outs = _book_by_choices(structured_client(None, key=""))
    assert set(_extract_failures(outs)) == {"not_ready"}


def test_reservation_free_text_still_reaches_confirmation_on_failure() -> None:
    from tests.reservation_flow_helpers import FakeReservation, make_orchestrator, make_request

    orch = make_orchestrator(FakeReservation())
    orch._loader = structured_client(FakeMessages(credit_error()))
    msg = "서울 서초구 강남메디내과 내일 오전 10시 내과 예약해줘"
    out = orch.handle(make_request(msg), {"storedFlow": {}, "rawMessage": msg})
    assert out is not None and out["ui"].field == "reservation_confirm"  # 규칙 추출만으로 확인 카드까지
