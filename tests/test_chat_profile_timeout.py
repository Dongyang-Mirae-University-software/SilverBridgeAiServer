"""2026-10-29: 예약 API 호출 대기 기본 10초·상한 15초 · MedGemma 프롬프트에는 나이·성별만."""
from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

from app.core import config
from app.models.chat_log import ChatLog
from app.schemas.chat_schema import ChatContext, ChatRequest
from app.services import chat_service as cs
from app.services.chat_deadline import chat_deadline
from app.services.reservation_api_client import ReservationApiClient
from tests.test_chat_history_server_side import _router
from tests.test_reservation_confirm_chat_service import env  # noqa: F401

# ---- 예약 API 시간 제한 ---------------------------------------------------------------------------------------


def test_reservation_timeout_default_is_10_and_capped_at_15(monkeypatch) -> None:
    name = "RESERVATION_API_TIMEOUT_SEC"
    cases = {"": 10, "120": 15, "1200": 15, "15": 15, "7": 7, "0": 10, "-3": 10, "abc": 10}
    for raw, expected in cases.items():
        monkeypatch.setenv(name, raw)
        assert config._as_int_capped(name, 10, 15) == expected, raw


def test_settings_uses_capped_reservation_timeout() -> None:
    # Settings 기본값이 상한 헬퍼를 쓰는지(필드 기본값을 되돌리면 이 테스트가 잡는다).
    field = config.Settings.model_fields["reservation_api_timeout_sec"]
    assert field.default <= 15


def _slow_client(monkeypatch, timeout_sec: int, seen: list[float]) -> ReservationApiClient:
    def slow_urlopen(req, timeout):  # noqa: ARG001
        seen.append(timeout)
        raise TimeoutError("timed out")  # 예약 서비스가 응답하지 않아 소켓 대기가 끝난 상황

    import app.services.reservation_api_client as rac

    monkeypatch.setattr(rac.urllib.request, "urlopen", slow_urlopen)
    return ReservationApiClient(SimpleNamespace(
        reservation_api_base_url="http://x/api/v1", reservation_api_timeout_sec=timeout_sec, reservation_user_jwt="",
        reservation_api_key="k",
    ))  # type: ignore[arg-type]


def test_stalled_reservation_service_fails_within_15_seconds(monkeypatch) -> None:
    monkeypatch.setenv("RESERVATION_API_TIMEOUT_SEC", "120")  # gosky .env 값
    seen: list[float] = []
    client = _slow_client(monkeypatch, config._as_int_capped("RESERVATION_API_TIMEOUT_SEC", 10, 15), seen)
    with chat_deadline(110):
        out = client.list_my_appointments(api_key_override="k")
    assert seen == [15]  # 챗 마감 110초가 남아 있어도 한 번에 15초까지만 기다린다
    assert out == {"httpStatus": 0, "data": {"message": "timeout", "code": "UPSTREAM_TIMEOUT"}}


# ---- MedGemma 입력 최소화 -------------------------------------------------------------------------------------

_FULL = dict(
    name="홍길동", phone="010-1234-5678", email="a@b.c", location="서울 강남구", address="서울 강남구 테헤란로 1",
    addressDetail="101호", postcode="06234", birthDate="1950-03-02", gender="FEMALE", guardianId=7, role="GUARDIAN",
)
_PRIVATE = ("홍길동", "김순자", "010", "a@b.c", "테헤란", "101호", "06234", "1950", "ward-9", "GUARDIAN", "강남")


def _spy_llm(env, monkeypatch) -> list[dict]:  # noqa: F811
    seen: list[dict] = []
    original = env.llm.generate

    def spy(message, ctx, history, **kwargs):
        seen.append(dict(ctx))
        return original(message, ctx, history, **kwargs)

    monkeypatch.setattr(env.llm, "generate", spy)
    return seen


def _send(env, message: str, **ctx):  # noqa: F811
    payload = ChatRequest(userId="user-1", message=message, sessionId="s1", context=ChatContext(**ctx))
    return env.svc.process_message(env.db, payload)


def test_medgemma_gets_only_age_and_gender_in_proxy_consult(env, monkeypatch) -> None:  # noqa: F811
    _router(env, monkeypatch, route="symptom")
    seen = _spy_llm(env, monkeypatch)
    _send(env, "어제부터 기침이 나요", age=76, wardId="ward-9", wardName="김순자", wardPhone="010-9999-8888", **_FULL)
    assert seen == [{"age": 76, "gender": "FEMALE"}]
    assert not any(p in json.dumps(seen, ensure_ascii=False) for p in _PRIVATE)


def test_medgemma_age_from_birth_date_when_age_missing(env, monkeypatch) -> None:  # noqa: F811
    _router(env, monkeypatch, route="symptom")
    seen = _spy_llm(env, monkeypatch)
    _send(env, "어제부터 기침이 나요", **_FULL)
    today = datetime.now(cs._KST).date()
    expected = today.year - 1950 - ((today.month, today.day) < (3, 2))
    assert seen == [{"age": expected, "gender": "FEMALE"}]


def test_medgemma_context_empty_without_profile(env, monkeypatch) -> None:  # noqa: F811
    _router(env, monkeypatch, route="symptom")
    seen = _spy_llm(env, monkeypatch)
    _send(env, "어제부터 기침이 나요", name="홍길동", phone="010-1234-5678")
    assert seen == [{}]


def test_bad_birth_date_is_dropped() -> None:
    for raw in ("", "1950", "1950-13-40", "abcd-ef-gh", "3000-01-01"):
        assert cs._age_from_birth_date(raw) is None, raw


def test_original_profile_still_used_for_chat_log_and_reservation(env, monkeypatch) -> None:  # noqa: F811
    _router(env, monkeypatch, route="symptom")
    _spy_llm(env, monkeypatch)
    _send(env, "어제부터 기침이 나요", wardId="ward-9", wardName="김순자", wardPhone="010-9999-8888", **_FULL)
    row = env.db.query(ChatLog).order_by(ChatLog.id.desc()).first()
    stored = json.loads(row.context_json)["userContext"]
    # 챗 기록은 피보호자 구분(wardId)·예약 환자 정보를 그대로 갖는다(_row_ward_id·초안 채우기가 쓴다).
    assert stored["wardId"] == "ward-9" and stored["name"] == "김순자" and stored["phone"] == "010-9999-8888"

    seen_cred: list[dict] = []

    class _Cred:
        def ensure_credential(self, db, **kwargs):  # noqa: ARG002
            seen_cred.append(kwargs)
            return "fake-key"

    monkeypatch.setattr(cs, "get_reservation_credential_service", lambda: _Cred())
    _router(env, monkeypatch, route="reservation")
    _send(env, "내과 예약해 주세요", wardId="ward-9", wardName="김순자", wardPhone="010-9999-8888", **_FULL)
    assert seen_cred and seen_cred[0]["name"] == "김순자" and seen_cred[0]["phone"] == "010-9999-8888"
    assert seen_cred[0]["user_id"] == "ward-9"
