"""예약 계정 키를 못 받으면 예약을 멈춘다 - .env 공용 예약 키·JWT·공용 계정으로 대신하지 않는다(2026-10-30)."""
from __future__ import annotations

import logging
from types import SimpleNamespace

from app.services import chat_service as cs
from app.services.reservation_api_client import ReservationApiClient
from tests.test_chat_history_server_side import _router
from tests.test_reservation_confirm_chat_service import _send, env  # noqa: F401


def _settings(**shared):
    base = dict(reservation_api_base_url="http://x/api/v1", reservation_api_timeout_sec=10, reservation_api_key="",
                reservation_user_jwt="", reservation_admin_jwt="", reservation_service_email="", reservation_service_password="")
    return SimpleNamespace(**{**base, **shared})


SHARED = dict(reservation_api_key="shared-key", reservation_user_jwt="shared-jwt", reservation_service_email="svc@x",
              reservation_service_password="pw")


def _capture(monkeypatch) -> list:
    sent: list = []

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"[]"

    def fake_urlopen(req, timeout):  # noqa: ARG001
        sent.append({k.lower(): v for k, v in req.header_items()})
        return _Resp()

    import app.services.reservation_api_client as rac

    monkeypatch.setattr(rac.urllib.request, "urlopen", fake_urlopen)
    return sent


def test_without_member_key_no_request_is_sent_even_if_shared_values_exist(monkeypatch) -> None:
    sent = _capture(monkeypatch)
    client = ReservationApiClient(_settings(**SHARED))  # type: ignore[arg-type]
    out = client.list_my_appointments(api_key_override=None)
    assert out["httpStatus"] == 401 and out["data"]["code"] == "SERVER_CONFIG"
    assert sent == []  # 공용 키·공용 JWT·공용 계정 로그인 어느 것으로도 부르지 않는다


def test_member_key_is_used_as_is(monkeypatch) -> None:
    sent = _capture(monkeypatch)
    client = ReservationApiClient(_settings(**SHARED))  # type: ignore[arg-type]
    client.list_my_appointments(api_key_override="member-key")
    assert sent and sent[0].get("x-api-key") == "member-key"


def test_unauthenticated_calls_do_not_carry_the_shared_key(monkeypatch) -> None:
    sent = _capture(monkeypatch)
    client = ReservationApiClient(_settings(**SHARED))  # type: ignore[arg-type]
    client.login_account("a@b.c", "pw")
    assert sent and "x-api-key" not in sent[0] and "authorization" not in sent[0]


def test_shared_values_are_reported_once_at_startup_without_values(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        ReservationApiClient(_settings(**SHARED))  # type: ignore[arg-type]
    text = caplog.text
    assert "RESERVATION_API_KEY" in text and "RESERVATION_USER_JWT" in text and "RESERVATION_SERVICE_EMAIL" in text
    assert "shared-key" not in text and "shared-jwt" not in text and "svc@x" not in text


def test_no_warning_when_shared_values_are_empty(caplog) -> None:
    with caplog.at_level(logging.WARNING):
        ReservationApiClient(_settings())  # type: ignore[arg-type]
    assert "공용 예약 설정" not in caplog.text


def test_chat_stops_reservation_when_member_key_is_unavailable(env, monkeypatch) -> None:  # noqa: F811
    monkeypatch.setattr(cs, "get_reservation_credential_service", lambda: SimpleNamespace(ensure_credential=lambda *a, **k: None))
    handled: list = []
    monkeypatch.setattr(cs, "get_reservation_orchestrator", lambda: SimpleNamespace(handle=lambda *a, **k: handled.append(1)))
    _router(env, monkeypatch, route="reservation")
    out = _send(env, "내과 예약해 주세요")
    assert handled == [] and env.fake.calls == []  # 예약 흐름(검색 포함)을 키 없이 시작하지 않는다
    assert "지금은 병원 예약 기능을 사용할 수 없어요" in out["reply"]
