"""예약 계정 비밀값·공용 계정은 코드 기본값을 갖지 않는다(2026-10-20).

비밀값이 비면 예약 기능만 꺼지고, 그 밖의 기능(상담·이상감지·클립)은 영향이 없어야 한다.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings
from app.database.base import Base
from app.models.reservation_credential import ReservationCredential
from app.services.reservation_credential_service import ReservationCredentialService


class _FakeClient:
    """ReservationApiClient 대역. 어떤 호출이 있었는지만 기록한다."""

    def __init__(self, *, has_secret: bool) -> None:
        self._settings = SimpleNamespace(
            has_reservation_account_secret=has_secret,
            reservation_account_secret="test-secret" if has_secret else "",
        )
        self.calls: list[str] = []

    def has_valid_reservation_key(self, api_key):
        self.calls.append("has_valid_reservation_key")
        return False

    def login_account(self, email, password):
        self.calls.append("login_account")
        return {"httpStatus": 200, "data": {"accessToken": "jwt"}}

    def signup_account(self, *args):
        self.calls.append("signup_account")
        return {"httpStatus": 201, "data": {}}

    def issue_api_key(self, access_token, name):
        self.calls.append("issue_api_key")
        return {"httpStatus": 201, "data": {"apiKey": "sbk_p1.secret1"}}


def _session():
    engine = create_engine("sqlite+pysqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_env_without_reservation_values_leaves_all_three_defaults_empty() -> None:
    """환경변수가 없으면 비밀값·공용 계정이 모두 빈 값이다(저장소에 공개된 기본값으로 동작하지 않는다)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("RESERVATION_")}
    code = (
        "import json; from app.core.config import Settings as S; "
        "print(json.dumps([S().reservation_account_secret, S().reservation_service_email, S().reservation_service_password]))"
    )
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert json.loads(out.stdout.strip().splitlines()[-1]) == ["", "", ""]


def test_blank_values_report_unavailable() -> None:
    assert Settings(reservation_account_secret="  ").has_reservation_account_secret is False
    assert Settings(reservation_account_secret="s").has_reservation_account_secret is True
    assert Settings(reservation_service_email="", reservation_service_password="").has_reservation_service_credentials is False


def test_ensure_credential_without_secret_makes_no_reservation_calls_and_stores_nothing() -> None:
    session = _session()
    client = _FakeClient(has_secret=False)

    key = ReservationCredentialService().ensure_credential(session, user_id="aaaaaa", client=client, name="홍길동", phone="")

    assert key is None
    assert client.calls == []
    assert session.query(ReservationCredential).count() == 0


def test_ensure_credential_with_secret_still_issues_key() -> None:
    session = _session()
    client = _FakeClient(has_secret=True)

    key = ReservationCredentialService().ensure_credential(session, user_id="aaaaaa", client=client, name="홍길동", phone="")

    assert key == "sbk_p1.secret1"
    assert "login_account" in client.calls
    assert "issue_api_key" in client.calls
