"""예약 키 저장용 암호화·옛 카메라 경로 스위치(2026-11-10 전체 점검 M-5·M-8). 키 값은 모두 지어낸 값이다."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.database.schema_upgrades import encrypt_plain_reservation_keys
from app.models.reservation_credential import ReservationCredential
from app.models.reservation_token_claim import ReservationTokenClaim
from app.services import reservation_credential_service as rcs
from app.services import reservation_key_cipher as cipher

SECRET = "unit-test-secret"
KEY = "sbk_abcd1234." + "x" * 24


class _S:
    def __init__(self, secret: str) -> None:
        self.reservation_account_secret = secret
        self.has_reservation_account_secret = bool(secret)


@pytest.fixture()
def db():
    engine = create_engine(f"sqlite:///{tempfile.mkdtemp(prefix='sb_rk_')}/t.db")
    ReservationCredential.__table__.create(bind=engine)
    ReservationTokenClaim.__table__.create(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session, engine
    session.close()


def _raw(engine, user_id: str) -> str:
    with engine.connect() as conn:
        return conn.execute(text("SELECT api_key FROM reservation_credentials WHERE user_id = :u"), {"u": user_id}).scalar_one()


def test_seal_roundtrip_and_wrong_secret() -> None:
    sealed = cipher.seal(KEY, SECRET)
    assert sealed.startswith(cipher.PREFIX) and KEY not in sealed
    assert cipher.unseal(sealed, SECRET) == KEY
    assert cipher.unseal(sealed, "other-secret") is None  # 비밀값이 바뀌면 키가 없는 것으로 본다(다시 발급)
    assert cipher.unseal(sealed, "") is None
    assert cipher.unseal(KEY, SECRET) == KEY  # 옛 평문 행은 그대로 읽는다
    assert cipher.seal(KEY, SECRET) != sealed  # 같은 키도 매번 다른 암호문


def test_seal_without_secret_is_refused() -> None:
    with pytest.raises(ValueError):
        cipher.seal(KEY, "")


def test_upsert_stores_ciphertext_and_reads_back(db, monkeypatch: pytest.MonkeyPatch) -> None:
    session, engine = db
    monkeypatch.setattr(rcs, "get_settings", lambda: _S(SECRET))
    svc = rcs.ReservationCredentialService()
    svc.upsert_credential(session, user_id="u1", reservation_email="a@b.c", api_key=KEY)
    stored = _raw(engine, "u1")
    assert stored.startswith(cipher.PREFIX) and KEY not in stored
    assert svc.get_api_key(session, "u1") == KEY and svc.has_credential(session, "u1")


def test_upsert_without_secret_does_not_store_plaintext(db, monkeypatch: pytest.MonkeyPatch) -> None:
    session, _ = db
    monkeypatch.setattr(rcs, "get_settings", lambda: _S(""))
    with pytest.raises(ValueError):
        rcs.ReservationCredentialService().upsert_credential(session, user_id="u1", reservation_email=None, api_key=KEY)
    assert session.query(ReservationCredential).count() == 0


def test_startup_upgrade_encrypts_plain_rows_once(db, monkeypatch: pytest.MonkeyPatch) -> None:
    session, engine = db
    session.add(ReservationCredential(user_id="old", reservation_email=None, api_key_prefix="sbk_abcd1234", api_key=KEY))
    session.commit()
    assert encrypt_plain_reservation_keys(engine, "") == 0 and _raw(engine, "old") == KEY  # 비밀값 없으면 그대로
    assert encrypt_plain_reservation_keys(engine, SECRET) == 1
    assert _raw(engine, "old").startswith(cipher.PREFIX)
    assert encrypt_plain_reservation_keys(engine, SECRET) == 0  # 다시 불러도 건드리지 않는다
    monkeypatch.setattr(rcs, "get_settings", lambda: _S(SECRET))
    session.expire_all()
    assert rcs.ReservationCredentialService().get_api_key(session, "old") == KEY


def _route_status(flag: str | None) -> str:
    env = {k: v for k, v in os.environ.items() if k != "LEGACY_CAMERA_API_ENABLED"}
    if flag is not None:
        env["LEGACY_CAMERA_API_ENABLED"] = flag
    code = (
        "from app.main import app\n"
        "paths = set(app.openapi()['paths'])\n"
        "print(int('/api/v1/cameras' in paths), int(any(p.startswith('/api/v1/analysis') for p in paths)),"
        " int(any('stream-sessions' in p for p in paths)))\n"
    )
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120).stdout.strip().splitlines()[-1]


def test_legacy_camera_routes_off_by_default_and_on_by_flag() -> None:
    assert _route_status(None) == "0 0 1"  # 프레임 업로드 경로는 그대로
    assert _route_status("true") == "1 1 1"
