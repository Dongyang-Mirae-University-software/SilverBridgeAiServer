"""예약 확인 토큰 선점이 DB 기반이라 서버가 여러 대여도 유효하고, DB 오류 때는 예약하지 않는지."""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta

import pytest
from sqlalchemy.exc import OperationalError

from app.database.session import SessionLocal
from app.models.reservation_token_claim import ReservationTokenClaim
from app.services import reservation_confirm_flow as flow
from tests.reservation_flow_helpers import FakeReservation, make_orchestrator
from tests.test_reservation_confirm_cancel import FULL, _confirm_card, _stored, _turn, _value


def _tok() -> str:
    return uuid.uuid4().hex[:12]


def test_second_claim_of_same_user_and_token_fails() -> None:
    claims, token = flow._TokenClaims(), _tok()
    assert claims.claim("u1", token) is True
    assert claims.claim("u1", token) is False


def test_separate_instances_share_the_claim_through_the_database() -> None:
    """서버가 두 대(= 메모리가 따로)여도 DB 가 하나라 한쪽이 선점하면 다른 쪽은 실패한다."""
    token = _tok()
    assert flow._TokenClaims().claim("u1", token) is True
    assert flow._TokenClaims().claim("u1", token) is False


def test_same_token_for_another_user_is_a_different_claim() -> None:
    claims, token = flow._TokenClaims(), _tok()
    assert claims.claim("u1", token) is True and claims.claim("u2", token) is True


def test_many_threads_claiming_one_token_get_exactly_one_winner() -> None:
    claims, token = flow._TokenClaims(), _tok()
    wins: list[bool] = []
    barrier = threading.Barrier(8)

    def run() -> None:
        barrier.wait()
        wins.append(claims.claim("u1", token))

    threads = [threading.Thread(target=run) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert wins.count(True) == 1 and len(wins) == 8


def test_expired_claims_are_purged_after_a_new_claim() -> None:
    old = _tok()
    with SessionLocal() as db:
        db.add(ReservationTokenClaim(user_id="u-old", token=old, created_at=datetime.utcnow() - timedelta(seconds=flow._TokenClaims.TTL_SECONDS + 60)))
        db.commit()
    fresh = _tok()
    assert flow._TokenClaims().claim("u-new", fresh) is True
    with SessionLocal() as db:
        assert db.get(ReservationTokenClaim, ("u-old", old)) is None  # 지워짐
        assert db.get(ReservationTokenClaim, ("u-new", fresh)) is not None  # 방금 것은 남음


def test_purge_failure_does_not_undo_a_successful_claim(monkeypatch) -> None:
    from sqlalchemy.exc import SQLAlchemyError

    claims, token = flow._TokenClaims(), _tok()
    real_session = flow.SessionLocal
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 2:  # 첫 번째 세션 = 선점, 두 번째 = 정리
            raise SQLAlchemyError("purge down")
        return real_session()

    monkeypatch.setattr(flow, "SessionLocal", flaky)
    assert claims.claim("u1", token) is True  # 정리가 실패해도 선점은 성공
    monkeypatch.setattr(flow, "SessionLocal", real_session)
    assert claims.claim("u1", token) is False  # 선점은 남아 있다


def test_database_error_during_claim_stops_the_booking(monkeypatch) -> None:
    """선점하지 못한 채 예약하는 것보다 요청이 실패하는 쪽이 안전하다."""
    fake = FakeReservation()
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)

    class _BrokenSession:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def add(self, *_a, **_k):
            return None

        def commit(self):
            raise OperationalError("INSERT", {}, Exception("db down"))

        def rollback(self):
            return None

    monkeypatch.setattr(flow, "SessionLocal", lambda: _BrokenSession())
    with pytest.raises(OperationalError):
        _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")))
    assert fake.names() == []
