from __future__ import annotations

import os
import tempfile

# app.core.config 는 임포트 시점에 기본값을 굳히고, app.database.session 은 임포트 시점에 엔진을 만든다.
# 어떤 테스트가 먼저 임포트하든 psycopg2·실 DB 없이 돌도록 수집 전에 정해 둔다(이미 설정돼 있으면 존중).
# 인메모리 sqlite 는 스레드마다 DB 가 달라 TestClient(별도 스레드)에서 테이블이 안 보인다 → 임시 파일.
os.environ.setdefault("DATABASE_URL", f"sqlite:///{tempfile.mkdtemp(prefix='sb_ai_test_')}/test.db")
os.environ.setdefault("STREAM_STATE_BACKEND", "memory")


import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _create_tables():
    """lifespan 의 create_all 을 거치지 않는 테스트도 DB 테이블(예: 예약 토큰 선점)을 쓸 수 있게 한다."""
    import app.models.reservation_token_claim  # noqa: F401
    from app.database.base import Base
    from app.database.session import engine

    Base.metadata.create_all(bind=engine)
