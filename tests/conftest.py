from __future__ import annotations

import os
import tempfile

# app.core.config 는 임포트 시점에 기본값을 굳히고, app.database.session 은 임포트 시점에 엔진을 만든다.
# 어떤 테스트가 먼저 임포트하든 psycopg2·실 DB 없이 돌도록 수집 전에 정해 둔다(이미 설정돼 있으면 존중).
# 인메모리 sqlite 는 스레드마다 DB 가 달라 TestClient(별도 스레드)에서 테이블이 안 보인다 → 임시 파일.
_TEST_DB_DIR = tempfile.mkdtemp(prefix="sb_ai_test_")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TEST_DB_DIR}/test.db")
# 말벗 검수 DB 기본값은 ./data/companion_review.db 다. 그대로 두면 data/ 가 없는 사본에서는 열리지 않고,
# gosky 체크아웃에서는 data/ 가 운영 컨테이너 /app/data 라 테스트 행이 운영 검수 DB 에 쌓인다 → 같은 임시 폴더로.
os.environ.setdefault("COMPANION_REVIEW_DATABASE_URL", f"sqlite:///{_TEST_DB_DIR}/companion_review.db")
os.environ.setdefault("STREAM_STATE_BACKEND", "memory")


import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _create_tables():
    """lifespan 의 create_all 을 거치지 않는 테스트도 DB 테이블(예: 예약 토큰 선점)을 쓸 수 있게 한다."""
    import app.models.reservation_token_claim  # noqa: F401
    # 외래키가 가리키는 테이블(cameras, ai_models 등)도 먼저 등록해야 create_all 이 순서를 정할 수 있다
    import app.models.ai_model  # noqa: F401
    import app.models.analysis_result  # noqa: F401
    import app.models.camera  # noqa: F401
    import app.models.chat_log  # noqa: F401
    import app.models.game  # noqa: F401
    import app.models.reservation_credential  # noqa: F401
    import app.models.stream_session  # noqa: F401
    from app.database.base import Base
    from app.database.session import engine

    Base.metadata.create_all(bind=engine)


@pytest.fixture(autouse=True)
def _clear_claude_credit_skip():
    """크레딧 소진 건너뛰기(프로세스 메모리, #189)가 테스트 사이에 새지 않게 한다 - 새면 뒤 테스트가 규칙 문장·503을 받는다(점검 L-1)."""
    from app.services import claude_availability

    claude_availability.reset_for_tests()
    yield
    claude_availability.reset_for_tests()


@pytest.fixture(autouse=True)
def _clear_purge_hold():
    """회원 데이터 삭제 뒤 예약 계정 재생성 보류(프로세스 메모리, 점검 L-2)가 테스트 사이에 새지 않게 한다."""
    from app.services import reservation_credential_service as rcs

    rcs._PURGED.clear()
    yield
    rcs._PURGED.clear()
