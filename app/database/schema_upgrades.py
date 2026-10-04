from __future__ import annotations

import logging

from sqlalchemy import text
from sqlalchemy.engine import Engine

_LOG = logging.getLogger(__name__)

# 이 서버는 마이그레이션 도구 없이 create_all 로 테이블을 만든다. create_all 은 이미 있는 컬럼을 바꾸지 않으므로,
# 모델에서 타입을 바꾼 컬럼은 여기서 한 번만 바꾼다(멱등 — 이미 바뀌었으면 아무것도 하지 않는다).
_MEMBER_ID_COLUMNS = (
    ("chat_logs", "user_id"),
    ("reservation_credentials", "user_id"),
)
_INTEGER_TYPES = {"integer", "bigint", "smallint"}


def upgrade_member_id_columns(engine: Engine) -> list[str]:
    """회원 ID 컬럼을 INTEGER → VARCHAR(64) 로 바꾼다(QA AI-3). 기존 값은 "1" 처럼 문자열로 보존된다.

    PostgreSQL 만 대상이다(SQLite 는 새로 만든 테이블이 이미 VARCHAR 이고, 옛 INTEGER 컬럼도 문자열을 받는다).
    바꾼 "테이블.컬럼" 목록을 돌려준다.
    """
    if engine.dialect.name != "postgresql":
        return []
    changed: list[str] = []
    with engine.begin() as conn:
        for table, column in _MEMBER_ID_COLUMNS:
            data_type = conn.execute(
                text(
                    "SELECT data_type FROM information_schema.columns "
                    "WHERE table_schema = current_schema() AND table_name = :t AND column_name = :c"
                ),
                {"t": table, "c": column},
            ).scalar()
            if data_type not in _INTEGER_TYPES:
                continue
            # 식별자는 위 상수에서만 온다(사용자 입력 아님).
            conn.execute(text(f'ALTER TABLE "{table}" ALTER COLUMN "{column}" TYPE VARCHAR(64) USING "{column}"::text'))
            changed.append(f"{table}.{column}")
    if changed:
        _LOG.info("회원 ID 컬럼을 문자열로 변경: %s", ", ".join(changed))
    return changed
