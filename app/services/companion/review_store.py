from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import DateTime, String, Text, create_engine, inspect, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker

from app.core.config import get_settings
from app.database.companion_base import CompanionBase

# 검수 목록 (명세 9-1 5·6번). AI가 만든 문장과 질문만 담는다. 어르신 대답 원문은 저장하지 않는다 (명세 9-4).
STATUSES = ("pending", "approved", "rejected")


class ReviewItem(CompanionBase):
    __tablename__ = "companion_review_item"

    itemId: Mapped[str] = mapped_column(String(32), primary_key=True)
    dialogueId: Mapped[str] = mapped_column(String(32), index=True)
    wardId: Mapped[str] = mapped_column(String(64), index=True, default="")  # 탈퇴 시 삭제할 때 찾는다 (노션 요구)
    turnNo: Mapped[int]
    text: Mapped[str] = mapped_column(Text)  # AI 가 만든 응답 문장 (꼬리질문 포함). 대답 원문 아님
    promptVersion: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16), index=True, default="pending")
    risk: Mapped[bool] = mapped_column(default=False)  # 위험 플래그(riskFlag)가 켜진 턴이면 참 (S7-5)
    createdAt: Mapped[datetime] = mapped_column(DateTime)
    decidedAt: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ReviewStore:
    """검수 항목 저장소. 엔진은 처음 쓸 때 만든다. 테스트에서는 다른 주소로 바꿔 끼운다."""

    def __init__(self) -> None:
        self._engine: Engine | None = None
        self._session_factory: sessionmaker | None = None
        self._lock = threading.Lock()

    def _sessions(self) -> sessionmaker:
        if self._session_factory is None:
            with self._lock:
                if self._session_factory is None:
                    url = get_settings().companion_review_database_url
                    self._engine = create_engine(url, future=True, pool_pre_ping=True)
                    self._prepare(self._engine)
                    self._session_factory = sessionmaker(self._engine, expire_on_commit=False)
        return self._session_factory

    def reset_engine(self, url: str) -> None:
        """테스트용: 다른 DB 주소로 바꾼다."""
        with self._lock:
            self._engine = create_engine(url, future=True)
            self._prepare(self._engine)
            self._session_factory = sessionmaker(self._engine, expire_on_commit=False)

    @staticmethod
    def _prepare(engine: Engine) -> None:
        """테이블을 만들고, 예전에 만든 테이블에 빠진 열(risk)을 채운다 (create_all 은 열을 추가하지 않는다)."""
        CompanionBase.metadata.create_all(engine, tables=[ReviewItem.__table__])
        columns = {c["name"] for c in inspect(engine).get_columns(ReviewItem.__tablename__)}
        added = {
            "risk": "ALTER TABLE {t} ADD COLUMN risk BOOLEAN NOT NULL DEFAULT 0",
            "wardId": "ALTER TABLE {t} ADD COLUMN wardId VARCHAR(64) NOT NULL DEFAULT ''",
        }
        for name, ddl in added.items():
            if name not in columns:
                with engine.begin() as conn:
                    conn.execute(text(ddl.format(t=ReviewItem.__tablename__)))

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc).replace(tzinfo=None)  # SQLite 는 시간대를 저장하지 않으므로 UTC 나이브 값으로 통일

    def add(self, dialogue_id: str, turn_no: int, text: str, prompt_version: str, risk: bool = False,
            ward_id: str = "") -> str:
        item_id = uuid.uuid4().hex
        with self._sessions()() as s, s.begin():
            s.add(ReviewItem(
                itemId=item_id, dialogueId=dialogue_id, turnNo=turn_no, text=text,
                promptVersion=prompt_version, status="pending", risk=risk, wardId=ward_id,
                createdAt=self._now(), decidedAt=None,
            ))
        return item_id

    def list_items(self, status: str | None = None) -> list[dict]:
        self.purge_expired()
        with self._sessions()() as s:
            q = select(ReviewItem).order_by(ReviewItem.createdAt)
            if status:
                q = q.where(ReviewItem.status == status)
            return [_to_dict(r) for r in s.scalars(q)]

    def decide(self, item_id: str, decision: str) -> dict | None:
        with self._sessions()() as s, s.begin():
            row = s.get(ReviewItem, item_id)
            if row is None:
                return None
            row.status = decision
            row.decidedAt = self._now()
            s.flush()
            return _to_dict(row)

    def delete_by_ward(self, ward_id: str) -> int:
        """피보호자 탈퇴 시: 그 피보호자의 검수 항목을 모두 지운다. 지운 개수를 돌려준다."""
        with self._sessions()() as s, s.begin():
            rows = list(s.scalars(select(ReviewItem).where(ReviewItem.wardId == ward_id)))
            for row in rows:
                s.delete(row)
            return len(rows)

    def purge_expired(self) -> int:
        """보관 기간(D18, 30일)이 지난 항목을 지운다. 목록을 읽을 때마다 확인한다."""
        days = get_settings().companion_log_retention_days
        cutoff = self._now() - timedelta(days=days)
        with self._sessions()() as s, s.begin():
            rows = list(s.scalars(select(ReviewItem).where(ReviewItem.createdAt < cutoff)))
            for row in rows:
                s.delete(row)
            return len(rows)


def _to_dict(row: ReviewItem) -> dict:
    return {
        "itemId": row.itemId,
        "dialogueId": row.dialogueId,
        "turnNo": row.turnNo,
        "text": row.text,
        "promptVersion": row.promptVersion,
        "status": row.status,
        "risk": row.risk,
        "createdAt": row.createdAt.isoformat(),
        "decidedAt": row.decidedAt.isoformat() if row.decidedAt else None,
    }


review_store = ReviewStore()
