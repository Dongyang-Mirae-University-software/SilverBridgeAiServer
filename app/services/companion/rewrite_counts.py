from __future__ import annotations

import logging
import threading
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import Integer, String, create_engine, delete, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, mapped_column, sessionmaker

from app.core.config import get_settings
from app.database.companion_base import CompanionBase

logger = logging.getLogger(__name__)

# 11/27 꼬리질문 고쳐 쓰기 날짜별 횟수. 고쳐 쓰기 결과는 로그에만 남아 gosky 가 배포마다 컨테이너를 새로 만들면 사라졌다
#   (BE docs/(2026-11-27) audit-companion-followup-rewrite-ops.md). 개수만 둔다 - 원문·피보호자·대화 식별자를 넣지 말 것.
#   검수 DB(COMPANION_REVIEW_DATABASE_URL, gosky 는 볼륨의 sqlite)에 둔다. 보관 90일(사용자 결정), 조회할 때 지운다.
KST = ZoneInfo("Asia/Seoul")
RETENTION_DAYS = 90
KINDS = {"followup": ("checked",), "tense": ("rewrite", "keep"), "vague": ("ok", "fail")}

_UPSERT = text(
    "INSERT INTO companion_rewrite_daily (day, kind, outcome, count) VALUES (:day, :kind, :outcome, 1) "
    "ON CONFLICT (day, kind, outcome) DO UPDATE SET count = companion_rewrite_daily.count + 1"
)


class RewriteDaily(CompanionBase):
    __tablename__ = "companion_rewrite_daily"

    day: Mapped[str] = mapped_column(String(10), primary_key=True)  # KST 'YYYY-MM-DD'
    kind: Mapped[str] = mapped_column(String(16), primary_key=True)
    outcome: Mapped[str] = mapped_column(String(16), primary_key=True)
    count: Mapped[int] = mapped_column(Integer, default=0)


def today_kst() -> date:
    return datetime.now(KST).date()


class RewriteCountStore:
    """고쳐 쓰기 횟수 저장소. 엔진은 처음 쓸 때 만든다 (검수 DB 와 같은 주소). 테스트에서는 다른 주소로 바꿔 끼운다."""

    def __init__(self) -> None:
        self._engine: Engine | None = None
        self._session_factory: sessionmaker | None = None
        self._lock = threading.Lock()

    def _sessions(self) -> sessionmaker:
        if self._session_factory is None:
            with self._lock:
                if self._session_factory is None:
                    self._set_engine(get_settings().companion_review_database_url)
        return self._session_factory

    def _set_engine(self, url: str) -> None:
        self._engine = create_engine(url, future=True, pool_pre_ping=True)
        CompanionBase.metadata.create_all(self._engine, tables=[RewriteDaily.__table__])
        self._session_factory = sessionmaker(self._engine, expire_on_commit=False)

    def reset_engine(self, url: str) -> None:
        """테스트용: 다른 DB 주소로 바꾼다."""
        with self._lock:
            self._set_engine(url)

    def record(self, outcomes: list[tuple[str, str]], day: date | None = None) -> None:
        """한 턴의 결과들을 한 트랜잭션으로 1씩 올린다. 실패해도 예외를 내지 않는다 - 대답은 그대로 나가야 한다."""
        if not outcomes:
            return
        d = (day or today_kst()).isoformat()
        try:
            with self._sessions()() as s, s.begin():
                for kind, outcome in outcomes:
                    if outcome not in KINDS.get(kind, ()):
                        continue
                    s.execute(_UPSERT, {"day": d, "kind": kind, "outcome": outcome})
        except Exception as e:  # DB 잠김·디스크 등 - 로그에는 클래스명만 (원문 없음)
            logger.warning("[COMPANION-REWRITE-COUNT] record failed: %s", type(e).__name__)

    def purge_expired(self, today: date | None = None) -> int:
        # 오늘 포함 90일(오늘-89 ~ 오늘)을 남긴다 - 조회 범위 상한과 같다
        cutoff = ((today or today_kst()) - timedelta(days=RETENTION_DAYS - 1)).isoformat()
        with self._sessions()() as s, s.begin():
            return s.execute(delete(RewriteDaily).where(RewriteDaily.day < cutoff)).rowcount or 0

    def by_day(self, start: date, end: date) -> list[dict]:
        """start~end(포함) 날짜별 표. 없는 날은 빼고, 없는 칸은 0."""
        self.purge_expired()
        with self._sessions()() as s:
            rows = s.scalars(select(RewriteDaily).where(RewriteDaily.day >= start.isoformat(),
                                                        RewriteDaily.day <= end.isoformat()))
            table: dict[str, dict] = {}
            for r in rows:
                cell = table.setdefault(r.day, {"day": r.day, **{f"{k}_{o}": 0 for k, os_ in KINDS.items() for o in os_}})
                cell[f"{r.kind}_{r.outcome}"] = r.count
        return [table[k] for k in sorted(table)]


rewrite_counts = RewriteCountStore()
