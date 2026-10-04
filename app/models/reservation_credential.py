from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class ReservationCredential(Base):
    __tablename__ = "reservation_credentials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    # 회원 ID(문자열). 운영 DB 의 옛 INTEGER 컬럼은 시작 시 schema_upgrades 가 바꾼다(QA AI-3).
    user_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    reservation_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    api_key_prefix: Mapped[str] = mapped_column(String(64), index=True)
    api_key: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
