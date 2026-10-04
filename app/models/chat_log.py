from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    chat_no: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # 회원 ID(문자열). 운영 DB 의 옛 INTEGER 컬럼은 시작 시 schema_upgrades 가 바꾼다(QA AI-3).
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    message: Mapped[str] = mapped_column(Text)
    context_json: Mapped[str] = mapped_column(Text, default="{}")
    reply: Mapped[str] = mapped_column(Text)
    risk_level: Mapped[str] = mapped_column(String(32), default="low")
    recommended_action: Mapped[str] = mapped_column(String(64), default="home_monitoring")
    reservation_required: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
