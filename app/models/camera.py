from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    camera_no: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    identifier: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    stream_url: Mapped[str] = mapped_column(String(1024))
    stream_type: Mapped[str] = mapped_column(String(32), index=True)
    target_user_id: Mapped[str] = mapped_column(String(64), default="")
    guardian_user_id: Mapped[str] = mapped_column(String(64), default="")
    location_name: Mapped[str] = mapped_column(String(255), default="")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
