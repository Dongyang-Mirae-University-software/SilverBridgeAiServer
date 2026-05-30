from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class AIModel(Base):
    __tablename__ = "ai_models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    model_no: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    identifier: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    type: Mapped[str] = mapped_column(String(64), index=True)
    file_path: Mapped[str] = mapped_column(String(1024))
    framework: Mapped[str] = mapped_column(String(64))
    version: Mapped[str] = mapped_column(String(64), default="v1")
    description: Mapped[str] = mapped_column(Text, default="")
    threshold: Mapped[float] = mapped_column(Float, default=0.75)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
