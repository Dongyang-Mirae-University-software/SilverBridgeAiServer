from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    analysis_no: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    camera_id: Mapped[int] = mapped_column(Integer, ForeignKey("cameras.id"), index=True)
    camera_identifier: Mapped[str] = mapped_column(String(128), index=True)
    model_id: Mapped[int] = mapped_column(Integer, ForeignKey("ai_models.id"), index=True)
    model_identifier: Mapped[str] = mapped_column(String(128), index=True)
    detected_type: Mapped[str] = mapped_column(String(64), default="unknown")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    danger: Mapped[bool] = mapped_column(Boolean, default=False)
    snapshot_path: Mapped[str] = mapped_column(String(1024), default="")
    raw_result_json: Mapped[str] = mapped_column(Text, default="{}")
    analyzed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
