from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class GameCatalog(Base):
    __tablename__ = "game_catalogs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(128))
    description: Mapped[str] = mapped_column(Text, default="")
    total_stages: Mapped[int] = mapped_column(Integer, default=0)
    theme_color: Mapped[str] = mapped_column(String(32), default="#2563eb")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class GameStage(Base):
    __tablename__ = "game_stages"
    __table_args__ = (UniqueConstraint("game_slug", "stage_no", name="uq_game_stage_game_slug_stage_no"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    game_slug: Mapped[str] = mapped_column(String(64), index=True)
    stage_no: Mapped[int] = mapped_column(Integer, index=True)
    title: Mapped[str] = mapped_column(String(128))
    stage_type: Mapped[str] = mapped_column(String(32))
    prompt: Mapped[str] = mapped_column(Text, default="")
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    answer_json: Mapped[str] = mapped_column(Text, default="{}")
    max_score: Mapped[int] = mapped_column(Integer, default=100)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class GameProgress(Base):
    __tablename__ = "game_progresses"
    __table_args__ = (UniqueConstraint("user_id", "game_slug", name="uq_game_progress_user_game"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    game_slug: Mapped[str] = mapped_column(String(64), index=True)
    current_stage_no: Mapped[int] = mapped_column(Integer, default=1)
    score: Mapped[int] = mapped_column(Integer, default=0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    cleared: Mapped[bool] = mapped_column(Boolean, default=False)
    last_answer_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    state_json: Mapped[str] = mapped_column(Text, default="{}")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    cleared_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class GameAttempt(Base):
    __tablename__ = "game_attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, index=True)
    game_slug: Mapped[str] = mapped_column(String(64), index=True)
    stage_no: Mapped[int] = mapped_column(Integer, index=True)
    attempt_no: Mapped[int] = mapped_column(Integer, default=1)
    answer_json: Mapped[str] = mapped_column(Text, default="{}")
    is_correct: Mapped[bool] = mapped_column(Boolean, default=False)
    score_delta: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
