from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class CheckinSession(Base):
    """안부 대화 한 번. 세션 ID 는 백엔드가 만든다(사용자 식별자를 싣지 않는 무작위 값)."""

    __tablename__ = "checkin_sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)  # 피보호자 회원 ID (탈퇴 정리 기준)
    slot: Mapped[str] = mapped_column(String(5))
    question: Mapped[str] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # finish 응답을 그대로 보관(같은 요청에 같은 답)


class CheckinTurn(Base):
    """한 턴. 음성·사진은 저장하지 않고 받아쓴 글자·AI 문장·숫자만 남긴다."""

    __tablename__ = "checkin_turns"
    __table_args__ = (UniqueConstraint("session_id", "turn_no", name="uq_checkin_turn"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    turn_no: Mapped[int] = mapped_column(Integer)
    question: Mapped[str] = mapped_column(Text)  # 이 턴에서 어르신이 들은 질문
    transcript: Mapped[str] = mapped_column(Text)
    reply: Mapped[str] = mapped_column(Text)
    done: Mapped[int] = mapped_column(Integer, default=0)
    voice_json: Mapped[str] = mapped_column(Text, default="{}")
    face_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
