from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base


class ReservationTokenClaim(Base):
    """예약 확인·취소 토큰을 한 번만 쓰게 하는 선점 기록. (user_id, token) 이 기본키라 먼저 INSERT 한 요청만 성공한다.

    서버를 여러 대로 늘려도 DB 가 하나이므로 유효하다. 행은 선점 직후 오래된 것부터 지운다(reservation_confirm_flow)."""

    __tablename__ = "reservation_token_claims"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    token: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
