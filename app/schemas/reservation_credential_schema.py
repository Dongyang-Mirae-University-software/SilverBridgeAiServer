from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.common import MemberId


class ReservationCredentialUpsertRequest(BaseModel):
    userId: MemberId
    reservationEmail: str | None = None
    apiKey: str = Field(min_length=10)


class ReservationCredentialUpsertResponse(BaseModel):
    userId: str
    reservationEmail: str | None = None
    apiKeyPrefix: str
    createdAt: str
    updatedAt: str
