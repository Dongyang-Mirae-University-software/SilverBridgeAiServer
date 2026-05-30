from __future__ import annotations

from pydantic import BaseModel, Field


class ReservationCredentialUpsertRequest(BaseModel):
    userId: int
    reservationEmail: str | None = None
    apiKey: str = Field(min_length=10)


class ReservationCredentialUpsertResponse(BaseModel):
    userId: int
    reservationEmail: str | None = None
    apiKeyPrefix: str
    createdAt: str
    updatedAt: str
