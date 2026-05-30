from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class ChatContext(BaseModel):
    age: int | None = None
    gender: str | None = None
    guardianId: int | None = None


class ChatRequest(BaseModel):
    userId: int
    message: str
    context: ChatContext | None = None


class ChatReplyData(BaseModel):
    reply: str
    riskLevel: str
    recommendedAction: str
    reservationRequired: bool


class ChatLogOut(BaseModel):
    id: int
    chatNo: str
    userId: int
    message: str
    contextJson: str
    reply: str
    riskLevel: str
    recommendedAction: str
    reservationRequired: bool
    createdAt: datetime
