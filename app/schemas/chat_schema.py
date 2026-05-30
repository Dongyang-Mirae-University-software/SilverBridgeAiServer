from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatContext(BaseModel):
    age: int | None = None
    email: str | None = None
    name: str | None = None
    phone: str | None = None
    gender: str | None = None
    birthDate: str | None = None
    postcode: str | None = None
    address: str | None = None
    addressDetail: str | None = None
    guardianId: int | None = None
    location: str | None = None
    role: str | None = None


class ChatHistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class UiOption(BaseModel):
    value: str
    label: str


class UiPayload(BaseModel):
    kind: Literal["select", "date", "text"]
    field: str
    label: str
    options: list[UiOption] = Field(default_factory=list)
    placeholder: str | None = None


class UiSelection(BaseModel):
    field: str
    value: str


class ChatRequest(BaseModel):
    userId: int
    message: str = ""
    sessionId: str | None = None
    history: list[ChatHistoryItem] = Field(default_factory=list)
    context: ChatContext | None = None
    uiSelection: UiSelection | None = None


class ChatReplyData(BaseModel):
    sessionId: str
    reply: str
    riskLevel: str
    recommendedAction: str
    reservationRequired: bool
    intent: str
    engine: str
    modelName: str | None = None
    type: str = "message"
    ui: UiPayload | None = None
    tool: str | None = None
    data: Any | None = None


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
