from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

SLOTS = ("09:00", "12:00", "15:00", "18:00", "21:00")
EMOTIONS = ("기쁨", "편안", "외로움", "기운 없음", "걱정")
ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"


class CheckinSessionStart(BaseModel):
    sessionId: str = Field(pattern=r"^[A-Za-z0-9_-]{8,64}$")
    userId: str = Field(pattern=ID_PATTERN)
    slot: Literal["09:00", "12:00", "15:00", "18:00", "21:00"]
    question: str = Field(min_length=1, max_length=200)


class CheckinFinish(BaseModel):
    userId: str = Field(pattern=ID_PATTERN)
    # 대화 끝 이모지(본인 응답). 누르지 않았으면 null
    selfReport: Literal["기쁨", "편안", "외로움", "기운 없음", "걱정"] | None = None
    reason: Literal["TURN_LIMIT", "TIME_LIMIT", "USER_END", "AI_END"] = "USER_END"
