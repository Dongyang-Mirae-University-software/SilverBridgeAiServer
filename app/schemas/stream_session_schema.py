from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class StreamSessionCreate(BaseModel):
    sessionId: str = Field(min_length=1)
    cameraIdentifier: str = Field(min_length=1)
    deviceType: str = "ipad"


class StreamSessionOut(BaseModel):
    sessionId: str
    cameraIdentifier: str
    deviceType: str
    status: str
    lastFrameAt: str | None
    viewerUrl: str
    latestAnalysis: dict | None


class StreamStatusOut(BaseModel):
    sessionId: str
    status: str
    lastFrameAt: str | None
    fps: float
    viewerCount: int
    isAnalyzing: bool


class FrameIngestResult(BaseModel):
    sessionId: str
    status: str
    lastFrameAt: datetime
