from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


StreamType = Literal["rtsp", "http_mjpeg", "droidcam"]


class CameraCreate(BaseModel):
    cameraNo: str = Field(min_length=1)
    identifier: str = Field(min_length=1)
    name: str
    streamUrl: str
    streamType: StreamType
    targetUserId: str = ""
    guardianUserId: str = ""
    locationName: str = ""
    isActive: bool = True


class CameraUpdate(BaseModel):
    name: str | None = None
    streamUrl: str | None = None
    streamType: StreamType | None = None
    targetUserId: str | None = None
    guardianUserId: str | None = None
    locationName: str | None = None
    isActive: bool | None = None


class CameraOut(BaseModel):
    id: int
    cameraNo: str
    identifier: str
    name: str
    streamUrl: str
    streamType: str
    targetUserId: str
    guardianUserId: str
    locationName: str
    isActive: bool
    createdAt: datetime
    updatedAt: datetime
