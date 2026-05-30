from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


ModelType = Literal["fall_detection", "fire_detection", "weapon_detection", "medical_chat", "custom"]
ModelFramework = Literal["ultralytics", "pytorch", "onnx", "transformers", "custom"]


class AIModelCreate(BaseModel):
    modelNo: str = Field(min_length=1)
    identifier: str = Field(min_length=1)
    name: str
    type: ModelType
    filePath: str
    framework: ModelFramework
    version: str = "v1"
    description: str = ""
    threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    isActive: bool = True


class AIModelUpdate(BaseModel):
    name: str | None = None
    filePath: str | None = None
    framework: ModelFramework | None = None
    version: str | None = None
    description: str | None = None
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    isActive: bool | None = None


class AIModelOut(BaseModel):
    id: int
    modelNo: str
    identifier: str
    name: str
    type: str
    filePath: str
    framework: str
    version: str
    description: str
    threshold: float
    isActive: bool
    createdAt: datetime
    updatedAt: datetime
