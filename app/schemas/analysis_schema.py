from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class AnalysisImageRequest(BaseModel):
    cameraIdentifier: str
    modelIdentifier: str
    imagePath: str | None = None


class AnalysisStreamControlRequest(BaseModel):
    cameraIdentifier: str
    modelIdentifier: str


class AnalysisOut(BaseModel):
    id: int
    analysisNo: str
    cameraId: int
    cameraIdentifier: str
    modelId: int
    modelIdentifier: str
    detectedType: str
    confidence: float
    danger: bool
    snapshotPath: str
    rawResultJson: str
    analyzedAt: datetime
    createdAt: datetime
