from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class ClipRequest(BaseModel):
    """클립 요청. 모든 필드 선택 — 빈 본문이면 '지금' 기준 앞 3초·뒤 2초."""

    # ISO-8601. 시간대가 있으면 UTC로 바꾸고, 없으면(analyzedAt 원본 naive UTC) UTC로 해석한다.
    detectedAt: datetime | None = None
    preSeconds: float = Field(default=3, ge=0, le=5)
    postSeconds: float = Field(default=2, ge=0, le=3)

    @model_validator(mode="after")
    def _non_empty_window(self) -> "ClipRequest":
        if self.preSeconds + self.postSeconds <= 0:
            raise ValueError("preSeconds + postSeconds must be > 0")
        return self
