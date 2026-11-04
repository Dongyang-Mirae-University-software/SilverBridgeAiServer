from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.schemas.companion_schema import TIME_SLOTS


class SlotSummary(BaseModel):
    """하루 한 시간대의 대화 결과 (백엔드가 모아서 보낸다)."""

    slot: str
    label: str | None = None  # 턴 결과 라벨. 무응답이면 None
    noResponse: bool = False

    @field_validator("slot")
    @classmethod
    def _slot_known(cls, value: str) -> str:
        if value not in TIME_SLOTS:
            raise ValueError("시각 값이 5개 중 하나가 아닙니다")
        return value


class DailySummaryRequest(BaseModel):
    """하루 요약 요청 양식 (S10-1). 판정 기준(D22·D12)은 아직 정해지지 않았다."""

    wardId: str = Field(min_length=1, max_length=64)
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    slots: list[SlotSummary] = Field(max_length=len(TIME_SLOTS))
