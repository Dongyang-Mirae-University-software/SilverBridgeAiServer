from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Slot = Literal["MORNING", "LUNCH", "DINNER", "BEDTIME"]
SLOTS: tuple[str, ...] = ("MORNING", "LUNCH", "DINNER", "BEDTIME")


class OcrDrug(BaseModel):
    """약 한 종. 성분·환자·병원·약국·금액 칸은 일부러 없다(성분은 확신 high로 틀려 뺐다 - 2026-10-22 측정)."""

    name: str
    perDose: float | None = None
    timesPerDay: int | None = None
    days: int | None = None
    slots: list[Slot] = Field(default_factory=list)
    timing: str | None = None
    confidence: Literal["high", "low"]


class MedicationOcrResponse(BaseModel):
    """BE 계약 그대로 - 공용 응답 래퍼({success, data})로 감싸지 않는다."""

    readable: bool
    reason: str | None = None
    drugs: list[OcrDrug] = Field(default_factory=list)


_NULLABLE_STR = {"type": ["string", "null"]}

# Claude 출력 강제용 JSON 스키마(output_config.format). 응답 모델과 같은 칸만 둔다.
CLAUDE_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "readable": {"type": "boolean"},
        "reason": _NULLABLE_STR,
        "drugs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "perDose": {"type": ["number", "null"]},
                    "timesPerDay": {"type": ["integer", "null"]},
                    "days": {"type": ["integer", "null"]},
                    "slots": {"type": "array", "items": {"type": "string", "enum": list(SLOTS)}},
                    "timing": _NULLABLE_STR,
                    "confidence": {"type": "string", "enum": ["high", "low"]},
                },
                "required": ["name", "perDose", "timesPerDay", "days", "slots", "timing", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["readable", "reason", "drugs"],
    "additionalProperties": False,
}
