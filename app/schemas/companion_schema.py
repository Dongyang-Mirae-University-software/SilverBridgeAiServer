from __future__ import annotations

from pydantic import BaseModel, Field

# 시간대 (아키텍처 1절 · 명세 1번): 09·12·15·18·21시
TIME_SLOTS = ("09:00", "12:00", "15:00", "18:00", "21:00")


class CompanionStatus(BaseModel):
    """/companion/status 응답 (키 보호 확인용). 경로·키 값은 담지 않는다."""

    enabled: bool
    sessionLimitSec: int
    maxTurns: int
    fusionRule: str
