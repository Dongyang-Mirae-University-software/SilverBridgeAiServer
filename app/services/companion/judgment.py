from __future__ import annotations

from dataclasses import dataclass, field

from app.core.config import get_settings

# 부정 라벨 (아키텍처 08: 턴 라벨 중 슬픔·화남·우울이 부정)
NEGATIVE_LABELS = frozenset({"슬픔", "화남", "우울"})
# 비율이 너무 작은 차이에 민감하지 않도록 하는 최소 차이 (비율 0.2 이상 차이가 나야 변화로 본다)
MIN_RATIO_GAP = 0.2


@dataclass
class DayRecord:
    """하루치 기록. labels 는 그날 턴 라벨 목록 (무응답 턴은 None), riskCount 는 위험 플래그가 켜진 턴 수."""

    date: str
    labels: list[str | None] = field(default_factory=list)
    riskCount: int = 0


def _answered(days: list[DayRecord]) -> list[str]:
    return [label for day in days for label in day.labels if label]


def negative_ratio(days: list[DayRecord]) -> float | None:
    """답한 턴 중 부정 라벨 비율. 답한 턴이 없으면 None."""
    answered = _answered(days)
    if not answered:
        return None
    return sum(1 for label in answered if label in NEGATIVE_LABELS) / len(answered)


def baseline_ready(baseline: list[DayRecord]) -> bool:
    """기준선 판정을 시작할 수 있는지 (S10-2). 기록이 있는 날이 최소 일수 이상이어야 한다."""
    days_with_data = sum(1 for day in baseline if any(label for label in day.labels))
    return days_with_data >= get_settings().companion_baseline_min_days


def judge(baseline: list[DayRecord], recent: list[DayRecord]) -> dict:
    """기준선과 최근 기록을 비교해 알림 등급을 낸다 (S10-2, S10-3, S10-4).

    - 기준선 부족: 판정하지 않는다 (alertLevel None, state BASELINE_INSUFFICIENT)
    - 위험 플래그가 켜진 날이 있으면 URGENT (risk true)
    - 부정 신호 비율이 기준선의 설정 배수 이상 올랐으면 NOTICE
    - 그 밖에는 알림 없음 (INFO: 기록만 남긴다)

    사용자 결정(2026-11-02): 알림 등급은 2단계 (URGENT·NOTICE 는 알림, INFO 는 알림 없이 기록).
    """
    if not baseline_ready(baseline):
        return {"state": "BASELINE_INSUFFICIENT", "risk": False, "alertLevel": None,
                "baselineRatio": None, "recentRatio": None}
    if any(day.riskCount for day in recent):
        return {"state": "URGENT", "risk": True, "alertLevel": "URGENT",
                "baselineRatio": negative_ratio(baseline), "recentRatio": negative_ratio(recent)}
    base = negative_ratio(baseline) or 0.0
    now = negative_ratio(recent)
    ratio_limit = get_settings().companion_baseline_ratio
    changed = now is not None and now > 0 and now >= ratio_limit * base and now - base >= MIN_RATIO_GAP
    state = "NOTICE" if changed else "NORMAL"
    return {"state": state, "risk": False, "alertLevel": "NOTICE" if changed else "INFO",
            "baselineRatio": round(base, 4), "recentRatio": None if now is None else round(now, 4)}
