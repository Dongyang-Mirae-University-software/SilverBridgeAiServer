"""Phase 1: 지금 시각에 맞는 안부 질문을 고른다.

- 기준 시각은 9·12·15·18·21시다.
- 보호자가 설정한 시각이 있으면 그것을 쓰고, 없으면 COMPANION_DEFAULT_SLOTS 기본값을 쓴다.
- 질문은 questions.json 의 시간대 세트에서 고른다. 같은 날 같은 시각에는 같은 질문을 쓴다 (날짜를 시드로 씀).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

QUESTIONS = Path(__file__).resolve().parent / "questions.json"
DEFAULT_SLOTS = ["09:00", "12:00", "15:00", "18:00", "21:00"]
TZ = ZoneInfo("Asia/Seoul")


def load_slots() -> dict[str, list[str]]:
    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    return {s["time"]: s["questions"] for s in data["slots"]}


def parse_slots(raw: str | None) -> list[str]:
    """COMPANION_DEFAULT_SLOTS 같은 'HH:MM,HH:MM' 문자열을 목록으로 바꾼다."""
    if not raw:
        return DEFAULT_SLOTS
    return [x.strip() for x in raw.split(",") if x.strip()]


def due_slot(now: datetime, slots: list[str]) -> str | None:
    """지금 시각이 어느 안부 시각인지 본다. 시각 기준 ±30분 안이면 그 시각을 돌려준다."""
    for s in slots:
        h, m = map(int, s.split(":"))
        target = datetime.combine(now.date(), time(h, m), tzinfo=TZ)
        if abs((now - target).total_seconds()) <= 30 * 60:
            return s
    return None


def pick_question(slot: str, day: datetime, bank: dict[str, list[str]]) -> str:
    """같은 날·같은 시각에는 항상 같은 질문을 고른다 (새로고침해도 바뀌지 않게)."""
    options = bank.get(slot)
    if not options:
        raise KeyError(f"질문 세트에 {slot} 시각이 없습니다")
    key = f"{day.date().isoformat()}|{slot}".encode("utf-8")
    idx = int(hashlib.sha256(key).hexdigest(), 16) % len(options)
    return options[idx]


def question_for(now: datetime, raw_slots: str | None = None) -> str | None:
    """지금 물어볼 질문을 돌려준다. 안부 시각이 아니면 None."""
    slot = due_slot(now, parse_slots(raw_slots))
    if slot is None:
        return None
    return pick_question(slot, now, load_slots())


def _self_test() -> None:
    bank = load_slots()
    day = datetime(2026, 10, 25, 9, 5, tzinfo=TZ)
    # 시각 판정
    assert due_slot(datetime(2026, 10, 25, 9, 0, tzinfo=TZ), DEFAULT_SLOTS) == "09:00"
    assert due_slot(datetime(2026, 10, 25, 9, 25, tzinfo=TZ), DEFAULT_SLOTS) == "09:00"
    assert due_slot(datetime(2026, 10, 25, 10, 30, tzinfo=TZ), DEFAULT_SLOTS) is None
    # 보호자 설정 시각
    assert due_slot(datetime(2026, 10, 25, 8, 0, tzinfo=TZ), parse_slots("08:00,20:00")) == "08:00"
    assert parse_slots(None) == DEFAULT_SLOTS
    # 같은 날·같은 시각은 항상 같은 질문
    q1 = pick_question("09:00", day, bank)
    q2 = pick_question("09:00", day, bank)
    assert q1 == q2 and q1 in bank["09:00"]
    # 안부 시각이 아니면 질문 없음
    assert question_for(datetime(2026, 10, 25, 10, 30, tzinfo=TZ)) is None
    print("시각 선택 검사 통과")
    print("09:05 질문 예:", question_for(day))


if __name__ == "__main__":
    _self_test()
