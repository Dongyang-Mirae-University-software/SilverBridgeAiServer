"""어제 대답 참고 (Phase 1 남은 항목, 2026-10-25).

원칙
- 보호자가 저장한 질문이 기본이다. 이 모듈은 '후속 질문 제안'만 만든다.
- 제안은 자동으로 발송되지 않는다. 보호자가 보고 골라서 저장한다.
- 어제 대답이 없으면 제안하지 않는다 (None).

흐름: 어제 같은 시각의 대답 찾기 -> EXAONE 으로 후속 질문 한 문장 만들기 -> 질문 규칙 검사 -> 제안
"""

from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate_questions import ask, check, clean  # noqa: E402

BANNED_ASSUMPTION = ("지난", "어제", "예전", "옛날", "여행")  # 과거 사실을 가정하지 않는다
PROMPT_RULE = (
    "어르신이 어제 '{prev}'라고 하셨습니다. 오늘 같은 시각에 드릴 짧은 안부 질문 한 문장을 만드세요. "
    "어제 말씀을 자연스럽게 이어 가되, 어제 일을 다시 캐묻거나 사실을 가정하지 마세요. "
    "존댓말, 물음표로 끝나는 질문 한 문장만 출력하세요."
)


def find_yesterday_answer(rows: list[dict], ward_id: str, slot: str, today: date) -> Optional[dict]:
    """어제 같은 피보호자·같은 시각에 저장된 대답을 찾는다. 여러 개면 가장 나중 것."""
    yesterday = (today - timedelta(days=1)).isoformat()
    hits = [r for r in rows if r.get("wardId") == ward_id and r.get("slot") == slot and r.get("date") == yesterday]
    return hits[-1] if hits else None


def suggest_followup(prev_text: str, slot: str) -> Optional[str]:
    """어제 대답을 바탕으로 후속 질문 제안 한 개를 돌려준다. 규칙을 통과하지 못하면 None."""
    import generate_questions as gq

    gq.SYSTEM = gq.SYSTEM + "\n6. 이모지, 괄호, 영어를 쓰지 않습니다."
    for temp in (0.6, 0.8, 1.0):
        q = clean(ask(PROMPT_RULE.format(prev=prev_text[:200]), temp))
        if q and not check(q) and not any(w in q for w in BANNED_ASSUMPTION):
            return q
    return None


def followup_for(rows: list[dict], ward_id: str, slot: str, today: date) -> Optional[dict]:
    """보호자 화면에 보여 줄 제안 묶음. 어제 대답이 없으면 None."""
    prev = find_yesterday_answer(rows, ward_id, slot, today)
    if prev is None:
        return None
    suggestion = suggest_followup(prev["text"], slot)
    return {
        "basedOn": {"date": prev["date"], "text": prev["text"]},
        "suggestion": suggestion,  # None 이면 제안 없음 (규칙 통과 실패)
        "note": "참고용 제안입니다. 자동으로 발송되지 않습니다.",
    }


def _self_test() -> None:
    today = date(2026, 10, 25)
    rows = [
        {"wardId": "W1", "slot": "09:00", "date": "2026-10-24", "text": "손주가 와서 같이 아침을 먹었어요"},
        {"wardId": "W1", "slot": "12:00", "date": "2026-10-24", "text": "점심은 먹었어요"},
        {"wardId": "W2", "slot": "09:00", "date": "2026-10-24", "text": "잘 잤어요"},
    ]
    hit = find_yesterday_answer(rows, "W1", "09:00", today)
    assert hit and hit["text"].startswith("손주"), hit
    assert find_yesterday_answer(rows, "W1", "15:00", today) is None
    assert find_yesterday_answer(rows, "W3", "09:00", today) is None
    print("어제 대답 찾기 검사 통과")
    res = followup_for(rows, "W1", "09:00", today)
    print("제안 예시:", res)


if __name__ == "__main__":
    _self_test()
