"""AI가 만든 공감·꼬리질문 문장의 검수 목록 (관리자 검수 + AI 규칙 검사).

- 대화 중에는 관리자가 바로 볼 수 없으므로, 문장은 AI 규칙 검사를 통과하면 그 자리에서 쓰고 목록에 쌓는다.
- 관리자는 나중에 목록을 보고 승인(approved) 또는 반려(rejected)를 남긴다. 반려된 문장 유형은 규칙과 프롬프트를 고치는 데 쓴다.
- 어르신의 대답 원문은 저장하지 않는다. 질문과 AI 문장만 저장한다 (개인정보 최소화).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

STATUSES = ("pending", "approved", "rejected")


def make_item(question: str, reply: str, source: str, rule_problems: list[str], slot: str) -> dict:
    return {
        "id": uuid.uuid4().hex,
        "slot": slot,
        "question": question,
        "reply": reply,
        "source": source,  # model 또는 fallback
        "aiCheck": "통과" if not rule_problems else "확인 필요: " + ", ".join(rule_problems),
        "status": "pending",
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "reviewedAt": None,
    }


def list_items(rows: list[dict], status: Optional[str] = None) -> list[dict]:
    if status is None:
        return rows
    return [r for r in rows if r["status"] == status]


def decide(rows: list[dict], item_id: str, decision: str) -> Optional[dict]:
    """승인 또는 반려를 기록한다. 없는 항목이면 None."""
    if decision not in ("approved", "rejected"):
        raise ValueError("decision 은 approved 또는 rejected 여야 합니다")
    for r in rows:
        if r["id"] == item_id:
            r["status"] = decision
            r["reviewedAt"] = datetime.now().isoformat(timespec="seconds")
            return r
    return None
