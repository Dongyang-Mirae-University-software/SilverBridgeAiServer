from __future__ import annotations

from collections import Counter

from app.schemas.companion_daily import SlotSummary


def summary_sentence(slots: list[SlotSummary]) -> str:
    """보호자에게 보여 줄 하루 요약 한 문장 (S10-5). 규칙으로 만들고 LLM은 쓰지 않는다.

    라벨은 "가장 많이 나온 기분"만 말하고, 무응답 수를 함께 적는다.
    """
    if not slots:
        return "오늘은 말벗 대화가 없었습니다."
    answered = [s for s in slots if not s.noResponse and s.label]
    no_reply = len(slots) - len([s for s in slots if not s.noResponse])
    if not answered:
        return f"오늘 {len(slots)}번 대화 중 답을 주시지 않은 경우가 {no_reply}번 있었습니다."
    top_label, _ = Counter(s.label for s in answered).most_common(1)[0]
    base = f"오늘 {len(slots)}번 대화 중 가장 많이 나온 기분은 '{top_label}'입니다."
    if no_reply:
        base += f" 답을 주시지 않은 경우는 {no_reply}번입니다."
    return base
