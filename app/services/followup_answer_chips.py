"""되묻기 답 버튼(응답 suggestedQuestions + suggestedKind="answer", 2026-11-02 → 2026-11-09 단계 문답).

되묻기 턴에 어르신이 타이핑하지 않고 눌러서 답할 수 있게, 지금 질문 하나에 맞는 버튼 2~3개를 준다("네"/"아니요",
"오늘부터요"/"며칠 됐어요"/"오래됐어요"). 문장은 followup_steps 표의 것만 나간다 - 모델이 만들지 않는다.
- 버튼을 눌러 보낸 문장도 일반 문장과 똑같이 판정된다(응급·자해·위기는 단계 처리보다 먼저).
- 단계 상태(meta.followupStep)가 있는 되묻기 턴에만 준다. 응급 답(high·경고 박스)에는 주지 않는다.
  위험 신호 낱말이 있는 첫 턴(redFlag119, 119 한 줄이 함께 나간다)에도 위험 신호 질문의 "네"/"아니요"를 준다 -
  "네"는 규칙으로 바로 119 답이라 라우터에 기대지 않는다(2026-11-02 의 "칩 없음"은 그 칩이 라우터 판정에 기댔기 때문이다).
"""

from __future__ import annotations

from typing import Any

from app.services import followup_steps

KIND = "answer"


def answer_chips(result: dict[str, Any], message: str) -> list[str]:
    """이 답이 답 버튼을 줄 단계 되묻기 턴이면 지금 질문의 버튼, 아니면 빈 목록."""
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    if not meta.get("followupAsked"):
        return []
    if result.get("riskLevel") == "high" or result.get("emergencyWarning"):
        return []
    reply = result.get("reply")
    if not isinstance(reply, str) or not reply.strip() or not message:
        return []
    state = followup_steps.valid_state(meta.get("followupStep"))
    return followup_steps.chips(state) if state else []
