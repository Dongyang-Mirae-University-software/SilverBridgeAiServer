from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class TurnRecord:
    """턴 결과만 보관한다. 답변 원문은 여기에 두지 않는다 (결정: 원문은 백엔드에도 보내지 않는다)."""

    turnNo: int
    label: str | None  # 무응답 턴은 None
    source: str


@dataclass
class Dialogue:
    dialogueId: str
    wardId: str
    sessionId: str
    slot: str
    question: str
    maxTurns: int
    questionId: str | None = None  # 백엔드가 질문마다 발급한 ID. 결과에 그대로 되돌려 준다 (노션 요구)
    started_at: float = field(default_factory=time.monotonic)
    turns: list[TurnRecord] = field(default_factory=list)
    ended: bool = False
    endReason: str | None = None
    endedAt: str | None = None  # 대화가 끝난 시각 (UTC ISO 문자열)
    riskSeen: bool = False  # 이 대화 안에서 위험 플래그가 한 번이라도 켜졌는지
    resultSent: bool = False  # 대화 결과를 백엔드로 보냈는지 (한 번만 보낸다)
    lastSignals: dict | None = None  # 직전 턴의 7개 신호. 다음 턴 응답 프롬프트에 쓴다 (S7-1)
    noResponseStreak: int = 0  # 연속 무응답 횟수 (D15). 무응답 턴이 아니면 0으로 돌아간다

    @property
    def turnNo(self) -> int:
        return len(self.turns)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DialogueStore:
    """메모리 세션 저장소 (D2 결정: AI 메모리. 서버를 재시작하면 대화는 사라진다). 동시 요청은 lock 으로 막는다."""

    def __init__(self) -> None:
        self._items: dict[str, Dialogue] = {}
        self._lock = threading.Lock()

    def create(self, ward_id: str, session_id: str, slot: str, question: str, max_turns: int,
               question_id: str | None = None) -> Dialogue:
        d = Dialogue(uuid.uuid4().hex, ward_id, session_id, slot, question, max_turns, question_id)
        with self._lock:
            self._items[d.dialogueId] = d
        return d

    def get(self, dialogue_id: str) -> Dialogue | None:
        with self._lock:
            return self._items.get(dialogue_id)

    def add_turn(self, dialogue_id: str, label: str | None, source: str, no_response_limit: int) -> Dialogue | None:
        with self._lock:
            d = self._items.get(dialogue_id)
            if d is None or d.ended:
                return d
            d.turns.append(TurnRecord(d.turnNo + 1, label, source))
            d.noResponseStreak = d.noResponseStreak + 1 if source == "NO_RESPONSE" else 0
            if d.noResponseStreak >= no_response_limit:
                d.ended, d.endReason = True, "연속 무응답"  # D15
            elif d.turnNo >= d.maxTurns:
                d.ended, d.endReason = True, "최대 턴 수"
            if d.ended:
                d.endedAt = _now_iso()
            return d

    def mark_risk(self, dialogue_id: str) -> None:
        with self._lock:
            d = self._items.get(dialogue_id)
            if d is not None:
                d.riskSeen = True

    def set_last_signals(self, dialogue_id: str, signals: dict | None) -> None:
        with self._lock:
            d = self._items.get(dialogue_id)
            if d is not None:
                d.lastSignals = signals

    def finish(self, dialogue_id: str, reason: str) -> Dialogue | None:
        with self._lock:
            d = self._items.get(dialogue_id)
            if d is not None and not d.ended:
                d.ended, d.endReason, d.endedAt = True, reason, _now_iso()
            return d

    def delete_ward(self, ward_id: str) -> int:
        """피보호자 탈퇴 시: 메모리에 있는 그 피보호자의 대화를 모두 지운다. 지운 개수를 돌려준다."""
        with self._lock:
            gone = [k for k, v in self._items.items() if v.wardId == ward_id]
            for k in gone:
                del self._items[k]
            return len(gone)

    def mark_result_sent(self, dialogue_id: str) -> bool:
        """결과 전송을 처음 표시하면 True. 이미 보냈으면 False (중복 전송 방지)."""
        with self._lock:
            d = self._items.get(dialogue_id)
            if d is None or d.resultSent or not d.ended:
                return False
            d.resultSent = True
            return True


store = DialogueStore()
