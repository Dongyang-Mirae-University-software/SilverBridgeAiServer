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
    face: str | None = None  # 표정 모델 라벨 (HSEmotion, 예: Happiness). 표정을 못 읽었으면 None
    summary: str | None = None  # 대답 한 줄 요약. 대답 원문은 저장하지 않는다
    needsCheck: bool = False  # 음성 인식 결과가 실제 단어가 아닐 수 있어 다시 확인한 턴


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
    last_active: float = field(default_factory=time.monotonic)  # 마지막으로 쓰인 시각. 오래 멈춘 대화를 지우는 기준
    turns: list[TurnRecord] = field(default_factory=list)
    ended: bool = False
    endReason: str | None = None
    endedAt: str | None = None  # 대화가 끝난 시각 (UTC ISO 문자열)
    riskSeen: bool = False  # 이 대화 안에서 위험 플래그가 한 번이라도 켜졌는지
    resultSent: bool = False  # 대화 결과를 백엔드로 보냈는지 (한 번만 보낸다)
    lastSignals: dict | None = None  # 직전 턴의 7개 신호. 다음 턴 응답 프롬프트에 쓴다 (S7-1)
    noResponseStreak: int = 0  # 연속 무응답 횟수 (D15). 무응답 턴이 아니면 0으로 돌아간다
    replies: list[str] = field(default_factory=list)  # 이 대화에서 AI가 낸 문장 (질문 포함). 같은 문장 반복을 막는다

    @property
    def turnNo(self) -> int:
        return len(self.turns)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 마지막 활동 뒤 이 시간(초)이 지나면 대화를 메모리에서 지운다 (30분). 끝난 대화도 같이 지운다.
IDLE_LIMIT_SEC = 30 * 60


class DialogueStore:
    """메모리 세션 저장소 (D2 결정: AI 메모리. 서버를 재시작하면 대화는 사라진다). 동시 요청은 lock 으로 막는다."""

    def __init__(self, idle_limit_sec: float = IDLE_LIMIT_SEC) -> None:
        self._items: dict[str, Dialogue] = {}
        self._lock = threading.Lock()
        self._idle_limit_sec = idle_limit_sec

    def _expire_locked(self, now: float) -> None:
        """lock 을 잡은 상태에서 부른다. 오래 멈춘 대화를 지운다 (메모리가 계속 늘어나지 않게)."""
        stale = [k for k, v in self._items.items() if now - v.last_active > self._idle_limit_sec]
        for k in stale:
            del self._items[k]

    def create(self, ward_id: str, session_id: str, slot: str, question: str, max_turns: int,
               question_id: str | None = None) -> Dialogue:
        d = Dialogue(uuid.uuid4().hex, ward_id, session_id, slot, question, max_turns, question_id)
        d.replies.append(question)  # 질문도 AI가 한 말이므로 같은 문장이 다시 나오지 않게 한다
        with self._lock:
            self._expire_locked(time.monotonic())
            self._items[d.dialogueId] = d
        return d

    def get(self, dialogue_id: str) -> Dialogue | None:
        with self._lock:
            return self._items.get(dialogue_id)

    def add_turn(self, dialogue_id: str, label: str | None, source: str, no_response_limit: int,
                 face: str | None = None, summary: str | None = None, needs_check: bool = False) -> Dialogue | None:
        with self._lock:
            d = self._items.get(dialogue_id)
            if d is None or d.ended:
                return d
            d.last_active = time.monotonic()
            d.turns.append(TurnRecord(d.turnNo + 1, label, source, face, summary, needs_check))
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
                d.last_active = time.monotonic()

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
                d.last_active = time.monotonic()
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
