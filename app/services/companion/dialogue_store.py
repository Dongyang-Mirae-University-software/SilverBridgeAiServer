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
    lowConfidenceStreak: int = 0  # 확신도가 낮아 되물은 턴의 연속 횟수 (11/09 점검 M-2). 다른 턴이면 0으로 돌아간다
    avoidCount: int = 0  # 이 대화에서 말을 피한 턴 수 (위험 턴 제외, 11/12). COMPANION_AVOID_END_COUNT 에 닿으면 마무리한다
    replies: list[str] = field(default_factory=list)  # 이 대화에서 AI가 낸 문장 (질문 포함). 같은 문장 반복을 막는다
    # 어르신이 지금 대답하는 말벗 문장 (11/13). 대답 프롬프트의 "질문:"에 넣는다. 처음은 여는 질문, 턴마다 나간 대답으로 바뀐다
    # (되묻기 턴은 바꾸지 않는다 - 다시 말한 대답은 그 전 질문에 대한 것이다). None 이면 여는 질문을 쓴다
    lastQuestion: str | None = None
    # 말벗이 화제를 정해 물은 문장 (11/13): 여는 질문과 모델이 만든 대답만. 되묻기·안전 안내·대체 문장은 넣지 않는다.
    # v3 프롬프트의 '앞서 한 질문'에 쓴다 (create 에서 여는 질문을 넣는다)
    askedQuestions: list[str] = field(default_factory=list)

    @property
    def turnNo(self) -> int:
        return len(self.turns)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 마지막 활동 뒤 이 시간(초)이 지나면 대화를 메모리에서 지운다 (10분, BE 기준에 맞춤). 끝난 대화도 같이 지운다.
IDLE_LIMIT_SEC = 10 * 60


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
        d.askedQuestions.append(question)
        with self._lock:
            self._expire_locked(time.monotonic())
            self._items[d.dialogueId] = d
        return d

    def get(self, dialogue_id: str) -> Dialogue | None:
        with self._lock:
            return self._items.get(dialogue_id)

    def add_turn(self, dialogue_id: str, label: str | None, source: str, no_response_limit: int,
                 face: str | None = None, summary: str | None = None, needs_check: bool = False,
                 low_confidence: bool = False, low_confidence_limit: int = 0,
                 end_reason: str | None = None) -> Dialogue | None:
        """턴을 더하고 종료 규칙을 본다. end_reason: 다른 종료 규칙에 걸리지 않았을 때 이 이유로 끝낸다 (말 피함 마무리, 11/12)."""
        with self._lock:
            d = self._items.get(dialogue_id)
            if d is None or d.ended:
                return d
            d.last_active = time.monotonic()
            d.turns.append(TurnRecord(d.turnNo + 1, label, source, face, summary, needs_check))
            d.noResponseStreak = d.noResponseStreak + 1 if source == "NO_RESPONSE" else 0
            d.lowConfidenceStreak = d.lowConfidenceStreak + 1 if low_confidence else 0
            if d.noResponseStreak >= no_response_limit or (low_confidence_limit and d.lowConfidenceStreak >= low_confidence_limit):
                # 알아들을 수 있는 대답이 계속 없으면(TV·옆 사람 말소리 포함) 무응답 종료와 같은 이유 코드로 끝낸다
                d.ended, d.endReason = True, "NO_RESPONSE"  # D15 (백엔드 요청 결정 D: 영문 코드)
            elif d.turnNo >= d.maxTurns:
                d.ended, d.endReason = True, "MAX_TURNS"
            elif end_reason:
                d.ended, d.endReason = True, end_reason
            if d.ended:
                d.endedAt = _now_iso()
            return d

    def mark_risk(self, dialogue_id: str) -> None:
        with self._lock:
            d = self._items.get(dialogue_id)
            if d is not None:
                d.riskSeen = True
                d.last_active = time.monotonic()

    def mark_avoid(self, dialogue_id: str) -> int:
        """말을 피한 턴을 하나 센다. 센 뒤의 횟수를 돌려준다 (대화가 없으면 0)."""
        with self._lock:
            d = self._items.get(dialogue_id)
            if d is None:
                return 0
            d.avoidCount += 1
            d.last_active = time.monotonic()
            return d.avoidCount

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
