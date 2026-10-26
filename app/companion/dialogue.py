"""대화형 말벗 엔진 (계획서 Phase 5 변경: 질문 하나 → 여러 턴 대화, 최대 2분).

흐름
- 보호자가 정한 시각의 질문으로 대화를 연다.
- 피보호자가 대답할 때마다: 텍스트 감정 분석 → 그 턴을 저장 → AI가 공감 한 문장과 꼬리질문 한 문장을 만든다.
- 2분이 지나거나, 최대 턴 수에 닿거나, 피보호자가 끝내면 대화가 끝난다. 끝난 뒤 턴별 결과를 묶어 세션 결과를 낸다.

이 모듈은 저장·모델 호출을 직접 하지 않는다. 호출하는 쪽이 analyze, persist, generate 를 넘긴다 (시험하기 쉽게).
"""

from __future__ import annotations

import json
from collections import Counter
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

import generate_questions as gq

TIME_LIMIT_SEC = 120  # 대화 한 번의 상한 (사용자 결정: 2분을 넘기지 않음)
MAX_TURNS = 8  # 꼬리질문 최대 턴 수 (사용자 확정, 2026-10-25)
REPLY_MAX_CHARS = 90
BANNED_ASSUMPTION = ("지난", "어제", "예전", "옛날", "여행")  # 과거 사실을 가정하지 않는다
ENDING = re.compile(r"(요|죠|나요|까요|세요|시나요|셨어요|셨나요|신가요|습니까)\?$")
FALLBACK_REPLIES = [
    "말씀해 주셔서 고맙습니다. 조금 더 자세히 들려주실 수 있을까요?",
    "그러셨군요. 그때 기분은 어떠셨나요?",
    "좋은 이야기 들려주셔서 감사합니다. 요즘 가장 즐거운 일은 무엇인가요?",
]


def check_reply(text: str) -> list[str]:
    """공감 문장 + 꼬리질문 한 덩어리가 규칙을 지키는지 본다. 문제 목록을 돌려준다 (빈 목록이면 통과)."""
    problems = []
    t = text.strip()
    if t.count("?") != 1 or not t.endswith("?"):
        problems.append("물음표는 끝에 한 번만")
    if len(t) > REPLY_MAX_CHARS:
        problems.append(f"{REPLY_MAX_CHARS}자 초과")
    sentences = [s for s in re.split(r"(?<=[.!])\s*", t) if s]
    if len(sentences) < 2:
        problems.append("공감 문장과 질문 문장이 모두 필요함")
    last = sentences[-1] if sentences else t
    if not ENDING.search(last):
        problems.append("질문 어미가 존댓말이 아님")
    if gq.has_english(t):
        problems.append("영어 포함")
    if gq.EMOJI.search(t):
        problems.append("이모지 포함")
    if any(b in t for b in gq.BANNED):
        problems.append("의료 관련 표현")
    if any(b in t for b in BANNED_ASSUMPTION):
        problems.append("과거 사실 가정")
    return problems


def _ollama(prompt: str, temperature: float) -> str:
    body = json.dumps({
        "model": gq.MODEL, "system": gq.SYSTEM, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature, "num_predict": 120},
    }).encode("utf-8")
    req = urllib.request.Request(gq.OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as res:
        return json.loads(res.read().decode("utf-8"))["response"]


def tidy(raw: str) -> str:
    """모델 출력을 한 덩어리로 다듬는다: 이모지·괄호 제거, 중복 물음표 정리, 첫 질문 뒤는 자른다. 느낌표는 남긴다."""
    one = " ".join(raw.split())
    one = gq.EMOJI.sub("", one)
    one = re.sub(r"\(\?\)|\?\s*\?+", "?", one)
    one = re.sub(r"\([^)]*\)", "", one)
    if "?" in one:
        one = one[: one.index("?") + 1]
    return re.sub(r"\s+", " ", one).strip()


def generate_reply(question: str, answer: str, turn_index: int) -> dict:
    """공감 + 꼬리질문을 만든다. 규칙을 통과하면 model, 아니면 fallback 으로 표시한다."""
    prompt = (
        f"질문: {question}\n어르신 대답: {answer[:200]}\n"
        "위 대답에 짧게 공감하는 문장 한 개를 쓰고, 이어서 자연스럽게 물어볼 질문 한 개를 쓰세요. "
        "질문은 한 개만, 물음표로 끝내세요. 이모지와 괄호는 쓰지 마세요. 존댓말을 쓰고, 과거 일을 가정하지 마세요. 두 문장만 출력하세요."
    )
    for temp in (0.6, 0.8, 1.0):
        try:
            raw = _ollama(prompt, temp).strip()
        except Exception:
            return {"reply": FALLBACK_REPLIES[turn_index % len(FALLBACK_REPLIES)], "source": "fallback"}
        raw = tidy(raw)
        if raw and not check_reply(raw):
            return {"reply": raw, "source": "model"}
    return {"reply": FALLBACK_REPLIES[turn_index % len(FALLBACK_REPLIES)], "source": "fallback"}


def _decided(labels: list[str]) -> list[str]:
    """결과가 정해진 턴만 센다 (분석 실패·판단 불가는 빼고)."""
    return [x for x in labels if x and x not in ("분석 실패", "판단 불가")]


def combine_label(labels: list[str]) -> str:
    """결과 라벨: 가장 많이 나온 결과(1위). 동점 1위 중에 확인 필요가 있으면 확인 필요, 없으면 판단 보류.
    정해진 턴이 하나도 없으면 판단 불가. (2026-10-25: 차이 기준 규칙은 빼고 1위로 정함)"""
    decided = _decided(labels)
    if not decided:
        return "판단 불가"
    counts = Counter(decided)
    top_n = max(counts.values())
    leaders = [k for k, v in counts.items() if v == top_n]
    if len(leaders) == 1:
        return leaders[0]
    return "확인 필요" if "확인 필요" in leaders else "판단 보류"


def summarize(labels: list[str]) -> str:
    """모든 턴의 결과별 개수와 비율: '기쁨 3회 (50%) · 확인 필요 1회 (17%) · 판단 불가 1회 (17%) ...' (많은 순)."""
    shown = [x for x in labels if x]
    if not shown:
        return ""
    n = len(shown)
    return " · ".join(f"{k} {v}회 ({round(100 * v / n)}%)" for k, v in Counter(shown).most_common())


def distribution(labels: list[str]) -> dict:
    """결과별 비율(%). 많이 나온 순서대로."""
    decided = _decided(labels)
    if not decided:
        return {}
    counts = Counter(decided)
    n = len(decided)
    return {k: round(100 * v / n) for k, v in counts.most_common()}


@dataclass
class Dialogue:
    dialogue_id: str
    ward_id: str
    session_id: str
    slot: str
    question: str
    analyze: Callable[[str, Optional[dict]], dict]
    persist: Callable[[dict], None]
    generate: Callable[[str, str, int], dict]
    clock: Callable[[], float] = time.monotonic
    started_at: float = 0.0
    turns: list[dict] = field(default_factory=list)
    ended: bool = False
    end_reason: Optional[str] = None

    def __post_init__(self) -> None:
        self.started_at = self.clock()

    def seconds_left(self) -> int:
        return max(0, int(TIME_LIMIT_SEC - (self.clock() - self.started_at)))

    def _end(self, reason: str) -> None:
        self.ended = True
        self.end_reason = reason

    def reply(self, text: str, face: Optional[dict] = None) -> dict:
        """피보호자의 대답 한 번을 받아 분석하고, 다음 말을 돌려준다."""
        if self.ended:
            raise ValueError("이미 끝난 대화입니다")
        over_time = self.seconds_left() <= 0
        question = self.turns[-1]["next"] if self.turns else self.question
        analysis = self.analyze(text, face)
        turn_no = len(self.turns) + 1
        last_turn = over_time or turn_no >= MAX_TURNS

        # 다음 말은 마지막 턴이 아닐 때만 만든다. 턴은 다 정해진 뒤 한 번만 저장한다.
        nxt = None if last_turn else self.generate(question, text, turn_no)
        row = {"turn": turn_no, "question": question, "text": text, **analysis,
               "next": nxt["reply"] if nxt else None, "replySource": nxt["source"] if nxt else None}
        self.turns.append(row)
        self.persist(row)  # 턴마다 바로 저장 (대화가 중간에 끊겨도 남도록)

        if last_turn:
            reason = "시간 초과" if over_time else "최대 턴 수"
            self._end(reason)
        return {"turn": turn_no, "label": analysis["label"], "reason": analysis["reason"],
                "reply": row["next"], "replySource": row["replySource"], "done": self.ended,
                "endReason": self.end_reason, "secondsLeft": self.seconds_left()}

    def finish(self, reason: str = "사용자 종료") -> dict:
        if not self.ended:
            self._end(reason)
        labels = [t["label"] for t in self.turns]
        return {"dialogueId": self.dialogue_id, "turnCount": len(self.turns), "endReason": self.end_reason,
                "sessionLabel": combine_label(labels), "summary": summarize(labels), "distribution": distribution(labels), "turns": [
                    {"turn": t["turn"], "label": t["label"], "reason": t["reason"]} for t in self.turns]}
