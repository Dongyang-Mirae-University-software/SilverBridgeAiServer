from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.core.config import get_settings
from app.services.companion.llm_client import LlmClient, LlmUnavailable

REPLY_PROMPT_VERSION = "reply-v2"  # 응답 프롬프트 파일을 고치면 올린다 (v2: 대답 속 낱말을 이어 가는 꼬리질문)
# 대답과 관계없는 문장을 거를 때 기준으로 삼지 않는 흔한 낱말의 앞 두 글자 (조사가 붙은 말도 걸리도록 앞 글자로 본다)
COMMON_STEMS = {"그냥", "오늘", "어제", "요즘", "정말", "너무", "조금", "많이", "그래", "있어", "없어",
                "했어", "같아", "좋아", "괜찮", "아니", "그럼", "이제", "다시"}
_PROMPT_DIR = Path(__file__).parent / "prompts"

# S6-2: 규칙 검사에 쓰는 목록. 한 곳에만 둔다 (C-23 결정 전까지 이 목록을 그대로 쓴다)
DISEASE_TERMS = ("우울증", "치매", "불안증", "공황장애", "조현병", "조울증")
PAST_TERMS = ("지난", "어제", "예전", "옛날", "여행")

# S6-3: 규칙을 통과하지 못하면 이 목록에서 고른다. 모든 문장이 규칙을 통과해야 한다 (시험으로 확인)
FALLBACK_FOLLOWUP = (
    "그러셨군요. 오늘 하루는 어떻게 보내고 계세요?",
    "말씀해 주셔서 고맙습니다. 요즘 식사는 잘 챙겨 드시고 계세요?",
    "그랬군요. 조금 더 들려주실 수 있을까요?",
)
FALLBACK_LAST = (
    "오늘도 이야기 나눠 주셔서 고맙습니다. 편안한 하루 보내세요.",
)


@dataclass
class ReplyResult:
    text: str
    isFollowUp: bool
    source: str  # LLM | FALLBACK
    failures: list[str]  # 마지막으로 걸린 규칙 이름 (없으면 빈 목록)


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[가-힣]+", text) if len(w) >= 2 and w[:2] not in COMMON_STEMS]


def check_reply(text: str, is_last_turn: bool, transcript: str = "") -> list[str]:
    """규칙을 하나씩 본다. 걸린 항목 이름 목록을 돌려준다. 빈 목록이면 통과."""
    failures: list[str] = []
    body = text.strip()
    if not body:
        return ["비어 있음"]
    if len(body) > get_settings().companion_reply_max_chars:
        failures.append("길이 초과")
    if any(term in body for term in DISEASE_TERMS):
        failures.append("질병명")
    if any(term in body for term in PAST_TERMS):
        failures.append("과거 가정")
    if not is_last_turn and not body.endswith("?"):
        failures.append("물음으로 끝나지 않음")
    # 꼬리질문은 어르신 대답에 나온 낱말 가운데 하나를 이어 받아야 한다 (2글자 앞부분으로 비교: 토스트 ↔ 토스트를)
    if not is_last_turn and transcript:
        words = _content_words(transcript)
        if words and not any(w[:2] in body for w in words):
            failures.append("대답과 무관")
    return failures


def _signal_line(prev_signals: dict | None) -> str:
    if not prev_signals:
        return "(직전 신호 없음: 첫 턴입니다)"
    return "직전 턴 신호: " + ", ".join(f"{k}={v}" for k, v in prev_signals.items())


def build_reply_prompt(question: str, transcript: str, prev_signals: dict | None, is_last_turn: bool) -> str:
    name = "reply-v1-last.txt" if is_last_turn else "reply-v2-followup.txt"
    template = (_PROMPT_DIR / name).read_text(encoding="utf-8")
    return (template
            .replace("{{PREV_SIGNALS}}", _signal_line(prev_signals))
            .replace("{{QUESTION}}", question)
            .replace("{{TRANSCRIPT}}", transcript))


def _fallback(turn_no: int, is_last_turn: bool) -> str:
    options = FALLBACK_LAST if is_last_turn else FALLBACK_FOLLOWUP
    return options[turn_no % len(options)]  # 턴마다 다른 문장이 나오도록 번갈아 쓴다


def generate_reply(
    llm: LlmClient,
    question: str,
    transcript: str,
    prev_signals: dict | None,
    is_last_turn: bool,
    turn_no: int,
) -> ReplyResult:
    """응답 문장을 만든다. (S6-3, D20)

    LLM 실패(연결 불가·시간 초과)와 규칙 위반을 모두 한 번씩 더 시도한다 (retry 설정, 기본 1회).
    그래도 안 되면 미리 정한 문장을 쓴다. 대화는 끊기지 않는다.
    """
    s = get_settings()
    prompt = build_reply_prompt(question, transcript, prev_signals, is_last_turn)
    failures: list[str] = []
    for _ in range(1 + s.companion_llm_retry):
        try:
            raw = llm.generate(prompt, temperature=s.companion_llm_temperature,
                               num_predict=s.companion_llm_num_predict, timeout=s.companion_llm_timeout_sec)
        except LlmUnavailable:
            failures = ["LLM 실패"]
            continue
        text = re.sub(r"\s+", " ", raw).strip()
        failures = check_reply(text, is_last_turn, transcript)
        if not failures:
            return ReplyResult(text=text, isFollowUp=not is_last_turn, source="LLM", failures=[])
    return ReplyResult(text=_fallback(turn_no, is_last_turn), isFollowUp=not is_last_turn,
                       source="FALLBACK", failures=failures)


def no_response_reply(turn_no: int, is_last_turn: bool) -> ReplyResult:
    """어르신이 답하지 않은 턴의 응답. LLM을 부르지 않고 미리 정한 문장을 쓴다 (S7-8)."""
    return ReplyResult(text=_fallback(turn_no, is_last_turn), isFollowUp=not is_last_turn,
                       source="FALLBACK", failures=["무응답"])
