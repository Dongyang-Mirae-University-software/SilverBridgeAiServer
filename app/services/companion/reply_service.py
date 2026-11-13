from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from app.core.config import get_settings
from app.services.companion.llm_client import REPLY_ADAPTER, LlmClient, LlmUnavailable
from app.services.companion.risk import body_emergency_only

logger = logging.getLogger(__name__)

REPLY_PROMPT_VERSION = "reply-v2"  # 응답 프롬프트 파일을 고치면 올린다 (v2: 대답 속 낱말을 이어 가는 꼬리질문)
# 대답과 관계없는 문장을 거를 때 기준으로 삼지 않는 흔한 낱말의 앞 두 글자 (조사가 붙은 말도 걸리도록 앞 글자로 본다)
COMMON_STEMS = {"그냥", "오늘", "어제", "요즘", "정말", "너무", "조금", "많이", "그래", "있어", "없어",
                "했어", "같아", "좋아", "괜찮", "아니", "그럼", "이제", "다시"}
_PROMPT_DIR = Path(__file__).parent / "prompts"

# S6-2: 규칙 검사에 쓰는 목록. 한 곳에만 둔다 (C-23 결정 전까지 이 목록을 그대로 쓴다)
DISEASE_TERMS = ("우울증", "치매", "불안증", "공황장애", "조현병", "조울증")
PAST_TERMS = ("지난", "어제", "예전", "옛날", "여행")

# S6-3: 규칙을 통과하지 못하면 이 목록에서 고른다. 모든 문장이 규칙을 통과해야 한다 (시험으로 확인)
# 11/11: 대체 문장은 어르신 말을 받아 주기만 하고 화제를 바꾸지 않는다 (식사·하루 일과 질문 금지).
#   힘든 이야기에 "요즘 식사는 잘 챙겨 드시고 계세요?"가 나간 일이 있었다 - '대답과 무관'에 걸린 모델 대답이 이 문장으로 바뀌었다
FALLBACK_FOLLOWUP = (
    "그러셨군요. 조금 더 들려주실 수 있을까요?",
    "말씀해 주셔서 고맙습니다. 그 이야기 조금 더 해 주실 수 있으세요?",
    "네, 잘 들었어요. 더 하고 싶으신 말씀 있으세요?",  # 11/12 M-1: 짧은 대답에도 맞게 시점(그때)을 뺐다
)
# 그 턴의 글 기분이 부정(슬픔·화남·우울)이면 이 목록에서 고른다 (11/11)
FALLBACK_EMPATHY = (
    "그러셨군요. 마음이 많이 쓰이셨겠어요. 조금 더 이야기해 주실 수 있을까요?",
    "많이 속상하셨겠어요. 어떤 점이 가장 마음에 걸리세요?",
    "힘든 이야기 꺼내 주셔서 고맙습니다. 제가 더 들어 드려도 될까요?",
)
NEGATIVE_TURN_LABELS = ("슬픔", "화남", "우울")
# 이 글 기분의 턴은 '대답과 무관' 검사를 하지 않는다 (11/12, COMPANION_REPLY_RELAX_CALM).
#   DGX 측정(CareCall 200): '대답과 무관'만으로 대체된 62개 중 약 50개가 멀쩡한 대답이었다.
#   힘든 이야기(93)는 풀면 살아난 대답의 절반이 질문 주제에 끌려간 오해라 부정 기분 턴은 그대로 둔다
RELAX_TURN_LABELS = ("평온", "기쁨")
# '권유' 규칙: 꼬리질문 대신 무엇을 하라고 권하는 문장을 거른다 (11/12).
#   11/12 점검 M-1: 처음 정규식("산책·운동·음악 … 어떠세요")은 주제를 이어 가는 물음("요즘 운동은 어떠세요?",
#   "산책 다녀오시니 기분은 어떠세요?", "말씀해 보세요?")까지 걸러서, 권하는 꼴만 보도록 좁혔다:
#   ① 해·드셔·들어·가 … + 보시는/보시면/보세요/보는 건 (단 '말씀해 보세요'는 이야기를 청하는 말이라 뺀다)
#   ② 권유 낱말 바로 뒤 어떠세요 ("가벼운 산책 어떠세요?" - 조사 '은·는'이 붙은 물음은 통과)
#      스트레칭·명상, '가벼운·간단한 …'은 어르신이 먼저 꺼내는 주제가 드물어 예전처럼 넓게 본다 (DGX 대조 11/12)
#   ③ 하시는·드시는 건 어떠세요  ④ 하시면 좋아요·좋겠·좋을 것 ("하시면 좋을까요?"는 물음이라 통과)
ADVICE_PATTERN = re.compile(
    r"(?:(?<!말씀)해|드셔|들어|맞아|걸어|나가|가|연락해|시작해)\s?보(?:시는|시면|시길|세요|시죠|실래요|는\s?건|는\s?거)"
    r"|(?:스트레칭|산책|걷기|운동|명상|음악)\s?(?:한\s?번\s?)?(?:어떠신가요|어떨까요|어떠세요)"
    r"|(?:스트레칭|명상|가벼운|간단한)[^.?!]*(?:어떠신가요|어떨까요|어떠세요)"
    r"|(?:하|드)시는\s?건\s?(?:어떠신가요|어떨까요|어떠세요)"
    r"|하시면\s?좋(?:아요|습니다|겠|을\s?것|을\s?거)"
)
# 어르신이 아무 말도 하지 않은 턴(무응답)의 문장 (11/12 점검 H-1). 말을 들은 것처럼 받지 않고, 편하게 다시 말을 건넨다.
# 무응답 턴에는 대화를 다시 여는 질문이 화제 전환이 아니다
FALLBACK_NO_RESPONSE = (
    "천천히 생각하셔도 괜찮아요. 편하실 때 말씀해 주시겠어요?",
    "제 말이 잘 들리셨을까요? 편하실 때 한 번 더 말씀해 주시겠어요?",
    "괜찮으시면 지금 기분이 어떠신지 들려주시겠어요?",
)
# 어르신이 말을 피하는 턴의 문장 (11/12, avoidance.is_avoidance). 캐묻지 않고 물러나되, 대화는 이어 가도록 물음으로 끝낸다.
#   "조금 더 들려주실 수 있을까요?"(받아 주는 대체 문장)·"특별히 걱정되는 일이 있으신가요?"(cc7)는 더 말하라고 캐묻게 됐다
FALLBACK_AVOID = (
    "괜찮아요, 말씀하고 싶지 않으시면 안 하셔도 돼요. 지금 마음은 좀 편안하세요?",
    "네, 알겠어요. 편하실 때 말씀해 주세요. 오늘 몸은 괜찮으세요?",
)
# 검수 목록에 남는 대체 문장 버전. 대체 문장 목록을 고치면 올린다 (v2: 11/11~12 화제 전환 제거·공감·무응답 목록)
FALLBACK_VERSION = "fallback-v3"  # v3: 말을 피하는 턴 문장 (11/12)
FALLBACK_LAST = (
    "오늘도 이야기 나눠 주셔서 고맙습니다. 편안한 하루 보내세요.",
)
# 위험 신호가 켜진 턴의 응답: 평소 규칙(꼬리질문 등)을 따르지 않고 안내 문장으로 답한다 (내용 확인 필요)
SAFETY_REPLY = (
    "많이 힘드셨겠어요. 지금 마음이 많이 무거우신 것 같아요. 혼자 견디지 마시고 가까운 가족이나 "
    "주변 분께 바로 연락해 주세요. 급하시면 119에 전화하실 수 있어요."
)

# 몸 응급 낱말(숨·가슴·쓰러짐·뇌졸중 징후)만 걸린 위험 턴의 안내 (11/12 영향 범위 점검 Q-1, COMPANION_BODY_SAFETY_REPLY).
#   SAFETY_REPLY("마음이 많이 무거우신 것 같아요")는 "혀가 굳어서 말을 못 하겠어" 같은 몸 응급에 맞지 않았다
BODY_SAFETY_REPLY = (
    "많이 놀라셨겠어요. 몸이 많이 불편하시거나 위급한 상황이면 바로 119에 전화하시고, 가까운 가족에게도 알려 주세요."
)


def safety_reply(transcript: str) -> str:
    """위험 턴의 안내 문장. 몸 응급 낱말만 걸렸으면 몸 응급용, 아니면 기존 안내 (스위치를 끄면 늘 기존 안내)."""
    if get_settings().companion_body_safety_reply and body_emergency_only(transcript):
        return BODY_SAFETY_REPLY
    return SAFETY_REPLY


@dataclass
class ReplyResult:
    text: str
    isFollowUp: bool
    source: str  # LLM | FALLBACK
    failures: list[str]  # 마지막으로 걸린 규칙 이름 (없으면 빈 목록)


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[가-힣]+", text) if len(w) >= 2 and w[:2] not in COMMON_STEMS]


def is_relaxed_mood(text_mood: dict | None) -> bool:
    """글 기분이 쓸 수 있는 결과이고 평온·기쁨인가 ('대답과 무관' 검사를 빼는 턴)."""
    return bool(text_mood and text_mood.get("usable") and text_mood.get("turnLabel") in RELAX_TURN_LABELS)


def check_reply(text: str, is_last_turn: bool, transcript: str = "", text_mood: dict | None = None) -> list[str]:
    """규칙을 하나씩 본다. 걸린 항목 이름 목록을 돌려준다. 빈 목록이면 통과.

    text_mood: 그 턴의 글 기분 (pipeline 의 textMood). COMPANION_REPLY_RELAX_CALM 이 켜져 있으면
    평온·기쁨 턴은 '대답과 무관'을 보지 않고, 꼬리질문 턴 전체에 '권유'를 본다 (11/12).
    없거나 부정·판단 불가면 '대답과 무관'은 전과 같다.
    """
    relax = get_settings().companion_reply_relax_calm
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
    if not is_last_turn and transcript and not (relax and is_relaxed_mood(text_mood)):
        words = _content_words(transcript)
        if words and not any(w[:2] in body for w in words):
            failures.append("대답과 무관")
    if relax and not is_last_turn and ADVICE_PATTERN.search(body):
        failures.append("권유")
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


def is_negative_mood(text_mood: dict | None) -> bool:
    """글 기분(턴 응답의 textMood)이 쓸 수 있는 결과이고 부정(슬픔·화남·우울)인가."""
    return bool(text_mood and text_mood.get("usable") and text_mood.get("turnLabel") in NEGATIVE_TURN_LABELS)


def fallback_options(is_last_turn: bool, negative: bool = False, no_response: bool = False,
                     avoid: bool = False) -> tuple[str, ...]:
    """대체 문장 목록. 마지막 턴은 마무리 문장, 무응답 턴은 다시 말을 건네는 문장,
    말을 피하는 턴은 물러나는 문장, 부정 기분 턴은 공감 문장, 그 밖은 받아 주는 문장."""
    if is_last_turn:
        return FALLBACK_LAST
    if no_response:
        return FALLBACK_NO_RESPONSE
    if avoid:
        return FALLBACK_AVOID
    return FALLBACK_EMPATHY if negative else FALLBACK_FOLLOWUP


def fallback_text(turn_no: int, is_last_turn: bool, negative: bool = False, no_response: bool = False,
                  avoid: bool = False) -> str:
    options = fallback_options(is_last_turn, negative, no_response, avoid)
    return options[turn_no % len(options)]  # 턴마다 다른 문장이 나오도록 번갈아 쓴다


def _fallback(turn_no: int, is_last_turn: bool) -> str:
    return fallback_text(turn_no, is_last_turn)


def _normalized(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum())


def reply_similarity(a: str, b: str) -> float:
    """두 문장이 얼마나 비슷한가 (0~1, 공백·문장부호를 뺀 글자 비교). 11/13 세션 203: 2번↔5번 대답 0.76, 다른 대답끼리 0.47 이하."""
    na, nb = _normalized(a), _normalized(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def is_similar_to_past(text: str, past_replies) -> bool:
    """이 대화의 지난 말벗 문장(여는 질문 포함) 가운데 COMPANION_REPLY_SIMILAR_RATIO 이상 비슷한 것이 있는가. 0이면 보지 않는다."""
    ratio = get_settings().companion_reply_similar_ratio
    if ratio <= 0:
        return False
    return any(reply_similarity(text, p) >= ratio for p in past_replies or ())


def _generate(llm: LlmClient, prompt: str, is_last_turn: bool, temperature: float | None = None) -> str:
    """꼬리질문 턴은 대답 어댑터(11/13부터 cc7, 이전 cc5)로 만든다 (11/10, COMPANION_REPLY_LORA). 프롬프트는 reply-v2 그대로다.

    어댑터는 꼬리질문 프롬프트로만 학습돼서 마지막 인사 턴은 기본 모델로 만든다.
    어댑터가 없으면(파일 없음, 어댑터 기능이 없는 LLM) 기본 모델로 만든다 (재희 동작).
    """
    s = get_settings()
    kwargs = {"temperature": s.companion_llm_temperature if temperature is None else temperature, "num_predict": s.companion_llm_num_predict,
              "timeout": s.companion_llm_timeout_sec}
    if s.companion_reply_lora and not is_last_turn:
        gen = getattr(llm, "generate_adapter", None)
        has = getattr(llm, "has_adapter", None)
        if gen is not None and (has is None or has(REPLY_ADAPTER)):
            return gen(prompt, adapter=REPLY_ADAPTER, **kwargs)
    return llm.generate(prompt, **kwargs)


def generate_reply(
    llm: LlmClient,
    question: str,
    transcript: str,
    prev_signals: dict | None,
    is_last_turn: bool,
    turn_no: int,
    text_mood: dict | None = None,
    past_replies=(),
) -> ReplyResult:
    """응답 문장을 만든다. (S6-3, D20)

    LLM 실패(연결 불가·시간 초과)와 규칙 위반을 COMPANION_LLM_RETRY 번 더 시도한다 (기본 0회 - 11/09 결정, 바로 대체 문장).
    그래도 안 되면 미리 정한 문장을 쓴다. 대화는 끊기지 않는다.
    text_mood 는 규칙 검사에 넘긴다 (평온·기쁨 턴 완화, 11/12). 없으면 전 규칙.
    past_replies: 이 대화에서 말벗이 이미 한 문장 (11/13). 규칙을 통과한 대답이 그 가운데 하나와 매우 비슷하면
    COMPANION_REPLY_REDO_TEMPERATURE 로 한 번 다시 만들고, 그래도 비슷하거나 규칙에 걸리면 대체 문장을 쓴다 (실패 이름 "반복").
    """
    s = get_settings()
    prompt = build_reply_prompt(question, transcript, prev_signals, is_last_turn)
    failures: list[str] = []
    for _ in range(1 + s.companion_llm_retry):
        try:
            raw = _generate(llm, prompt, is_last_turn)
        except LlmUnavailable:
            failures = ["LLM 실패"]
            continue
        text = re.sub(r"\s+", " ", raw).strip()
        failures = check_reply(text, is_last_turn, transcript, text_mood)
        if not failures and is_similar_to_past(text, past_replies):
            text, failures = _redo_once(llm, prompt, is_last_turn, transcript, text_mood, past_replies)
        if not failures:
            return ReplyResult(text=text, isFollowUp=not is_last_turn, source="LLM", failures=[])
    # 11/12 점검 L-2: 대체된 이유(규칙 이름)만 남긴다 - 대답·어르신 말 본문은 남기지 않는다
    logger.info("[COMPANION-REPLY-FALLBACK] failures=%s relaxed=%s last=%s",
                ",".join(failures), is_relaxed_mood(text_mood), is_last_turn)
    return ReplyResult(text=_fallback(turn_no, is_last_turn), isFollowUp=not is_last_turn,
                       source="FALLBACK", failures=failures)


def _redo_once(llm: LlmClient, prompt: str, is_last_turn: bool, transcript: str, text_mood: dict | None,
               past_replies) -> tuple[str, list[str]]:
    """지난 문장과 비슷한 대답을 한 번 다시 만든다 (같은 프롬프트, 온도만 올림). 반환: (문장, 실패 이름 목록)."""
    try:
        raw = _generate(llm, prompt, is_last_turn, temperature=get_settings().companion_reply_redo_temperature)
    except LlmUnavailable:
        return "", ["반복"]
    text = re.sub(r"\s+", " ", raw).strip()
    failures = check_reply(text, is_last_turn, transcript, text_mood)
    if not failures and is_similar_to_past(text, past_replies):
        failures = ["반복"]
    logger.info("[COMPANION-REPLY-REDO] ok=%s", not failures)  # 대답 본문은 남기지 않는다
    return text, failures


def no_response_reply(turn_no: int, is_last_turn: bool) -> ReplyResult:
    """어르신이 답하지 않은 턴의 응답. LLM을 부르지 않고 무응답 문장을 쓴다 (S7-8, 11/12 H-1)."""
    return ReplyResult(text=fallback_text(turn_no, is_last_turn, no_response=True), isFollowUp=not is_last_turn,
                       source="FALLBACK", failures=["무응답"])


def timeout_reply(turn_no: int, is_last_turn: bool) -> ReplyResult:
    """대답 생성이 턴 상한을 넘긴 턴의 응답. 어르신 말은 있으므로 받아 주는 문장을 쓴다 (11/12 H-1).

    부정 기분이면 pipeline 이 공감 문장으로 바꾼다. 실패 이름은 전과 같은 "무응답"이다(검수·로그 호환).
    """
    return ReplyResult(text=fallback_text(turn_no, is_last_turn), isFollowUp=not is_last_turn,
                       source="FALLBACK", failures=["무응답"])


def avoid_reply(turn_no: int, is_last_turn: bool) -> ReplyResult:
    """어르신이 말을 피한 턴의 응답. LLM을 부르지 않고 물러나는 문장을 쓴다 (마지막 턴은 마무리 문장, 11/12).

    위험 턴이면 pipeline 이 SAFETY_REPLY 로 덮어쓴다 (위험 판정이 우선)."""
    return ReplyResult(text=fallback_text(turn_no, is_last_turn, avoid=True), isFollowUp=not is_last_turn,
                       source="AVOID", failures=["말 피함"])
