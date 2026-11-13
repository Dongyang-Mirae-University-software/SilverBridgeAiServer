from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Callable, Literal

from app.core.config import get_settings
from app.services.companion.answer_summary import summarize_answer
from app.services.companion.avoidance import is_avoidance
from app.services.companion.fusion import (
    HSEMOTION_TO_TURN,
    MOOD_CODES,
    TEXT_TO_TURN,
    UNDECIDED,
    face_mood_code,
    fuse_prob_add,
    fuse_turn,
)
from app.services.companion.llm_client import LlmClient
from app.services.companion.reply_service import (
    ReplyResult,
    safety_reply,
    avoid_reply,
    fallback_text,
    generate_reply,
    is_negative_mood,
    no_response_reply,
    timeout_reply,
)
from app.services.companion.risk import is_risk, keyword_risk, risk_probability
from app.services.companion.signals import extract_signals
from app.services.companion.transcript_check import check_transcript, confirm_reply
from app.services.companion.text_emotion_service import TextEmotionResult, TextEmotionUnavailable
from app.services.companion.timeouts import call_with_timeout

logger = logging.getLogger(__name__)

# 대답 출처·대체 이유를 숫자처럼 셀 수 있는 코드로만 남긴다 (11/13). 어르신 말·대답 원문은 남기지 않는다.
#   시간 초과와 규칙 탈락이 같은 대체 문장으로 나가 gosky 에서 비율을 볼 수 없었다
REASON_CODES = {
    "비어 있음": "RULE_EMPTY",
    "길이 초과": "RULE_LENGTH",
    "질병명": "RULE_DISEASE",
    "과거 가정": "RULE_PAST",
    "물음으로 끝나지 않음": "RULE_QUESTION",
    "대답과 무관": "RULE_UNRELATED",
    "권유": "RULE_ADVICE",
    "반복": "RULE_REPEAT",
    "LLM 실패": "LLM_FAIL",
    "말 피함": "AVOID",
}

InputKind = Literal["voice", "text"]

# 확신도가 낮을 때 되묻는 문장 (11/09). 후보 낱말을 넣지 않는다
LOW_CONFIDENCE_REPLY = "잘 알아듣지 못했어요. 한 번 더 말씀해 주시겠어요?"

# 받아쓰기 확신도가 낮아도 되묻지 않는 짧은 부정 대답의 낱말 (11/13, 세션 203 3번 "아니 없어"가 되묻기로 빠졌다).
# 짧은 말은 토큰이 적어 평균 로그확률이 낮게 나오기 쉽다. "네·응·어"는 소음에서 잘못 받아 적히기 쉬워 넣지 않는다
SHORT_ANSWER_WORDS = frozenset({
    "아니", "아니요", "아뇨", "아니야", "없어", "없어요", "없다", "없다고", "없다니까", "없는데",
    "별로", "그냥", "몰라", "몰라요", "괜찮아", "괜찮아요",
})
SHORT_ANSWER_MAX_CHARS = 12


def is_short_plain_answer(transcript: str) -> bool:
    """문장부호를 뺀 낱말이 모두 SHORT_ANSWER_WORDS 에 있고 공백 뺀 길이가 12자 이하인가."""
    words = "".join(ch if ch.isalnum() else " " for ch in transcript).split()
    return bool(words) and len("".join(words)) <= SHORT_ANSWER_MAX_CHARS and all(w in SHORT_ANSWER_WORDS for w in words)


def min_text_chars() -> int:
    """말을 기분 판단에 쓰는 최소 글자 수. 설정이 없으면 엔진에 맞춘다 (e900 2자 · ggara02 10자)."""
    s = get_settings()
    if s.companion_min_text_chars is not None:
        return s.companion_min_text_chars
    return 2 if s.companion_text_mood_engine == "e900" else 10


def analyze_turn_result(
    transcript: str,
    face: dict,
    classify_text: Callable[[str], TextEmotionResult],
    text_out: dict | None = None,
) -> dict:
    """말과 표정을 합쳐 턴 결과를 만든다.

    prob_add(11/10 기본): 글 기분 4코드 확률 + 표정 코드 쪽으로 가중치 (fuse_prob_add).
    plan41(재희): 아키텍처 08 결합 표 (fuse_turn). 글 결과에 4코드 확률이 없으면(GGARA02 등) prob_add 설정이어도 이 표를 쓴다.
    텍스트 모델을 못 쓰면 텍스트는 빠지고 표정만으로 판단한다.

    text_out: 주면 글 기분 분류 결과를 채운다(백엔드 테스트 계정 화면용 `textMood`, 2026-11-10). 반환값 모양은 그대로다.
      {"label": 모델 원래 라벨|None, "turnLabel": 턴 라벨로 바꾼 값|None, "confidence": 0~1|None, "usable": bool,
       "reason": None | "SHORT"(최소 글자 수 미만) | "UNAVAILABLE"(모델 없음·시간 초과)}
    """
    s = get_settings()
    text = transcript.strip()
    text_label = None
    text_usable = False
    text_probs = None
    text_reason = "SHORT"
    r = None
    if text and len(text) >= min_text_chars():
        text_reason = "UNAVAILABLE"  # 최소 글자 수 미만은 어차피 쓰지 않으므로 모델을 부르지 않는다 (L-2)
        try:
            r = call_with_timeout(classify_text, s.companion_text_timeout_sec, text)
        except (TextEmotionUnavailable, TimeoutError):  # 시간 초과면 말은 빼고 표정만으로 판단 (S9-1)
            r = None
        if r is not None:
            text_reason = None
            text_label = TEXT_TO_TURN.get(r.label)
            # 발화가 최소 글자 수 이상이고 쓸 수 있는 결과일 때만 말을 채택한다 (아키텍처 08. e900 은 2자, 재희 10자)
            text_usable = bool(r.usable and text_label and len(text) >= min_text_chars())
            if text_usable and all(c in r.scores for c in MOOD_CODES):
                text_probs = {c: r.scores[c] for c in MOOD_CODES}
    if text_out is not None:
        text_out.update({
            "label": r.label if r is not None else None,
            "turnLabel": text_label,
            "confidence": round(float(r.confidence), 4) if r is not None else None,
            "usable": text_usable,
            "reason": text_reason,
        })
    face_label = HSEMOTION_TO_TURN.get(face["label"]) if face.get("label") else None
    face_usable = bool(face.get("usable") and face_label)
    if s.companion_fusion_rule == "prob_add" and (text_probs is not None or not text_usable):
        fused = fuse_prob_add(
            no_response=not text,
            text_probs=text_probs,
            face_code=face_mood_code(face["label"]) if face_usable else None,
            weight=s.companion_face_weight,
        )
        return {"label": fused.label, "source": fused.source}
    fused = fuse_turn(
        no_response=not text,
        text_label=text_label,
        text_usable=text_usable,
        face_label=face_label,
        face_usable=face_usable,
    )
    return {"label": fused.label, "source": fused.source}


def run_turn(
    transcript: str,
    face: dict,
    classify_text: Callable[[str], TextEmotionResult],
    llm: LlmClient,
    question: str,
    prev_signals: dict | None,
    is_last_turn: bool,
    turn_no: int,
    stt_logprob: float | None = None,
    past_replies=(),
    input_kind: InputKind = "voice",
) -> dict:
    """한 턴을 처리한다 (S6-4, S7-7). 신호 추출과 응답 생성은 같은 시점에 함께 돌린다.

    응답은 직전 턴 신호만 쓰므로 이번 턴 신호를 기다리지 않는다 (아키텍처 6번 단계).
    stt_logprob: turbo 받아쓰기의 토큰 평균 로그확률 (글 대답·확률을 내지 않는 엔진은 None).
    question: 어르신이 지금 대답하는 말벗 문장 (직전 대답, 첫 턴은 여는 질문 - 11/13).
    past_replies: 이 대화에서 말벗이 이미 한 문장. 비슷한 대답을 다시 만들 때 쓴다 (11/13).
    input_kind: "text" 는 어르신이 직접 쓴 글 대답이다. 받아쓰기가 없으므로 되묻기(확신도·낱말 규칙)를 하지 않는다 (11/13).
        "voice" 는 음성 대답이다. logprob 를 내는 엔진(turbo)은 확신도로, 내지 않는 엔진(sensevoice·whisper)은 낱말 규칙으로 되묻는다.
    """
    s = get_settings()
    if input_kind == "text":
        # 11/13: 글 대답이 낱말 규칙에 걸려 "잘 알아듣지 못했어요"·"혹시 편안 이야기이신가요?"로 되물었다.
        #   위험 판정·피하는 말·기분은 일반 턴에서 그대로 돈다
        return _analyzed_turn(transcript, face, classify_text, llm, question, prev_signals, is_last_turn, turn_no,
                              past_replies)
    if s.companion_confirm_mode == "logprob" and stt_logprob is not None:
        # 11/09: 받아쓰기 확신도가 낮으면 되묻는다. 후보 낱말은 내지 않는다 (틀린 후보를 말하면 더 헷갈린다)
        # 11/13: 짧은 부정 대답("아니 없어")은 확신도가 낮아도 되묻지 않는다 (위험 판정은 일반 턴에서 그대로 돈다)
        short_ok = s.companion_short_answer_pass and is_short_plain_answer(transcript)
        if stt_logprob < s.companion_confirm_logprob and not short_ok:
            return _low_confidence_turn(transcript, llm)
    else:
        return _lexicon_checked_turn(transcript, face, classify_text, llm, question, prev_signals, is_last_turn, turn_no,
                                     past_replies)
    return _analyzed_turn(transcript, face, classify_text, llm, question, prev_signals, is_last_turn, turn_no,
                          past_replies)


def _risk_prob_capped(llm: LlmClient, transcript: str) -> float | None:
    try:
        return call_with_timeout(risk_probability, get_settings().companion_turn_cap_sec, llm, transcript)
    except TimeoutError:
        return None


def _low_confidence_turn(transcript: str, llm: LlmClient) -> dict:
    """확신도가 낮아 되묻는 턴. 감정·신호·대답은 만들지 않지만 위험 판정(낱말 규칙 + 모델 확률)은 돌린다.

    위험이면 되묻는 문장 대신 안전 안내 문장으로 답한다 (일반 턴과 같은 규칙).
    """
    try:
        risk_prob = call_with_timeout(risk_probability, get_settings().companion_turn_cap_sec, llm, transcript)
    except TimeoutError:
        risk_prob = None
    risk_flag = keyword_risk(transcript) or is_risk(risk_prob)
    return {
        "turnResult": {"label": UNDECIDED, "source": "TEXT"},
        "reply": {"text": safety_reply(transcript) if risk_flag else LOW_CONFIDENCE_REPLY, "isFollowUp": True},
        "signals": None,
        "replySource": "SAFETY" if risk_flag else "CONFIRM",
        "summary": None,
        "needsCheck": True,
        "riskFlag": risk_flag,
        "riskProbability": risk_prob,  # 모델 위험 확률 (계산 못 했으면 None) - 턴 응답용, 저장·로그 없음
        # 확신도가 낮아 되물은 턴 (연속 종료 횟수용, 응답에는 싣지 않는다). 위험 턴은 안전 안내가 우선이라 세지 않는다
        "lowConfidence": not risk_flag,
    }


def _lexicon_checked_turn(transcript, face, classify_text, llm, question, prev_signals, is_last_turn, turn_no,
                          past_replies=()) -> dict:
    """재희 버전 되묻기: 형태소·어휘 목록 규칙으로 실제 단어가 아닐 수 있는 말을 찾는다."""
    check = check_transcript(transcript)
    if check.suspicious:
        # 실제 단어가 아닐 수 있는 말이 있으면 뜻을 추측하지 않는다: 감정·신호·대답을 만들지 않고 다시 확인한다
        if get_settings().companion_confirm_risk_model:
            # 11/10 점검 H-3: 확인 턴에서도 위험 모델을 돌리고, 위험이면 되묻는 대신 안전 안내로 답한다 (확신도 낮은 턴과 같은 규칙)
            risk_prob = _risk_prob_capped(llm, transcript)
            risk_flag = keyword_risk(transcript) or is_risk(risk_prob)
            reply_text, source = (safety_reply(transcript), "SAFETY") if risk_flag else (confirm_reply(check), "CONFIRM")
        else:
            risk_flag, reply_text, source = keyword_risk(transcript), confirm_reply(check), "CONFIRM"  # 낱말 규칙만 (#206)
            risk_prob = None  # 모델을 돌리지 않았다
        return {
            "turnResult": {"label": UNDECIDED, "source": "TEXT"},
            "reply": {"text": reply_text, "isFollowUp": True},
            "signals": None,
            "replySource": source,
            "summary": None,
            "needsCheck": True,
            "riskFlag": risk_flag,
            "riskProbability": risk_prob,
        }
    return _analyzed_turn(transcript, face, classify_text, llm, question, prev_signals, is_last_turn, turn_no,
                          past_replies)


def _analyzed_turn(transcript, face, classify_text, llm, question, prev_signals, is_last_turn, turn_no,
                   past_replies=()) -> dict:
    """감정·신호·대답·요약·위험을 함께 만드는 일반 턴."""
    # 한 턴 전체 상한 (백엔드 요청 60초). 넘게 걸리는 작업은 기다리지 않고, 그 결과는 쓰지 않는다
    cap = get_settings().companion_turn_cap_sec
    started = time.monotonic()

    def left() -> float:
        return max(0.05, cap - (time.monotonic() - started))

    risk_first = get_settings().companion_risk_first
    turn_result = None
    text_mood: dict = {}  # 글 기분 분류 결과 - 턴 응답의 textMood (저장·로그 없음, 2026-11-10)
    if risk_first:
        # 11/10 점검 H-1: 위험(0.1초)·기분(0.07초) 확률을 생성보다 먼저 끝낸다.
        # 함께 보내면 어댑터 문에서 대답 생성 뒤에 줄을 서 시간 제한에 걸리고, 위험이 낱말 규칙으로만 판정됐다.
        risk_prob = _risk_prob_capped(llm, transcript)
        turn_result = analyze_turn_result(transcript, face, classify_text, text_out=text_mood)
    # 11/12: 말을 피하는 대답이면 모델 대답을 만들지 않고 물러나는 문장으로 답한다 (캐묻지 않는다).
    #   위험 턴이면 아래에서 안전 안내(safety_reply)로 덮어쓴다 - 위험 판정이 항상 우선이다
    avoided = get_settings().companion_avoid_reply and is_avoidance(transcript)
    timed_out = False
    pool = ThreadPoolExecutor(max_workers=4)
    try:
        signal_future = pool.submit(extract_signals, transcript, llm, get_settings().companion_llm_retry)
        # 글 기분은 risk_first 일 때만 생성 전에 있다. 꺼져 있으면 None 으로 넘겨 전 규칙으로 검사한다 (11/12)
        reply_future = None if avoided else pool.submit(
            generate_reply, llm, question, transcript, prev_signals, is_last_turn, turn_no,
            text_mood if risk_first else None, tuple(past_replies))
        summary_future = pool.submit(summarize_answer, llm, transcript)
        risk_future = None if risk_first else pool.submit(risk_probability, llm, transcript)
        try:
            signals = signal_future.result(timeout=left())
        except FutureTimeout:
            signals = None
        if reply_future is None:
            reply = avoid_reply(turn_no, is_last_turn)
        else:
            try:
                reply = reply_future.result(timeout=left())
            except FutureTimeout:
                reply = timeout_reply(turn_no, is_last_turn)  # 시간 상한: 미리 정한 문장으로 답한다 (말은 있으므로 받아 주는 문장)
                timed_out = True
        try:
            summary = summary_future.result(timeout=left())
        except FutureTimeout:
            summary = None
        if risk_future is not None:
            try:
                risk_prob = risk_future.result(timeout=left())
            except FutureTimeout:
                risk_prob = None
    finally:
        pool.shutdown(wait=False)  # 늦게 끝나는 작업은 백그라운드에서 끝나고, 결과는 버린다
    # 위험 판정 = 낱말 규칙 또는 모델 확률 (모델이 꺼져 있어도 낱말 규칙으로 알린다)
    model_flag = is_risk(risk_prob) if risk_prob is not None else bool((signals or {}).get("riskFlag"))
    risk_flag = keyword_risk(transcript) or model_flag
    if turn_result is None:  # risk_first 가 꺼져 있으면 여기서 글 기분을 만든다 (대체 문장 고르기에 쓴다)
        turn_result = analyze_turn_result(transcript, face, classify_text, text_out=text_mood)
    if reply.source == "FALLBACK" and not is_last_turn and is_negative_mood(text_mood):
        # 11/11: 힘든 이야기에 대체 문장이 나가면 공감 문장으로 받는다 (화제를 바꾸지 않는다)
        reply = ReplyResult(text=fallback_text(turn_no, is_last_turn, negative=True), isFollowUp=reply.isFollowUp,
                            source=reply.source, failures=reply.failures)
    if risk_flag:
        # 위험 신호가 켜지면 평소 대답 대신 안내 문장으로 답한다 (식사 질문 등으로 넘기지 않는다)
        reply = ReplyResult(text=safety_reply(transcript), isFollowUp=reply.isFollowUp, source="SAFETY",
                            failures=reply.failures)
    if signals is not None:
        signals = {**signals, "riskFlag": risk_flag}
    _log_reply_source(reply, timed_out)
    return {
        "turnResult": turn_result,
        "textMood": text_mood or None,
        "reply": {"text": reply.text, "isFollowUp": reply.isFollowUp},
        "signals": signals,
        "replySource": reply.source,
        "summary": summary,  # 대답 원문 대신 저장하는 한 줄 요약 (보호자 화면용)
        "riskFlag": risk_flag,  # 대화 한 번의 위험 여부 (낱말 규칙 또는 모델. 신호 추출이 실패해도 판정한다)
        # 모델 위험 확률만 (낱말 규칙·signals 대체 판정은 섞지 않는다). 계산 못 했거나 시간 초과면 None. 저장·로그 없음
        "riskProbability": risk_prob,
        # 말을 피한 턴인가 (11/12). 위험 턴이면 안전 안내가 나가도 감지 여부는 그대로 남긴다. 원문은 싣지 않는다
        "avoided": avoided,
    }


def reply_reason(reply: ReplyResult, timed_out: bool) -> str:
    """대답이 왜 그 출처가 됐는지 코드 하나로 (11/13). 위험 턴은 SAFETY 가 우선한다."""
    if reply.source == "SAFETY":
        return "SAFETY"
    if timed_out:
        return "TIMEOUT"
    if not reply.failures:
        return "NONE"
    return "+".join(dict.fromkeys(REASON_CODES.get(f, "OTHER") for f in reply.failures))


def _log_reply_source(reply: ReplyResult, timed_out: bool) -> None:
    # 코드만 남긴다 (어르신 말·대답 원문 없음). 시간 초과로 대체됐는데 위험 턴이면 source=SAFETY, timeout=1 로 함께 보인다
    logger.info("[COMPANION-REPLY-SOURCE] source=%s reason=%s timeout=%d",
                reply.source, reply_reason(reply, timed_out), int(timed_out))


def run_no_response_turn(turn_no: int, is_last_turn: bool) -> dict:
    """어르신이 답하지 않은 턴. 분석 없이 턴 결과를 NO_RESPONSE 로 남기고 미리 정한 문장으로 답한다 (S7-8)."""
    reply = no_response_reply(turn_no, is_last_turn)
    return {
        "turnResult": {"label": None, "source": "NO_RESPONSE"},
        "reply": {"text": reply.text, "isFollowUp": reply.isFollowUp},
        "signals": None,
        "replySource": reply.source,
        "riskProbability": None,  # 무응답 턴은 위험 모델을 돌리지 않는다
    }


FAKE_LABEL = UNDECIDED  # 대화 결과 집계가 비었을 때 쓰는 값 (판단 보류)
