from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Callable

from app.core.config import get_settings
from app.services.companion.answer_summary import summarize_answer
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
from app.services.companion.reply_service import SAFETY_REPLY, ReplyResult, generate_reply, no_response_reply
from app.services.companion.risk import is_risk, keyword_risk, risk_probability
from app.services.companion.signals import extract_signals
from app.services.companion.transcript_check import check_transcript, confirm_reply
from app.services.companion.text_emotion_service import TextEmotionResult, TextEmotionUnavailable
from app.services.companion.timeouts import call_with_timeout

# 확신도가 낮을 때 되묻는 문장 (11/09). 후보 낱말을 넣지 않는다
LOW_CONFIDENCE_REPLY = "잘 알아듣지 못했어요. 한 번 더 말씀해 주시겠어요?"


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
) -> dict:
    """말과 표정을 합쳐 턴 결과를 만든다.

    prob_add(11/10 기본): 글 기분 4코드 확률 + 표정 코드 쪽으로 가중치 (fuse_prob_add).
    plan41(재희): 아키텍처 08 결합 표 (fuse_turn). 글 결과에 4코드 확률이 없으면(GGARA02 등) prob_add 설정이어도 이 표를 쓴다.
    텍스트 모델을 못 쓰면 텍스트는 빠지고 표정만으로 판단한다.
    """
    s = get_settings()
    text = transcript.strip()
    text_label = None
    text_usable = False
    text_probs = None
    if text:
        try:
            r = call_with_timeout(classify_text, s.companion_text_timeout_sec, text)
        except (TextEmotionUnavailable, TimeoutError):  # 시간 초과면 말은 빼고 표정만으로 판단 (S9-1)
            r = None
        if r is not None:
            text_label = TEXT_TO_TURN.get(r.label)
            # 발화가 최소 글자 수 이상이고 쓸 수 있는 결과일 때만 말을 채택한다 (아키텍처 08. e900 은 2자, 재희 10자)
            text_usable = bool(r.usable and text_label and len(text) >= min_text_chars())
            if text_usable and all(c in r.scores for c in MOOD_CODES):
                text_probs = {c: r.scores[c] for c in MOOD_CODES}
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
) -> dict:
    """한 턴을 처리한다 (S6-4, S7-7). 신호 추출과 응답 생성은 같은 시점에 함께 돌린다.

    응답은 직전 턴 신호만 쓰므로 이번 턴 신호를 기다리지 않는다 (아키텍처 6번 단계).
    stt_logprob: turbo 받아쓰기의 토큰 평균 로그확률 (글 대답·확률을 내지 않는 엔진은 None).
    """
    s = get_settings()
    if s.companion_confirm_mode == "logprob" and stt_logprob is not None:
        # 11/09: 받아쓰기 확신도가 낮으면 되묻는다. 후보 낱말은 내지 않는다 (틀린 후보를 말하면 더 헷갈린다)
        if stt_logprob < s.companion_confirm_logprob:
            return _low_confidence_turn(transcript, llm)
    else:
        return _lexicon_checked_turn(transcript, face, classify_text, llm, question, prev_signals, is_last_turn, turn_no)
    return _analyzed_turn(transcript, face, classify_text, llm, question, prev_signals, is_last_turn, turn_no)


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
        "reply": {"text": SAFETY_REPLY if risk_flag else LOW_CONFIDENCE_REPLY, "isFollowUp": True},
        "signals": None,
        "replySource": "SAFETY" if risk_flag else "CONFIRM",
        "summary": None,
        "needsCheck": True,
        "riskFlag": risk_flag,
        # 확신도가 낮아 되물은 턴 (연속 종료 횟수용, 응답에는 싣지 않는다). 위험 턴은 안전 안내가 우선이라 세지 않는다
        "lowConfidence": not risk_flag,
    }


def _lexicon_checked_turn(transcript, face, classify_text, llm, question, prev_signals, is_last_turn, turn_no) -> dict:
    """재희 버전 되묻기: 형태소·어휘 목록 규칙으로 실제 단어가 아닐 수 있는 말을 찾는다."""
    check = check_transcript(transcript)
    if check.suspicious:
        # 실제 단어가 아닐 수 있는 말이 있으면 뜻을 추측하지 않는다: 감정·신호·대답을 만들지 않고 다시 확인한다
        return {
            "turnResult": {"label": UNDECIDED, "source": "TEXT"},
            "reply": {"text": confirm_reply(check), "isFollowUp": True},
            "signals": None,
            "replySource": "CONFIRM",
            "summary": None,
            "needsCheck": True,
            "riskFlag": keyword_risk(transcript),  # 낱말 규칙은 확인 턴에도 적용한다
        }
    return _analyzed_turn(transcript, face, classify_text, llm, question, prev_signals, is_last_turn, turn_no)


def _analyzed_turn(transcript, face, classify_text, llm, question, prev_signals, is_last_turn, turn_no) -> dict:
    """감정·신호·대답·요약·위험을 함께 만드는 일반 턴."""
    # 한 턴 전체 상한 (백엔드 요청 60초). 넘게 걸리는 작업은 기다리지 않고, 그 결과는 쓰지 않는다
    cap = get_settings().companion_turn_cap_sec
    started = time.monotonic()

    def left() -> float:
        return max(0.05, cap - (time.monotonic() - started))

    pool = ThreadPoolExecutor(max_workers=4)
    try:
        signal_future = pool.submit(extract_signals, transcript, llm, get_settings().companion_llm_retry)
        reply_future = pool.submit(generate_reply, llm, question, transcript, prev_signals, is_last_turn, turn_no)
        summary_future = pool.submit(summarize_answer, llm, transcript)
        risk_future = pool.submit(risk_probability, llm, transcript)
        try:
            signals = signal_future.result(timeout=left())
        except FutureTimeout:
            signals = None
        try:
            reply = reply_future.result(timeout=left())
        except FutureTimeout:
            reply = no_response_reply(turn_no, is_last_turn)  # 시간 상한: 미리 정한 문장으로 답한다
        try:
            summary = summary_future.result(timeout=left())
        except FutureTimeout:
            summary = None
        try:
            risk_prob = risk_future.result(timeout=left())
        except FutureTimeout:
            risk_prob = None
    finally:
        pool.shutdown(wait=False)  # 늦게 끝나는 작업은 백그라운드에서 끝나고, 결과는 버린다
    # 위험 판정 = 낱말 규칙 또는 모델 확률 (모델이 꺼져 있어도 낱말 규칙으로 알린다)
    model_flag = is_risk(risk_prob) if risk_prob is not None else bool((signals or {}).get("riskFlag"))
    risk_flag = keyword_risk(transcript) or model_flag
    if risk_flag:
        # 위험 신호가 켜지면 평소 대답 대신 안내 문장으로 답한다 (식사 질문 등으로 넘기지 않는다)
        reply = ReplyResult(text=SAFETY_REPLY, isFollowUp=reply.isFollowUp, source="SAFETY", failures=reply.failures)
    if signals is not None:
        signals = {**signals, "riskFlag": risk_flag}
    return {
        "turnResult": analyze_turn_result(transcript, face, classify_text),
        "reply": {"text": reply.text, "isFollowUp": reply.isFollowUp},
        "signals": signals,
        "replySource": reply.source,
        "summary": summary,  # 대답 원문 대신 저장하는 한 줄 요약 (보호자 화면용)
        "riskFlag": risk_flag,  # 대화 한 번의 위험 여부 (낱말 규칙 또는 모델. 신호 추출이 실패해도 판정한다)
    }


def run_no_response_turn(turn_no: int, is_last_turn: bool) -> dict:
    """어르신이 답하지 않은 턴. 분석 없이 턴 결과를 NO_RESPONSE 로 남기고 미리 정한 문장으로 답한다 (S7-8)."""
    reply = no_response_reply(turn_no, is_last_turn)
    return {
        "turnResult": {"label": None, "source": "NO_RESPONSE"},
        "reply": {"text": reply.text, "isFollowUp": reply.isFollowUp},
        "signals": None,
        "replySource": reply.source,
    }


FAKE_LABEL = UNDECIDED  # 대화 결과 집계가 비었을 때 쓰는 값 (판단 보류)
