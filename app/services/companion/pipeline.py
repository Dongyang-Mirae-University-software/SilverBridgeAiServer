from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Callable

from app.core.config import get_settings
from app.services.companion.answer_summary import summarize_answer
from app.services.companion.fusion import GGARA_TO_TURN, HSEMOTION_TO_TURN, UNDECIDED, fuse_turn
from app.services.companion.llm_client import LlmClient
from app.services.companion.reply_service import generate_reply, no_response_reply
from app.services.companion.risk import is_risk, keyword_risk, risk_probability
from app.services.companion.signals import extract_signals
from app.services.companion.transcript_check import check_transcript, confirm_reply
from app.services.companion.text_emotion_service import TextEmotionResult, TextEmotionUnavailable
from app.services.companion.timeouts import call_with_timeout


def analyze_turn_result(
    transcript: str,
    face: dict,
    classify_text: Callable[[str], TextEmotionResult],
) -> dict:
    """말(GGARA02)과 표정(HSEmotion)을 아키텍처 08 결합 표로 합쳐 턴 결과를 만든다.

    텍스트 모델을 못 쓰면 텍스트는 빠지고 표정만으로 판단한다.
    """
    s = get_settings()
    text = transcript.strip()
    text_label = None
    text_usable = False
    if text:
        try:
            r = call_with_timeout(classify_text, s.companion_text_timeout_sec, text)
        except (TextEmotionUnavailable, TimeoutError):  # 시간 초과면 말은 빼고 표정만으로 판단 (S9-1)
            r = None
        if r is not None:
            text_label = GGARA_TO_TURN.get(r.label)
            # 발화 10자 이상이고 확신도가 기준 이상일 때만 말을 채택한다 (아키텍처 08)
            text_usable = bool(r.usable and text_label and len(text) >= s.companion_min_text_chars)
    face_label = HSEMOTION_TO_TURN.get(face["label"]) if face.get("label") else None
    face_usable = bool(face.get("usable") and face_label)
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
) -> dict:
    """한 턴을 처리한다 (S6-4, S7-7). 신호 추출과 응답 생성은 같은 시점에 함께 돌린다.

    응답은 직전 턴 신호만 쓰므로 이번 턴 신호를 기다리지 않는다 (아키텍처 6번 단계).
    """
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
