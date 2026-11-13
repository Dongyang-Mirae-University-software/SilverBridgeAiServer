from __future__ import annotations

import logging
import threading
import time

from typing import Literal

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import Response
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.core.companion_errors import CompanionError
from app.core.companion_security import require_companion_api_key
from app.core.config import Settings, get_settings
from app.core.response import success_response
from app.schemas.companion_schema import TIME_SLOTS, CompanionStatus
from app.services.companion import pipeline
from app.services.companion.image_format import detect_image_format
from app.services.companion.audio_format import ALLOWED_FORMATS, detect_audio_format
from app.services.companion.dialogue_store import Dialogue, store
from app.services.companion.face_service import FaceResult, FaceUnavailable, face_label_ko, face_service
from app.services.companion.llm_client import llm_client
from app.services.companion.multipart_memory import read_file_part, read_file_parts
from app.services.companion import reply_service
from app.services.companion.reply_service import REPLY_PROMPT_VERSION
from app.services.companion.review_store import STATUSES, review_store
from app.services.companion.stt_engine import get_stt_api_transcriber, transcribe_turn_audio
from app.services.companion.audio_decode import TARGET_RATE, decode_to_pcm16k
from app.services.companion.stt_service import SttUnavailable
from app.services.companion.mood_service import mood_service
from app.services.companion.text_emotion_service import text_emotion_service
from app.services.companion.timeouts import call_with_timeout
from app.services.companion.turn_record import build_dialogue_result
from app.services.companion.tts_service import TtsNotFound, TtsNotReady, tts_service

logger = logging.getLogger(__name__)

# 말벗 라우터. 모든 경로는 말벗 전용 키로 보호한다 (D4).
# 경로는 명세 9-1절 기준 (D3 결정): /companion/dialogues 계열.
router = APIRouter(
    prefix="/companion",
    tags=["Companion"],
    dependencies=[Depends(require_companion_api_key)],
)

FRAME_MAX_BYTES = 1 * 1024 * 1024  # 프레임 한 장 상한 (JPEG/PNG). BE 기준 1MB (2026-11-09 맞춤)
ANSWER_MAX_CHARS = 500  # 글 대답 상한 (BE 기준 1~500자)


def get_transcriber():
    """음성 인식 함수 (테스트에서 가짜로 바꿔 끼운다). 실제는 말소리 감지 + 설정된 엔진 (11/09 기본 Silero + turbo)."""
    return transcribe_turn_audio


def get_face_analyzer():
    """표정 분석 함수 (테스트에서 바꿔 끼운다). 실제는 YuNet + HSEmotion 래퍼 (S4)."""
    return face_service.analyze


def get_text_classifier():
    """글자 감정 함수 (테스트에서 바꿔 끼운다). 실제는 e900 기분(11/10 기본) 또는 GGARA02 래퍼(재희, S5-1)."""
    if get_settings().companion_text_mood_engine == "e900":
        return mood_service.classify
    return text_emotion_service.classify


def get_pcm_transcriber():
    """글자 변환 함수 (음성 인식 API용, 테스트에서 바꿔 끼운다). 실제는 COMPANION_STT_API_ENGINE (기본 SenseVoice 그대로)."""
    return get_stt_api_transcriber()


def get_llm():
    """LLM 클라이언트 (테스트에서 바꿔 끼운다). 실제는 EXAONE 래퍼 (S5-4)."""
    return llm_client


def _no_face() -> dict:
    """프레임이 없거나 표정을 쓸 수 없을 때의 표정 값 (대화는 계속된다)."""
    return {"label": None, "labelKo": None, "confidence": 0.0, "frameCount": 0, "usable": False, "totalFrames": 0,
            "facesFound": 0}


def _face_dict(f: FaceResult) -> dict:
    # totalFrames·facesFound: 보낸 프레임 수와 얼굴이 잡힌 프레임 수 (표정이 안 나올 때 이유를 보이려고 넣었다)
    # labelKo: 한국어 이름 (11/13). 표정을 쓸 수 없으면(얼굴 없음·프레임 부족) None - label 은 영어 그대로
    return {"label": f.label, "labelKo": face_label_ko(f.label) if f.usable else None, "confidence": f.confidence,
            "frameCount": f.frameCount, "usable": f.usable, "totalFrames": f.totalFrames, "facesFound": f.facesFound}


class DialogueOpen(BaseModel):
    wardId: str = Field(min_length=1, max_length=64)
    sessionId: str = Field(min_length=1, max_length=64)
    slot: str
    question: str = Field(min_length=1, max_length=200)  # 백엔드가 시각에 맞게 고른 질문 문장 (AI는 조회하지 않고 받는다)
    questionId: str | None = Field(default=None, max_length=64)  # 백엔드가 질문마다 준 ID (결과에 그대로 되돌려 준다)


class ReplyText(BaseModel):
    text: str = Field(default="")  # 길이(500자)는 핸들러에서 ANSWER_TOO_LONG 으로 거절한다


class ReviewDecision(BaseModel):
    decision: Literal["approved", "rejected"]


def _run_turn(d: Dialogue, transcript: str, face: dict, classify_text, llm, stt_logprob: float | None = None,
              input_kind: str = "voice") -> dict:
    """턴 하나를 처리하고, 직전 신호와 검수 목록을 갱신한다. 답이 비었으면 무응답 턴이다 (S7-8)."""
    turn_no = d.turnNo + 1
    is_last_turn = turn_no >= d.maxTurns
    if not transcript.strip():
        result = pipeline.run_no_response_turn(turn_no, is_last_turn)
    else:
        result = pipeline.run_turn(
            transcript=transcript,
            face=face,
            classify_text=classify_text,
            llm=llm,
            # 11/13: 여는 질문이 아니라 어르신이 지금 대답하는 직전 말벗 문장을 넣는다 (세션 203 5번이 2번 질문을 되풀이했다)
            question=d.lastQuestion or d.question,
            prev_signals=d.lastSignals,
            is_last_turn=is_last_turn,
            turn_no=turn_no,
            stt_logprob=stt_logprob,
            past_replies=tuple(d.replies),
            input_kind=input_kind,  # 11/13: 글 대답은 되묻지 않는다 (받아쓰기가 없다)
        )
        if result["signals"] is not None:  # 신호가 없는 턴(무응답·다시 확인)은 직전 신호를 그대로 둔다
            store.set_last_signals(d.dialogueId, result["signals"])
        if result.get("riskFlag"):
            store.mark_risk(d.dialogueId)  # 대화 결과의 alertLevel(URGENT)에 쓴다
    limit = get_settings().companion_confirm_streak_limit
    if result.get("lowConfidence") and limit and d.lowConfidenceStreak + 1 >= limit and not result.get("riskFlag"):
        # 이번 턴으로 대화가 끝나므로 다시 말해 달라고 하지 않고 마무리 문장으로 답한다
        result["reply"] = {"text": reply_service.FALLBACK_LAST[0], "isFollowUp": False}
        result["replySource"] = "FALLBACK"
    _apply_avoid_end(d, result, is_last_turn)
    result = _avoid_repeat(d, result, is_last_turn)
    _queue_for_review(d, result)
    if result.get("replySource") != "CONFIRM":
        # 되묻기 턴은 바꾸지 않는다: 다시 말해 주신 대답은 그 전 질문에 대한 것이다 (세션 203 4번 "없다고"는 2번 질문의 대답)
        d.lastQuestion = result["reply"]["text"]
    return result


def _apply_avoid_end(d: Dialogue, result: dict, is_last_turn: bool) -> None:
    """말을 피한 턴을 세고, 한 대화에서 COMPANION_AVOID_END_COUNT 번째면 마무리 문장으로 답하고 끝낸다 (11/12).

    위험 턴은 세지 않는다 (안전 안내가 우선이고, 대화를 일찍 끝내지 않는다).
    종료 이유는 AVOIDED (11/12 사용자 결정): 끝내기를 누른 USER_STOP 과 나눠 기록한다. 백엔드 V72 가 이 값을 받는다
    (백엔드를 먼저 배포한다 - 모르는 값은 백엔드가 USER_STOP 으로 바꿔 저장한다).
    """
    if not result.get("avoided") or result.get("riskFlag"):
        return
    count = store.mark_avoid(d.dialogueId)
    limit = get_settings().companion_avoid_end_count
    end = bool(limit) and count >= limit and not is_last_turn  # 마지막 턴은 원래대로 MAX_TURNS 로 끝난다
    if end:
        result["reply"] = {"text": reply_service.FALLBACK_LAST[0], "isFollowUp": False}
        result["endReason"] = "AVOIDED"
    logger.info("[COMPANION-AVOID] count=%d end=%s last=%s", count, end, is_last_turn)  # 어르신 말 원문은 남기지 않는다


def _avoid_repeat(d: Dialogue, result: dict, is_last_turn: bool) -> dict:
    """같은 대화에서 이미 한 말(질문 포함)이 다시 나오면, 아직 안 쓴 미리 정한 문장으로 바꾼다.

    미리 정한 문장을 모두 써 버렸으면 그대로 둔다. 바꾼 문장은 대화 기록에 남긴다.
    """
    text = result["reply"]["text"]
    # 되묻는 문장·안전 안내 문장은 반복돼도 그대로 둔다. 바꾸면 못 알아들은 말·위험한 말에 평범한 대답을 하게 된다 (11/09 점검 H-1·H-2)
    keep = get_settings().companion_keep_fixed_replies and result.get("replySource") in ("CONFIRM", "SAFETY")
    if text in d.replies and not keep:
        # 부정 기분 턴이면 공감 문장 가운데서 바꾼다 (11/11, 화제를 바꾸는 문장으로 되돌아가지 않게)
        # 무응답 턴은 무응답 문장 가운데서 바꾼다 (11/12 H-1, 침묵에 "말씀해 주셔서…"로 답하지 않게)
        no_response = (result.get("turnResult") or {}).get("source") == "NO_RESPONSE"
        # 말을 피한 턴은 물러나는 문장 가운데서 바꾼다 (11/12, 받아 주는 문장으로 캐묻지 않게). 마무리된 턴은 마무리 문장
        avoid = result.get("replySource") == "AVOID" and not result.get("endReason")
        candidates = reply_service.fallback_options(
            is_last_turn or bool(result.get("endReason")), reply_service.is_negative_mood(result.get("textMood")),
            no_response=no_response, avoid=avoid)
        unused = [c for c in candidates if c not in d.replies]
        if unused:
            result["reply"] = {"text": unused[0], "isFollowUp": not is_last_turn}
            text = unused[0]
    d.replies.append(text)
    return result


def _queue_for_review(d: Dialogue, result: dict) -> None:
    """AI 가 만든 응답 문장을 검수 목록에 넣는다. 대답 원문은 넣지 않는다 (명세 9-4).

    위험 플래그(riskFlag)가 켜진 턴은 위험 표시를 붙인다 (S7-5).
    위험 = 최상위 riskFlag 또는 signals.riskFlag. 다시 확인 턴과 신호 추출이 시간을 넘긴 턴은 signals 가 null 이고
    위험을 최상위에만 싣는다 - signals 만 보면 그 턴의 위험 표시가 빠진다 (백엔드 턴 처리와 같은 규칙).
    """
    version = REPLY_PROMPT_VERSION if result["replySource"] == "LLM" else reply_service.FALLBACK_VERSION
    signals = result.get("signals") or {}
    risk = bool(result.get("riskFlag")) or bool(signals.get("riskFlag"))
    review_store.add(d.dialogueId, d.turnNo + 1, result["reply"]["text"], version, risk=risk, ward_id=d.wardId)


@router.get(
    "/review/replies",
    summary="검수 목록 (AI 문장만, 어르신 대답 원문 없음)",
    description="AI가 만든 응답 문장 목록을 상태별로 본다. 어르신 대답 원문은 없다. "
                "`status`는 pending·approved·rejected 중 하나 (아니면 422 STATUS_INVALID). 보관 30일 (COMPANION_LOG_RETENTION_DAYS).",
)
def review_list(status: str | None = None) -> dict:
    if status is not None and status not in STATUSES:
        raise CompanionError(422, "STATUS_INVALID", "status 는 pending·approved·rejected 중 하나입니다.")
    return success_response("검수 목록", {"items": review_store.list_items(status)})


@router.post(
    "/review/replies/{item_id}",
    summary="검수 항목을 승인하거나 반려한다",
    description="운영자가 응답 문장을 `approved` 또는 `rejected`로 정한다. 없는 항목은 404 REVIEW_ITEM_NOT_FOUND.",
)
def review_decide(item_id: str, body: ReviewDecision) -> dict:
    row = review_store.decide(item_id, body.decision)
    if row is None:
        raise CompanionError(404, "REVIEW_ITEM_NOT_FOUND", "검수 항목을 찾을 수 없습니다.")
    return success_response("검수 결정을 저장했습니다.", row)


@router.delete(
    "/wards/{ward_id}/data",
    summary="피보호자 탈퇴 시 말벗 데이터 삭제 (AI 쪽 데이터)",
    description="이 피보호자의 메모리 대화와 검수 항목을 지운다. 백엔드의 대화 결과와 턴 기록은 백엔드가 지운다.",
)
def delete_ward_data(ward_id: str) -> dict:
    """AI 서버가 가진 말벗 데이터를 지운다: 메모리의 대화, 검수 항목. 백엔드의 대화 결과는 백엔드가 지운다 (노션 요구)."""
    dialogues = store.delete_ward(ward_id)
    reviews = review_store.delete_by_ward(ward_id)
    tts_service.delete_ward(ward_id)  # 그 피보호자 대화의 소리도 지운다 (응답 형식은 그대로 둔다 - 백엔드 계약)
    return success_response("말벗 데이터를 지웠습니다.", {"wardId": ward_id, "dialogues": dialogues, "reviewItems": reviews})


AUDIO_WAIT_MAX_SEC = 8.0
UTTERANCE_INDEX_MAX = 100


@router.get(
    "/sessions/{session_id}/utterances/{index}/audio",
    summary="말벗 문장의 소리(mp3)를 받는다",
    description="index 0 = 여는 질문, n = n번째 턴 뒤 말벗이 한 말. 합성 중이면 `wait`초(0~8, 기본 5)까지 기다린다. "
                "200 `audio/mpeg` (24kHz mono, 1MB 이하). 시간 안에 못 끝나면 503 TTS_NOT_READY, "
                "없음·만료(15분)·꺼짐·합성 실패면 404 TTS_NOT_FOUND. 소리는 메모리에만 잠시 두고 저장하지 않는다.",
    responses={200: {"content": {"audio/mpeg": {}}}},
)
def utterance_audio(
    session_id: str,
    index: int,
    wait: float = 5.0,
    # 대화 주인(BE #485가 싣는다). 있으면 소리를 만든 대화의 wardId와 대조하고 다르면 404 - 로그에 남기지 않는다
    ward_id: str | None = Header(default=None, alias="X-Companion-Ward-Id", include_in_schema=False),
) -> Response:
    if not (0.0 <= wait <= AUDIO_WAIT_MAX_SEC):
        raise CompanionError(422, "WAIT_INVALID", "wait 는 0~8초입니다.")
    if not (0 <= index <= UTTERANCE_INDEX_MAX) or not (1 <= len(session_id) <= 64):
        raise CompanionError(404, "TTS_NOT_FOUND", "소리가 없습니다.")
    try:
        data = tts_service.get(session_id, index, wait, ward_id=ward_id or None)
    except TtsNotReady as exc:
        raise CompanionError(503, "TTS_NOT_READY", "소리를 아직 만드는 중입니다.") from exc
    except TtsNotFound as exc:
        raise CompanionError(404, "TTS_NOT_FOUND", "소리가 없습니다.") from exc
    return Response(content=data, media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


@router.get(
    "/status",
    summary="말벗 설정 상태 (키 보호 확인용)",
    description="말벗이 켜졌는지, 설정이 비었는지만 돌려준다. 키 값은 응답에 넣지 않는다.",
)
def companion_status(settings: Settings = Depends(get_settings)) -> dict:
    status = CompanionStatus(
        enabled=bool(settings.companion_api_key),
        sessionLimitSec=settings.companion_session_limit_sec,
        maxTurns=settings.companion_max_turns,
        fusionRule=settings.companion_fusion_rule,
    )
    return success_response("말벗 상태", status.model_dump())


def _dialogue_or_404(dialogue_id: str) -> Dialogue:
    d = store.get(dialogue_id)
    if d is None:
        raise CompanionError(404, "DIALOGUE_NOT_FOUND", "대화를 찾을 수 없습니다.")
    if d.ended:
        raise CompanionError(409, "DIALOGUE_ENDED", "이미 끝난 대화입니다.")
    return d


def _risk_probability_out(p) -> float | None:
    """턴 응답의 riskProbability. 백엔드 누적 기준(0.97)이 v9 척도라 다른 프롬프트(v4)의 확률은 내보내지 않는다."""
    if p is None or get_settings().companion_risk_prompt != "v9":
        return None
    return round(float(p), 4)


def _start_audio(d: Dialogue, index: int, text: str, cache_text: bool = False) -> dict | None:
    """말벗 문장의 소리를 뒤에서 만들기 시작한다 (응답을 기다리게 하지 않는다). 시작했으면 {"index": index}."""
    return {"index": index} if tts_service.start(d.wardId, d.sessionId, index, text, cache_text=cache_text) else None


def _turn_response(d: Dialogue, result: dict, transcript: str | None = None, face: dict | None = None) -> dict:
    face_label = (face or {}).get("label") if (face or {}).get("usable") else None  # 표정을 못 읽었으면 기록하지 않는다
    updated = store.add_turn(d.dialogueId, result["turnResult"]["label"], result["turnResult"]["source"],
                             get_settings().companion_no_response_limit, face=face_label,
                             summary=result.get("summary"), needs_check=bool(result.get("needsCheck")),
                             low_confidence=bool(result.get("lowConfidence")),
                             low_confidence_limit=get_settings().companion_confirm_streak_limit,
                             end_reason=result.get("endReason"))
    body = {
        "dialogueId": d.dialogueId,
        "turnNo": updated.turnNo,
        "turnResult": result["turnResult"],
        "reply": result["reply"],
        "face": face if face is not None else _no_face(),
        "signals": result.get("signals"),  # 추출 실패면 null (명세: 대화 응답은 정상으로 보낸다)
        # 글 기분 분류 결과 (백엔드 테스트 계정 화면용, 2026-11-10). 무응답·다시 확인 턴은 null. 저장하지 않는다
        "textMood": result.get("textMood"),
        "riskFlag": bool(result.get("riskFlag")),  # 신호가 null 이어도 위험 여부는 항상 채운다 (백엔드 요청)
        # v9 위험 확률 (백엔드 여러 날 위험 누적용, 2026-11-11). 계산 안 했거나 시간 초과·v9 아님이면 null. 저장·로그 없음
        "riskProbability": _risk_probability_out(result.get("riskProbability")),
        "avoided": bool(result.get("avoided")),  # 말을 피한 턴인가 (11/12). 원문·이유는 싣지 않는다
        "ended": updated.ended,
        "endReason": updated.endReason,
        # 말벗 목소리 (11/13): 이 턴 응답 문장의 소리를 만들기 시작했으면 {"index": 턴 번호}, 꺼짐·포화면 null.
        # 소리는 GET /companion/sessions/{sessionId}/utterances/{index}/audio 로 받는다
        "audio": _start_audio(d, updated.turnNo, result["reply"]["text"]),
    }
    if transcript is not None:
        body["transcript"] = transcript  # 받아 적은 글은 응답으로만 나간다. 저장하지 않는다.
    return body


@router.post(
    "/dialogues",
    summary="시각의 질문으로 대화를 연다",
    description="시각(09:00·12:00·15:00·18:00·21:00)과 질문 문장(`question`)을 받아 대화를 연다. "
                "질문은 백엔드가 골라서 보낸다 (AI는 백엔드를 부르지 않는다). 질문 번호(`questionId`)는 결과에 그대로 돌려준다. "
                "응답의 `dialogueId`를 이후 호출에 쓴다. 오류: 422 SLOT_INVALID.",
)
def open_dialogue(body: DialogueOpen, settings: Settings = Depends(get_settings)) -> dict:
    if body.slot not in TIME_SLOTS:
        raise CompanionError(422, "SLOT_INVALID", "시각 값이 5개 중 하나가 아닙니다.")
    d = store.create(body.wardId, body.sessionId, body.slot, body.question, settings.companion_max_turns,
                     question_id=body.questionId)
    return success_response("대화를 열었습니다.", {
        "dialogueId": d.dialogueId,
        "question": d.question,
        "maxTurns": d.maxTurns,
        # 여는 질문은 index 0. 백엔드가 고른 질문 문장은 매번 같으므로 문장 기준으로도 보관한다 (11/13)
        "audio": _start_audio(d, 0, d.question, cache_text=True),
    })


@router.post(
    "/dialogues/{dialogue_id}/reply",
    summary="글로 된 대답을 보낸다",
    description="글 대답 1턴을 처리한다. 응답에 턴 판정(`turnResult`)과 응답 문장(`reply`), 종료 여부(`ended`)를 준다. "
                "대답 원문은 응답과 저장에 남기지 않는다. 오류: 422 ANSWER_EMPTY, 404 DIALOGUE_NOT_FOUND, 409 DIALOGUE_ENDED.",
)
def reply_text(
    dialogue_id: str,
    body: ReplyText,
    settings: Settings = Depends(get_settings),
    classify_text=Depends(get_text_classifier),
    llm=Depends(get_llm),
) -> dict:
    d = _dialogue_or_404(dialogue_id)
    if not body.text.strip():
        raise CompanionError(422, "ANSWER_EMPTY", "대답 글자가 비어 있습니다.")
    if len(body.text) > ANSWER_MAX_CHARS:
        raise CompanionError(422, "ANSWER_TOO_LONG", f"대답은 {ANSWER_MAX_CHARS}자 이하여야 합니다.")
    face = _no_face()
    result = _run_turn(d, body.text, face, classify_text, llm, input_kind="text")
    out = _turn_response(d, result)
    return success_response("턴 분석 완료", out)


@router.post(
    "/dialogues/{dialogue_id}/reply/audio",
    summary="음성 대답을 보낸다 (WAV·webm·mp3)",
    description="음성 1턴을 처리한다 (multipart: `audio` 1개 5MB 이하, `frame` 최대 5장 장당 1MB 이하, JPEG·PNG). "
                "응답에 받아 적은 글(`transcript`)을 넣는다. 글자가 없으면 무응답 턴. 얼굴이 없으면 `face.usable=false`. "
                "오류: 422 AUDIO_MISSING·AUDIO_FORMAT_UNSUPPORTED·FRAME_FORMAT_UNSUPPORTED, 413 AUDIO_TOO_LARGE·FRAMES_TOO_MANY·FRAME_TOO_LARGE, 503 STT_UNAVAILABLE.",
)
async def reply_audio(
    dialogue_id: str,
    request: Request,
    settings: Settings = Depends(get_settings),
    transcribe=Depends(get_transcriber),
    analyze_face=Depends(get_face_analyzer),
    classify_text=Depends(get_text_classifier),
    llm=Depends(get_llm),
) -> dict:
    d = _dialogue_or_404(dialogue_id)
    max_bytes = settings.companion_upload_max_mb * 1024 * 1024
    frames_allowance = settings.companion_frame_max * FRAME_MAX_BYTES
    body_limit = max_bytes + frames_allowance + 4096  # 음성 + 프레임 + multipart 머리 여유분
    too_large = f"음성 파일이 {settings.companion_upload_max_mb}MB를 넘습니다."
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > body_limit:
        raise CompanionError(413, "AUDIO_TOO_LARGE", too_large)
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > body_limit:
            raise CompanionError(413, "AUDIO_TOO_LARGE", too_large)
    ctype = request.headers.get("content-type", "")
    try:
        part = read_file_part(ctype, bytes(body), "audio")
        frames = read_file_parts(ctype, bytes(body), "frame")
    except ValueError as exc:
        raise CompanionError(422, "AUDIO_MISSING", "multipart/form-data 의 audio 파일이 필요합니다.") from exc
    if part is None:
        raise CompanionError(422, "AUDIO_MISSING", "audio 파일이 없습니다.")
    if len(part.data) > max_bytes:
        raise CompanionError(413, "AUDIO_TOO_LARGE", too_large)
    if len(frames) > settings.companion_frame_max:
        raise CompanionError(413, "FRAMES_TOO_MANY", f"프레임은 한 턴에 {settings.companion_frame_max}장까지입니다.")
    if any(len(f.data) > FRAME_MAX_BYTES for f in frames):
        raise CompanionError(413, "FRAME_TOO_LARGE", "프레임 한 장은 1MB 이하여야 합니다.")
    if any(detect_image_format(f.data) is None for f in frames):
        raise CompanionError(422, "FRAME_FORMAT_UNSUPPORTED", "얼굴 사진은 JPEG·PNG만 받습니다.")
    if detect_audio_format(part.data) is None:
        raise CompanionError(422, "AUDIO_FORMAT_UNSUPPORTED", f"{'·'.join(ALLOWED_FORMATS)} 파일만 받습니다.")
    try:
        stt = call_with_timeout(transcribe, settings.companion_stt_timeout_sec, part.data)
    except TimeoutError as exc:  # S9-1: 음성 인식 시간 초과
        raise CompanionError(503, "STT_UNAVAILABLE", "음성 인식 시간이 초과되었습니다.") from exc
    except SttUnavailable as exc:  # 응답에는 모델 폴더 경로·예외 원문을 싣지 않는다(보안 점검 M-6). 원인은 클래스명만 로그에
        stt_logger.warning("turn stt status=503 error=%s", type(exc).__name__)
        raise CompanionError(503, "STT_UNAVAILABLE", _STT_UNAVAILABLE_MESSAGE) from exc
    except Exception as exc:  # noqa: BLE001 - 모델 실행 중 오류(GPU 메모리 등)도 500 대신 503 (11/09 점검 L-2). 로그는 클래스명만
        stt_logger.warning("turn stt status=503 error=%s", type(exc).__name__)
        raise CompanionError(503, "STT_UNAVAILABLE", "음성 인식에 실패했습니다.") from exc
    transcript = stt.text
    # 글자가 없으면 무응답 턴이다 (아키텍처 3·4절: 200 + noResponse). 표정은 보지 않는다.
    # 표정: 프레임이 없으면 사용 불가. 표정 모델이 없거나 시간을 넘겨도 대화는 계속된다 (대화 우선 원칙)
    face = _no_face()
    if transcript.strip() and frames:
        try:
            face = _face_dict(call_with_timeout(analyze_face, settings.companion_face_timeout_sec,
                                                [f.data for f in frames]))
        except (FaceUnavailable, TimeoutError):
            face = _no_face()
    result = _run_turn(d, transcript, face, classify_text, llm, stt_logprob=getattr(stt, "avg_logprob", None))
    out = _turn_response(d, result, transcript=transcript, face=face)
    return success_response("턴 분석 완료", out)


# 챗봇 등과 공유하는 음성 인식 API의 동시 처리 자리 (설정 COMPANION_STT_API_CONCURRENCY)
_stt_slots = threading.BoundedSemaphore(get_settings().companion_stt_api_concurrency)
stt_logger = logging.getLogger("companion.stt_api")
_STT_UNAVAILABLE_MESSAGE = "음성 인식을 지금 사용할 수 없습니다."


class _AudioTooLong(Exception):
    def __init__(self, seconds: float) -> None:
        super().__init__("too long")
        self.seconds = seconds


def _transcribe_with_slot(transcribe_pcm, audio: bytes, max_sec: float, wait_sec: float, timeout_sec: float):
    """자리를 얻어 음성을 풀고 글자로 바꾼다. (결과, 길이 초)를 돌려준다. 자리가 없으면 None. 시간을 넘기면 TimeoutError.

    디코딩도 자리 안에서 한다(2026-11-10 전체 점검 M-2) - 작게 압축된 긴 음성을 여러 개 보내 메모리를 한꺼번에 쓰게 할 수 없다.
    그래서 자리가 없을 때는 읽을 수 없는 음성·너무 긴 음성도 먼저 STT_BUSY 를 받는다(다시 보내면 원래 오류가 나온다).
    자리는 작업이 실제로 끝날 때까지 쥔다. 시간 초과로 응답은 먼저 나가도, 돌고 있는 작업이 끝나야 자리가 풀리므로
    동시에 도는 작업 수가 설정값을 넘지 않는다.
    """
    if not _stt_slots.acquire(timeout=wait_sec):
        return None
    started = False

    def work():
        nonlocal started
        started = True  # 이 순간부터 자리 해제는 work 의 finally 몫
        try:
            pcm = decode_to_pcm16k(audio)  # 읽을 수 없으면 ValueError
            seconds = pcm.size / TARGET_RATE
            if seconds > max_sec:
                raise _AudioTooLong(seconds)
            return transcribe_pcm(pcm), seconds
        finally:
            _stt_slots.release()

    try:
        return call_with_timeout(work, timeout_sec)
    except BaseException as exc:
        # work 가 시작되지 못했고 시간 초과도 아니면 submit 자체가 실패한 것(풀 종료 등) - 자리를 풀고 다시 던진다.
        # 시간 초과는 작업이 대기열에 남아 나중에 work 가 풀므로 여기서 풀지 않는다(이중 해제 방지).
        if not started and not isinstance(exc, TimeoutError):
            _stt_slots.release()
        raise


async def _read_limited_body(request: Request, limit: int, too_large: CompanionError) -> bytes:
    """Content-Length 로 먼저 거절하고, 읽는 중 상한을 넘으면 끝까지 읽지 않고 중단한다(약 봉투 OCR 과 같은 방식)."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise too_large
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            raise too_large
    return bytes(body)


@router.post(
    "/stt",
    summary="음성을 글자로만 바꾼다 (챗봇 공유용). 저장·감정 분석 없음",
    description="응답은 `data.text` (글자만, 못 알아들으면 빈 글자). 음성과 글자는 저장하지 않고 로그에도 남기지 않는다. "
                "WAV·webm·mp3, 10MB·30초 이하, 동시 2건 (자리가 없으면 5초 대기 후 503 STT_BUSY). "
                "오류: 413 AUDIO_TOO_LARGE·AUDIO_TOO_LONG, 422 AUDIO_MISSING·AUDIO_FORMAT_UNSUPPORTED·AUDIO_UNREADABLE, 503 STT_UNAVAILABLE·STT_BUSY.",
)
async def speech_to_text(request: Request, settings: Settings = Depends(get_settings),
                         transcribe_pcm=Depends(get_pcm_transcriber)) -> dict:
    """음성 파일 1개(audio)를 받아 글자만 돌려준다. 음성과 글자는 저장하지 않고, 로그에도 남기지 않는다."""
    started = time.monotonic()
    max_bytes = settings.companion_stt_api_max_mb * 1024 * 1024
    body = await _read_limited_body(request, max_bytes + 4096, CompanionError(
        413, "AUDIO_TOO_LARGE", f"음성 파일이 {settings.companion_stt_api_max_mb}MB를 넘습니다."))
    try:
        part = read_file_part(request.headers.get("content-type", ""), body, "audio")
    except ValueError as exc:
        raise CompanionError(422, "AUDIO_MISSING", "multipart/form-data 의 audio 파일이 필요합니다.") from exc
    if part is None:
        raise CompanionError(422, "AUDIO_MISSING", "audio 파일이 없습니다.")
    if len(part.data) > max_bytes:
        raise CompanionError(413, "AUDIO_TOO_LARGE", f"음성 파일이 {settings.companion_stt_api_max_mb}MB를 넘습니다.")
    if detect_audio_format(part.data) is None:
        raise CompanionError(422, "AUDIO_FORMAT_UNSUPPORTED", f"{'·'.join(ALLOWED_FORMATS)} 파일만 받습니다.")
    size_kb = len(part.data) // 1024
    try:
        outcome = await run_in_threadpool(_transcribe_with_slot, transcribe_pcm, part.data, settings.companion_stt_api_max_sec,
                                          settings.companion_stt_api_wait_sec, settings.companion_stt_api_timeout_sec)
    except _AudioTooLong as exc:
        raise CompanionError(413, "AUDIO_TOO_LONG", f"음성은 {int(settings.companion_stt_api_max_sec)}초 이하여야 합니다.") from exc
    except TimeoutError as exc:  # 받아쓰기가 제한 시간을 넘김 → 백엔드는 503 STT_UNAVAILABLE 로 받는다
        stt_logger.info("stt status=503 kb=%d timeout", size_kb)
        raise CompanionError(503, "STT_UNAVAILABLE", "음성 인식 시간이 초과되었습니다.") from exc
    except SttUnavailable as exc:
        stt_logger.info("stt status=503 kb=%d error=%s", size_kb, type(exc).__name__)
        raise CompanionError(503, "STT_UNAVAILABLE", _STT_UNAVAILABLE_MESSAGE) from exc
    except ValueError as exc:  # 디코딩 실패(decode_to_pcm16k)
        raise CompanionError(422, "AUDIO_UNREADABLE", "음성을 읽을 수 없습니다.") from exc
    except Exception as exc:  # noqa: BLE001 - 모델 실행 중 오류도 503 (11/09 점검 L-2). 로그는 클래스명만
        stt_logger.warning("stt status=503 kb=%d error=%s", size_kb, type(exc).__name__)
        raise CompanionError(503, "STT_UNAVAILABLE", "음성 인식에 실패했습니다.") from exc
    if outcome is None:
        stt_logger.info("stt status=503 kb=%d busy", size_kb)
        raise CompanionError(503, "STT_BUSY", "음성 인식이 바빠요. 잠시 뒤 다시 시도해 주세요.")
    result, seconds = outcome
    # 로그에는 길이·처리 시간·상태만. 글자 내용은 남기지 않는다.
    stt_logger.info("stt status=200 seconds=%.1f elapsed_ms=%d", seconds, int((time.monotonic() - started) * 1000))
    return success_response("음성을 글자로 바꿨습니다.", {"text": result.text})


@router.post(
    "/dialogues/{dialogue_id}/finish",
    summary="대화를 끝내고 결과를 받는다",
    description="대화를 끝낸다 (`USER_STOP`). 응답 `data`에 대화 결과 전체를 담는다 (기분, 턴 수, 종료 이유, 한 줄 요약, 위험 여부, 알림 등급, 질문 정보). "
                "백엔드로 따로 보내지 않는다. 원문은 없다. "
                "오류: 404 DIALOGUE_NOT_FOUND.",
)
def finish_dialogue(dialogue_id: str) -> dict:
    d = store.get(dialogue_id)
    if d is None:
        raise CompanionError(404, "DIALOGUE_NOT_FOUND", "대화를 찾을 수 없습니다.")
    # D16 (사용자 결정): 중간에 그만두면 그때까지의 턴 수와 결과를 그대로 기록한다. 이미 끝난 대화는 이유를 바꾸지 않는다.
    d = store.finish(dialogue_id, "USER_STOP") or d
    # 결과 전체를 응답으로 돌려준다. 백엔드로 따로 보내지 않는다 (백엔드가 finish로 받는다).
    return success_response("대화를 마쳤습니다.", build_dialogue_result(d))
