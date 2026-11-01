from __future__ import annotations

import logging
import threading
import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.core.response import error_response, success_response
from app.database.session import SessionLocal, get_db
from app.schemas.checkin_schema import ID_PATTERN, CheckinFinish, CheckinSessionStart
from app.services.checkin import service
from app.services.checkin.config import get_checkin_settings
from app.services.checkin.models import CheckinBusy

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/checkin", tags=["Checkin (안부 말벗)"])

_purge_lock = threading.Lock()
_last_purge = 0.0


def _fail(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail=error_response(message, code, None))


def _require_enabled() -> None:
    if not get_checkin_settings().enabled:
        raise _fail(503, "CHECKIN_DISABLED", "안부 말벗 기능이 꺼져 있습니다.")


def _maybe_purge_expired() -> None:
    """보관 기간(최대 30일)이 지난 대화를 한 시간에 한 번 지운다. 실패해도 요청은 막지 않는다."""
    global _last_purge
    if time.monotonic() - _last_purge < 3600 or not _purge_lock.acquire(blocking=False):
        return
    try:
        _last_purge = time.monotonic()
        with SessionLocal() as db:
            n = service.purge_expired(db)
        if n["sessions"]:
            log.info("[CHECKIN] retention purge sessions=%d turns=%d", n["sessions"], n["turns"])
    except Exception as e:  # noqa: BLE001
        log.warning("[CHECKIN] retention purge failed type=%s", type(e).__name__)
    finally:
        _purge_lock.release()


def _run(fn, *args):
    try:
        return fn(*args)
    except service.CheckinError as e:
        raise _fail(e.status, e.code, e.message) from None
    except CheckinBusy:
        raise _fail(429, "CHECKIN_BUSY", "지금 다른 대화를 처리하고 있어요. 잠시 후 다시 시도해 주세요.") from None


async def _read_limited(upload: UploadFile, limit: int, code: str) -> bytes:
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise _fail(413, code, "파일이 너무 큽니다.")
    return data


@router.post("/sessions", summary="안부 대화 시작 (세션 ID 는 백엔드가 만든다)", dependencies=[Depends(_require_enabled)])
def start_session(body: CheckinSessionStart, db: Session = Depends(get_db)) -> dict:
    _maybe_purge_expired()
    return success_response("대화를 시작했습니다.", _run(service.start, db, body.sessionId, body.userId, body.slot, body.question))


@router.post("/sessions/{session_id}/turns", summary="대답 한 번 (음성 + 선택 사진) -> 받아쓴 글자·공감·꼬리질문",
             dependencies=[Depends(_require_enabled)])
async def add_turn(
    session_id: str,
    userId: str = Form(..., pattern=ID_PATTERN),
    turnNo: int = Form(..., ge=1, le=12),
    audio: UploadFile = File(...),
    frames: list[UploadFile] | None = File(default=None),
    db: Session = Depends(get_db),
) -> dict:
    cfg = get_checkin_settings()
    raw = await _read_limited(audio, cfg.max_audio_bytes, "AUDIO_TOO_LARGE")
    if not raw:
        raise _fail(422, "AUDIO_EMPTY", "음성이 비어 있습니다.")
    frames = frames or []
    if len(frames) > cfg.max_frames:
        raise _fail(413, "FRAMES_TOO_MANY", "사진이 너무 많습니다.")
    images = []
    for f in frames:
        data = await _read_limited(f, cfg.max_frame_bytes, "FRAME_TOO_LARGE")
        if data[:2] != b"\xff\xd8":
            raise _fail(422, "FRAME_NOT_JPEG", "사진은 JPEG 만 받습니다.")
        images.append(data)
    # 무거운 처리(받아쓰기·생성)는 이벤트 루프 밖 스레드에서 한다
    from starlette.concurrency import run_in_threadpool

    result = await run_in_threadpool(_run, service.turn, db, session_id, userId, turnNo, raw, images)
    return success_response("대답을 처리했습니다.", result)


@router.post("/sessions/{session_id}/finish", summary="대화 끝 -> 정서·생활 신호·요약 (같은 요청에 같은 답)",
             dependencies=[Depends(_require_enabled)])
def finish_session(session_id: str, body: CheckinFinish, db: Session = Depends(get_db)) -> dict:
    return success_response("대화를 정리했습니다.", _run(service.finish, db, session_id, body.userId, body.selfReport, body.reason))


@router.delete("/logs", summary="회원의 안부 대화 기록 전체 삭제 (회원 탈퇴 정리용)")
def delete_logs(userId: str | None = Query(default=None, max_length=64), db: Session = Depends(get_db)) -> dict:
    # 킬 스위치가 꺼져 있어도 삭제는 동작해야 한다. userId 없이 부르면 전원 삭제가 되므로 필수.
    if not userId or not userId.strip():
        raise _fail(422, "CHECKIN_USER_ID_REQUIRED", "userId 가 필요합니다.")
    return success_response("안부 대화 기록을 삭제했습니다.", service.purge_user(db, userId.strip()))
