from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.response import error_response, success_response
from app.database.session import SessionLocal, get_db
from app.schemas.clip_schema import ClipRequest
from app.schemas.stream_session_schema import StreamSessionCreate
from app.services.clip_service import ClipError, clip_service
from app.services.live_ws_manager import live_ws_manager
from app.services.stream_session_service import StreamSessionService, frame_store

router = APIRouter(tags=["LiveStreams"])


@router.post("/api/v1/stream-sessions", summary="iPad 송출 세션 생성")
def create_stream_session(payload: StreamSessionCreate, db: Session = Depends(get_db)) -> dict:
    service = StreamSessionService(db, frame_store)
    session = service.create_or_restart(payload.sessionId, payload.cameraIdentifier, payload.deviceType)
    live_items = service.list_live()
    current = next((item for item in live_items if item["sessionId"] == session.session_id), None)
    status_payload = service.get_status_payload(session.session_id)
    latest_analysis = service.latest_analysis_for_session(session.session_id, session.camera_identifier)
    live_ws_manager.broadcast_nowait(
        {"type": "session_status", "sessionId": session.session_id, "data": status_payload},
        session_id=session.session_id,
    )
    live_ws_manager.broadcast_nowait(
        {"type": "latest_analysis", "sessionId": session.session_id, "data": latest_analysis},
        session_id=session.session_id,
    )
    live_ws_manager.broadcast_nowait({"type": "live_streams", "data": live_items})
    return success_response(
        "송출 세션 생성 완료",
        {
            "sessionId": session.session_id,
            "cameraIdentifier": session.camera_identifier,
            "deviceType": session.device_type,
            "status": session.status,
            "viewerUrl": current["viewerUrl"] if current else f"/api/v1/live-streams/{session.session_id}/mjpeg",
            "hlsUrl": current["hlsUrl"] if current else None,
            "ingestUrl": current["ingestUrl"] if current else None,
        },
    )


@router.post("/api/v1/stream-sessions/{session_id}/frame", summary="iPad 프레임 수신")
async def ingest_stream_frame(
    session_id: str,
    frame: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict:
    if frame.content_type not in {"image/jpeg", "image/jpg"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_response("JPEG 프레임만 허용됩니다.", "STREAM_FRAME_INVALID_TYPE", None),
        )
    service = StreamSessionService(db, frame_store)
    session = service.require_session(session_id)
    frame_bytes = await frame.read()
    if not frame_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_response("프레임 데이터가 비어 있습니다.", "STREAM_FRAME_EMPTY", None),
        )
    updated = service.ingest_frame(session, frame_bytes)
    latest_analysis = await service.analyze_stream_frame_async(updated.session_id, frame_bytes)
    status_payload = service.get_status_payload(updated.session_id)
    live_ws_manager.broadcast_nowait(
        {"type": "session_status", "sessionId": updated.session_id, "data": status_payload},
        session_id=updated.session_id,
    )
    live_ws_manager.broadcast_nowait(
        {"type": "latest_analysis", "sessionId": updated.session_id, "data": latest_analysis},
        session_id=updated.session_id,
    )
    return success_response(
        "프레임 수신 완료",
        {
            "sessionId": updated.session_id,
            "status": updated.status,
            "lastFrameAt": updated.last_frame_at.isoformat() if updated.last_frame_at else None,
        },
    )


@router.post("/api/v1/stream-sessions/{session_id}/stop", summary="송출 세션 종료")
def stop_stream_session(session_id: str, db: Session = Depends(get_db)) -> dict:
    service = StreamSessionService(db, frame_store)
    session = service.require_session(session_id)
    updated = service.stop(session)
    status_payload = service.get_status_payload(updated.session_id)
    live_items = service.list_live()
    live_ws_manager.broadcast_nowait(
        {"type": "session_status", "sessionId": updated.session_id, "data": status_payload},
        session_id=updated.session_id,
    )
    live_ws_manager.broadcast_nowait({"type": "live_streams", "data": live_items})
    return success_response(
        "송출 세션 종료 처리 완료",
        {
            "sessionId": updated.session_id,
            "status": updated.status,
            "stoppedAt": updated.stopped_at.isoformat() if updated.stopped_at else None,
        },
    )


@router.get("/api/v1/live-streams", summary="현재 송출 중인 세션 목록 조회")
def list_live_streams(db: Session = Depends(get_db)) -> dict:
    service = StreamSessionService(db, frame_store)
    items = service.list_live()
    return success_response("라이브 세션 목록 조회 완료", items)


@router.get("/api/v1/live-streams/{session_id}/latest-frame", summary="특정 세션 최신 프레임 조회")
def latest_frame(session_id: str, db: Session = Depends(get_db)) -> Response:
    service = StreamSessionService(db, frame_store)
    _ = service.require_session(session_id)
    frame_bytes = frame_store.get_frame(session_id)
    if not frame_bytes:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_response("최신 프레임이 없습니다.", "STREAM_FRAME_NOT_FOUND", None),
        )
    return Response(content=frame_bytes, media_type="image/jpeg")


@router.get("/api/v1/live-streams/{session_id}/status", summary="특정 세션 상태 조회")
def live_status(session_id: str, db: Session = Depends(get_db)) -> dict:
    service = StreamSessionService(db, frame_store)
    session = service.require_session(session_id)
    viewer_count = frame_store.get_viewer_count(session_id)
    return success_response(
        "세션 상태 조회 완료",
        {
            "sessionId": session.session_id,
            "status": session.status,
            "lastFrameAt": session.last_frame_at.isoformat() if session.last_frame_at else None,
            "fps": frame_store.get_fps(session_id),
            "viewerCount": viewer_count,
            "isAnalyzing": bool(session.is_analyzing),
        },
    )


@router.get("/api/v1/live-streams/{session_id}/latest-analysis", summary="특정 세션 최신 분석 결과 조회")
def latest_analysis(session_id: str, db: Session = Depends(get_db)) -> dict:
    service = StreamSessionService(db, frame_store)
    session = service.require_session(session_id)
    result = service.latest_analysis_for_session(session_id, session.camera_identifier)
    return success_response("최신 분석 결과 조회 완료", result)


@router.get("/api/v1/live-streams/{session_id}/mjpeg", summary="특정 세션 실시간 보기(MJPEG)")
async def stream_mjpeg(session_id: str, db: Session = Depends(get_db)) -> StreamingResponse:
    service = StreamSessionService(db, frame_store)
    _ = service.require_session(session_id)

    async def frame_generator() -> AsyncGenerator[bytes, None]:
        frame_store.increment_viewer(session_id)
        try:
            while True:
                frame_bytes = frame_store.get_frame(session_id)
                if frame_bytes:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )
                await asyncio.sleep(0.2)
        finally:
            frame_store.decrement_viewer(session_id)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


async def _parse_clip_request(request: Request) -> ClipRequest:
    # FastAPI 기본 검증 오류(422)는 errorCode 없는 {"detail": [...]} 로 나가므로 본문을 직접 검증한다.
    raw = await request.body()
    try:
        payload = json.loads(raw) if raw.strip() else {}
        if not isinstance(payload, dict):
            raise ValueError("body must be an object")
        return ClipRequest.model_validate(payload)
    except (ValueError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=error_response("클립 요청 파라미터가 올바르지 않습니다.", "CLIP_INVALID_PARAMS", None),
        ) from None


def _require_stream_session(session_id: str) -> None:
    # 클립은 최대 20초 걸린다 — 요청 내내 DB 세션을 붙들지 않게 확인만 하고 바로 닫는다.
    db = SessionLocal()
    try:
        StreamSessionService(db, frame_store).require_session(session_id)
    finally:
        db.close()


@router.post(
    "/api/v1/live-streams/{session_id}/clips",
    summary="이상감지 클립 생성(WebM, 감지 앞 3초 + 뒤 2초)",
    response_class=Response,
    responses={200: {"content": {"video/webm": {}}}},
    openapi_extra={
        "requestBody": {
            "required": False,
            "content": {"application/json": {"schema": ClipRequest.model_json_schema()}},
        },
    },
)
async def create_clip(session_id: str, request: Request) -> Response:
    if not get_settings().clip_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=error_response("클립 기능이 꺼져 있습니다.", "CLIP_DISABLED", None),
        )
    params = await _parse_clip_request(request)
    _require_stream_session(session_id)
    try:
        result = await clip_service.create_clip(session_id, params)
    except ClipError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=error_response(exc.message, exc.error_code, None),
        ) from None
    return Response(
        content=result.data,
        media_type="video/webm",
        headers={
            "X-Clip-Duration-Ms": str(result.duration_ms),
            "X-Clip-Frames": str(result.frames),
            "X-Clip-Width": str(result.width),
            "X-Clip-Height": str(result.height),
            "X-Clip-Started-At": result.started_at.isoformat(timespec="milliseconds") + "Z",
            "Cache-Control": "no-store",
        },
    )
