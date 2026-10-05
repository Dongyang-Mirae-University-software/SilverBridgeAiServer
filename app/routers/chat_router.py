from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.response import error_response, success_response
from app.database.session import get_db
from app.schemas.chat_schema import ChatRequest
from app.services.chat_service import ChatService

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])
chat_service = ChatService()


def _require_user_id(user_id: str | None) -> None:
    if user_id is None or not user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=error_response("userId 가 필요합니다.", "CHAT_USER_ID_REQUIRED", None),
        )


@router.post("", summary="의료 챗 상담 (대화 이력 + MedGemma)")
def chat(payload: ChatRequest, db: Session = Depends(get_db)) -> dict:
    try:
        result = chat_service.process_message(db, payload)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_response(str(exc), "CHAT_INVALID_REQUEST", None),
        ) from exc
    return success_response("AI 응답 생성 완료", result)


@router.get("/logs", summary="챗 로그 목록 조회")
def list_logs(userId: str | None = Query(default=None, max_length=64), db: Session = Depends(get_db)) -> dict:
    # userId 없이 부르면 전 회원의 상담 기록이 한꺼번에 나갔다 — 본인 것만 돌려주도록 필수로 한다.
    # FE 는 항상 userId 를 붙여 부르고 빈 값이면 호출하지 않아 하위 호환이다.
    _require_user_id(userId)
    return success_response("챗 로그 조회 완료", chat_service.list_logs(db, userId))


@router.get("/logs/{chat_id}", summary="챗 로그 상세 조회")
def get_log(
    chat_id: int,
    userId: str | None = Query(default=None, max_length=64),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> dict:
    # userId 가 오면 소유자 일치일 때만 준다(불일치도 404 — 남의 기록이 있는지 드러내지 않는다).
    # chat_id 는 자동 증가 번호라 추측 가능하다 → CHAT_REQUIRE_USER_ID 를 켜면 userId 자체를 필수로 한다.
    if settings.chat_require_user_id:
        _require_user_id(userId)
    log = chat_service.get_log(db, chat_id, userId)
    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_response("챗 로그를 찾을 수 없습니다.", "CHAT_LOG_NOT_FOUND", None),
        )
    return success_response("챗 로그 상세 조회 완료", log)
