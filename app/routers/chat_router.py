from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.response import error_response, success_response
from app.database.session import get_db
from app.schemas.chat_schema import ChatRequest
from app.services.chat_service import ChatService

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])
chat_service = ChatService()


@router.post("", summary="의료 챗 요청")
def chat(payload: ChatRequest, db: Session = Depends(get_db)) -> dict:
    result = chat_service.process_message(db, payload)
    return success_response("AI 응답 생성 완료", result)


@router.get("/logs", summary="챗 로그 목록 조회")
def list_logs(db: Session = Depends(get_db)) -> dict:
    return success_response("챗 로그 조회 완료", chat_service.list_logs(db))


@router.get("/logs/{chat_id}", summary="챗 로그 상세 조회")
def get_log(chat_id: int, db: Session = Depends(get_db)) -> dict:
    log = chat_service.get_log(db, chat_id)
    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_response("챗 로그를 찾을 수 없습니다.", "CHAT_LOG_NOT_FOUND", None),
        )
    return success_response("챗 로그 상세 조회 완료", log)
