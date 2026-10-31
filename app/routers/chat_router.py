from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.response import error_response, success_response
from app.database.session import get_db
from app.schemas.chat_schema import ChatRequest
from app.services.chat_service import ChatService
from app.services.chat_stats_service import ChatStatsRangeError, ChatStatsService, parse_range
from app.services.reservation_credential_service import mark_proxy_purged, mark_purged

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


@router.get("/stats", summary="챗 상담 전체 집계 (관리자 통계용, 개수만)")
def stats(
    from_: str | None = Query(default=None, alias="from"),
    to: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> dict:
    # KST 날짜(둘 다 포함, 최대 31일). 상담 원문·userId 는 응답에 싣지 않는다 - 숫자만.
    try:
        start, end = parse_range(from_, to)
    except ChatStatsRangeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_response(str(exc), "CHAT_STATS_INVALID_RANGE", None),
        ) from exc
    return success_response("챗 집계 조회 완료", ChatStatsService.stats(db, start, end))


@router.get("/logs", summary="챗 로그 목록 조회")
def list_logs(userId: str | None = Query(default=None, max_length=64), db: Session = Depends(get_db)) -> dict:
    # userId 없이 부르면 전 회원의 상담 기록이 한꺼번에 나갔다 — 본인 것만 돌려주도록 필수로 한다.
    # FE 는 항상 userId 를 붙여 부르고 빈 값이면 호출하지 않아 하위 호환이다.
    _require_user_id(userId)
    return success_response("챗 로그 조회 완료", chat_service.list_logs(db, userId))


@router.delete("/logs", summary="회원의 챗 로그 전체 삭제 (회원 탈퇴 정리용)")
def delete_logs(userId: str | None = Query(default=None, max_length=64), db: Session = Depends(get_db)) -> dict:
    # userId 없이 부르면 전 회원 기록이 지워질 수 있으므로 반드시 필수로 한다(본문·로그에는 상담 내용을 싣지 않는다).
    _require_user_id(userId)
    mark_purged(userId.strip())  # 진행 중이던 상담 턴이 정리 뒤 기록을 새로 남기지 않게(2026-10-30 통합 점검 L-1)
    own, proxy = chat_service.delete_logs(db, userId.strip())
    # deleted = 본인 기록 + 이 회원을 대신한 보호자 대리 상담 기록(proxyDeleted) - 탈퇴 정리 점검 M-1.
    return success_response("챗 로그 삭제 완료", {"deleted": own + proxy, "proxyDeleted": proxy})


@router.delete("/logs/proxy", summary="보호자의 특정 피보호자 대리 상담 기록 삭제 (연결 해제 정리용)")
def delete_proxy_logs(
    guardianId: str | None = Query(default=None, max_length=64),
    wardId: str | None = Query(default=None, max_length=64),
    db: Session = Depends(get_db),
) -> dict:
    # 연결이 끝나면(해제·역할 변경·관리자 해제) 그 보호자가 피보호자 이름·전화·건강 대화를 담아 남긴 대리 상담 기록을 지운다
    # (BE 2026-10 전체 점검 L-13). 둘 다 필수 - 하나라도 비면 범위가 넓어진다. 보호자 본인 상담 기록은 남긴다.
    _require_user_id(guardianId)
    _require_user_id(wardId)
    mark_proxy_purged(guardianId.strip(), wardId.strip())  # 진행 중이던 대리 상담 턴이 기록을 다시 남기지 않게
    deleted = chat_service.delete_proxy_logs(db, guardianId.strip(), wardId.strip())
    return success_response("대리 상담 기록 삭제 완료", {"deleted": deleted})


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
