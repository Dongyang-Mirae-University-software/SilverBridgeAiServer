from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.response import error_response, success_response
from app.database.session import get_db
from app.schemas.reservation_credential_schema import (
    ReservationCredentialUpsertRequest,
)
from app.services.reservation_credential_service import get_reservation_credential_service

router = APIRouter(prefix="/api/v1/reservation-credentials", tags=["Reservation Credentials"])


@router.post("", summary="예약 API 키 저장")
def upsert_reservation_credential(
    payload: ReservationCredentialUpsertRequest,
    db: Session = Depends(get_db),
) -> dict:
    service = get_reservation_credential_service()
    try:
        result = service.upsert_credential(
            db,
            user_id=payload.userId,
            reservation_email=payload.reservationEmail,
            api_key=payload.apiKey,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_response(str(exc), "RESERVATION_CREDENTIAL_INVALID", None),
        ) from exc
    return success_response("예약 API 키가 저장되었습니다.", result.__dict__)


@router.delete("", summary="회원의 예약 API 키 삭제 (회원 탈퇴 정리용)")
def delete_reservation_credential(
    userId: str | None = Query(default=None, max_length=64),
    db: Session = Depends(get_db),
) -> dict:
    # userId 없이 부르면 의미가 모호하므로 반드시 필수로 한다(키 값은 로그·응답에 싣지 않는다).
    if userId is None or not userId.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=error_response("userId 가 필요합니다.", "RESERVATION_CREDENTIAL_USER_ID_REQUIRED", None),
        )
    deleted = get_reservation_credential_service().delete_credential(db, userId.strip())
    return success_response("예약 API 키 삭제 완료", {"deleted": deleted})
