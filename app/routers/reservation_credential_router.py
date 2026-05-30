from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
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
