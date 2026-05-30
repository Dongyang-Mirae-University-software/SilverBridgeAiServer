from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.response import error_response, success_response
from app.database.session import get_db
from app.schemas.camera_schema import CameraCreate, CameraUpdate
from app.services.camera_service import CameraService

router = APIRouter(prefix="/api/v1/cameras", tags=["Camera"])


def _serialize(entity) -> dict:
    return {
        "id": entity.id,
        "cameraNo": entity.camera_no,
        "identifier": entity.identifier,
        "name": entity.name,
        "streamUrl": entity.stream_url,
        "streamType": entity.stream_type,
        "targetUserId": entity.target_user_id,
        "guardianUserId": entity.guardian_user_id,
        "locationName": entity.location_name,
        "isActive": entity.is_active,
        "createdAt": entity.created_at.isoformat(),
        "updatedAt": entity.updated_at.isoformat(),
    }


@router.post("", summary="카메라 등록")
def create_camera(payload: CameraCreate, db: Session = Depends(get_db)) -> dict:
    service = CameraService(db)
    if not service.validate_stream_url(payload.streamUrl):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_response("유효하지 않은 streamUrl 입니다.", "CAMERA_STREAM_INVALID", None),
        )
    if service.get_by_identifier(payload.identifier):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=error_response("이미 등록된 카메라입니다.", "CAMERA_ALREADY_EXISTS", None),
        )
    entity = service.create(payload)
    return success_response("카메라 등록 완료", _serialize(entity))


@router.get("", summary="카메라 목록 조회")
def list_cameras(db: Session = Depends(get_db)) -> dict:
    service = CameraService(db)
    return success_response("카메라 목록 조회 완료", [_serialize(item) for item in service.list()])


@router.get("/by-identifier/{identifier}", summary="identifier 기준 카메라 조회")
def get_camera_by_identifier(identifier: str, db: Session = Depends(get_db)) -> dict:
    service = CameraService(db)
    entity = service.get_by_identifier(identifier)
    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_response("카메라를 찾을 수 없습니다.", "CAMERA_NOT_FOUND", None),
        )
    return success_response("카메라 조회 완료", _serialize(entity))


@router.get("/{camera_id}", summary="카메라 상세 조회")
def get_camera(camera_id: int, db: Session = Depends(get_db)) -> dict:
    service = CameraService(db)
    entity = service.get(camera_id)
    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_response("카메라를 찾을 수 없습니다.", "CAMERA_NOT_FOUND", None),
        )
    return success_response("카메라 조회 완료", _serialize(entity))


@router.patch("/{camera_id}", summary="카메라 수정")
def update_camera(camera_id: int, payload: CameraUpdate, db: Session = Depends(get_db)) -> dict:
    service = CameraService(db)
    entity = service.get(camera_id)
    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_response("카메라를 찾을 수 없습니다.", "CAMERA_NOT_FOUND", None),
        )
    if payload.streamUrl and not service.validate_stream_url(payload.streamUrl):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_response("유효하지 않은 streamUrl 입니다.", "CAMERA_STREAM_INVALID", None),
        )
    updated = service.update(entity, payload)
    return success_response("카메라 수정 완료", _serialize(updated))


@router.delete("/{camera_id}", summary="카메라 삭제/비활성화")
def delete_camera(camera_id: int, db: Session = Depends(get_db)) -> dict:
    service = CameraService(db)
    entity = service.get(camera_id)
    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_response("카메라를 찾을 수 없습니다.", "CAMERA_NOT_FOUND", None),
        )
    deactivated = service.deactivate(entity)
    return success_response("카메라 비활성화 완료", _serialize(deactivated))


@router.post("/{camera_id}/test-connection", summary="카메라 연결 테스트")
def test_connection(camera_id: int, db: Session = Depends(get_db)) -> dict:
    service = CameraService(db)
    entity = service.get(camera_id)
    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_response("카메라를 찾을 수 없습니다.", "CAMERA_NOT_FOUND", None),
        )
    is_valid = service.validate_stream_url(entity.stream_url)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_response("스트림 연결에 실패했습니다.", "CAMERA_STREAM_INVALID", None),
        )
    return success_response("카메라 연결 테스트 성공", {"cameraIdentifier": entity.identifier, "connected": True})
