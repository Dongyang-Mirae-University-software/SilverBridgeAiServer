from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.response import error_response, success_response
from app.database.session import get_db
from app.schemas.ai_model_schema import AIModelCreate, AIModelUpdate
from app.services.model_service import ModelService

router = APIRouter(prefix="/api/v1/models", tags=["Model"])


def _serialize(entity) -> dict:
    return {
        "id": entity.id,
        "modelNo": entity.model_no,
        "identifier": entity.identifier,
        "name": entity.name,
        "type": entity.type,
        "filePath": entity.file_path,
        "framework": entity.framework,
        "version": entity.version,
        "description": entity.description,
        "threshold": entity.threshold,
        "isActive": entity.is_active,
        "createdAt": entity.created_at.isoformat(),
        "updatedAt": entity.updated_at.isoformat(),
    }


@router.post("", summary="모델 등록")
def create_model(payload: AIModelCreate, db: Session = Depends(get_db)) -> dict:
    service = ModelService(db)
    if service.get_by_identifier(payload.identifier) or service.get_by_model_no(payload.modelNo):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=error_response("이미 등록된 모델입니다.", "MODEL_ALREADY_EXISTS", None),
        )
    entity = service.create(payload)
    return success_response("모델 등록 완료", _serialize(entity))


@router.get("", summary="모델 목록 조회")
def list_models(db: Session = Depends(get_db)) -> dict:
    service = ModelService(db)
    return success_response("모델 목록 조회 완료", [_serialize(item) for item in service.list()])


@router.get("/by-identifier/{identifier}", summary="identifier 기준 모델 조회")
def get_model_by_identifier(identifier: str, db: Session = Depends(get_db)) -> dict:
    service = ModelService(db)
    entity = service.get_by_identifier(identifier)
    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_response("모델을 찾을 수 없습니다.", "MODEL_NOT_FOUND", None),
        )
    return success_response("모델 조회 완료", _serialize(entity))


@router.get("/by-model-no/{model_no}", summary="modelNo 기준 모델 조회")
def get_model_by_model_no(model_no: str, db: Session = Depends(get_db)) -> dict:
    service = ModelService(db)
    entity = service.get_by_model_no(model_no)
    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_response("모델을 찾을 수 없습니다.", "MODEL_NOT_FOUND", None),
        )
    return success_response("모델 조회 완료", _serialize(entity))


@router.get("/{model_id}", summary="모델 상세 조회")
def get_model(model_id: int, db: Session = Depends(get_db)) -> dict:
    service = ModelService(db)
    entity = service.get(model_id)
    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_response("모델을 찾을 수 없습니다.", "MODEL_NOT_FOUND", None),
        )
    return success_response("모델 조회 완료", _serialize(entity))


@router.patch("/{model_id}", summary="모델 수정")
def update_model(model_id: int, payload: AIModelUpdate, db: Session = Depends(get_db)) -> dict:
    service = ModelService(db)
    entity = service.get(model_id)
    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_response("모델을 찾을 수 없습니다.", "MODEL_NOT_FOUND", None),
        )
    updated = service.update(entity, payload)
    return success_response("모델 수정 완료", _serialize(updated))


@router.delete("/{model_id}", summary="모델 삭제/비활성화")
def delete_model(model_id: int, db: Session = Depends(get_db)) -> dict:
    service = ModelService(db)
    entity = service.get(model_id)
    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_response("모델을 찾을 수 없습니다.", "MODEL_NOT_FOUND", None),
        )
    deactivated = service.deactivate(entity)
    return success_response("모델 비활성화 완료", _serialize(deactivated))


@router.patch("/{model_id}/activate", summary="특정 모델 활성화")
def activate_model(model_id: int, db: Session = Depends(get_db)) -> dict:
    service = ModelService(db)
    entity = service.get(model_id)
    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_response("모델을 찾을 수 없습니다.", "MODEL_NOT_FOUND", None),
        )
    activated = service.activate(entity)
    return success_response("모델 활성화 완료", _serialize(activated))
