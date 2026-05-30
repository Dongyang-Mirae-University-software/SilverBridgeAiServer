from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.response import error_response, success_response
from app.database.session import SessionLocal, get_db
from app.models.analysis_result import AnalysisResult
from app.schemas.analysis_schema import AnalysisImageRequest, AnalysisStreamControlRequest
from app.services.camera_service import CameraService
from app.services.detection_service import DetectionService
from app.services.model_service import ModelService
from app.services.stream_service import stream_analysis_manager
from app.utils.file_utils import make_snapshot_path

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

router = APIRouter(prefix="/api/v1/analysis", tags=["Analysis"])
settings = get_settings()


def _serialize(entity: AnalysisResult) -> dict:
    return {
        "id": entity.id,
        "analysisNo": entity.analysis_no,
        "cameraId": entity.camera_id,
        "cameraIdentifier": entity.camera_identifier,
        "modelId": entity.model_id,
        "modelIdentifier": entity.model_identifier,
        "detectedType": entity.detected_type,
        "confidence": entity.confidence,
        "danger": entity.danger,
        "snapshotPath": entity.snapshot_path,
        "rawResultJson": entity.raw_result_json,
        "analyzedAt": entity.analyzed_at.isoformat(),
        "createdAt": entity.created_at.isoformat(),
    }


@router.post("/image", summary="단일 이미지 분석")
def analyze_image(payload: AnalysisImageRequest, db: Session = Depends(get_db)) -> dict:
    camera_service = CameraService(db)
    model_service = ModelService(db)
    camera = camera_service.get_by_identifier(payload.cameraIdentifier)
    model = model_service.get_by_identifier(payload.modelIdentifier)
    if not camera:
        raise HTTPException(status_code=404, detail=error_response("카메라를 찾을 수 없습니다.", "CAMERA_NOT_FOUND", None))
    if not model:
        raise HTTPException(status_code=404, detail=error_response("모델을 찾을 수 없습니다.", "MODEL_NOT_FOUND", None))

    frame_available = False
    snapshot_path = ""
    if payload.imagePath and Path(payload.imagePath).exists():
        frame_available = True
        snapshot_path = payload.imagePath
    elif cv2 is not None:
        cap = cv2.VideoCapture(camera.stream_url)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        frame_available = bool(ok and frame is not None)
        if frame_available:
            snapshot_path = make_snapshot_path(settings.snapshot_base_path)
            cv2.imwrite(snapshot_path, frame)

    result = DetectionService.detect_from_frame(model, frame_available=frame_available)
    entity = AnalysisResult(
        analysis_no=result["analysisNo"],
        camera_id=camera.id,
        camera_identifier=camera.identifier,
        model_id=model.id,
        model_identifier=model.identifier,
        detected_type=result["detectedType"],
        confidence=result["confidence"],
        danger=result["danger"],
        snapshot_path=snapshot_path,
        raw_result_json=json.dumps(result["rawResultJson"], ensure_ascii=False),
        analyzed_at=datetime.utcnow(),
    )
    db.add(entity)
    db.commit()
    db.refresh(entity)
    return success_response("단일 이미지 분석 완료", _serialize(entity))


@router.post("/start", summary="카메라 스트림 분석 시작")
def start_stream_analysis(payload: AnalysisStreamControlRequest, db: Session = Depends(get_db)) -> dict:
    camera_service = CameraService(db)
    model_service = ModelService(db)
    camera = camera_service.get_by_identifier(payload.cameraIdentifier)
    model = model_service.get_by_identifier(payload.modelIdentifier)
    if not camera:
        raise HTTPException(status_code=404, detail=error_response("카메라를 찾을 수 없습니다.", "CAMERA_NOT_FOUND", None))
    if not model:
        raise HTTPException(status_code=404, detail=error_response("모델을 찾을 수 없습니다.", "MODEL_NOT_FOUND", None))

    started = stream_analysis_manager.start(SessionLocal, camera, model)
    if not started:
        return success_response("이미 분석이 실행 중입니다.", {"cameraIdentifier": camera.identifier, "running": True})
    return success_response("스트림 분석 시작", {"cameraIdentifier": camera.identifier, "modelIdentifier": model.identifier, "running": True})


@router.post("/stop", summary="카메라 스트림 분석 중지")
def stop_stream_analysis(payload: AnalysisStreamControlRequest) -> dict:
    stopped = stream_analysis_manager.stop(payload.cameraIdentifier)
    if not stopped:
        raise HTTPException(status_code=404, detail=error_response("분석 중인 세션을 찾을 수 없습니다.", "ANALYSIS_NOT_RUNNING", None))
    return success_response("스트림 분석 중지 요청 완료", {"cameraIdentifier": payload.cameraIdentifier, "running": False})


@router.get("/status/{camera_identifier}", summary="특정 카메라 분석 상태 조회")
def analysis_status(camera_identifier: str) -> dict:
    running = stream_analysis_manager.is_running(camera_identifier)
    return success_response("분석 상태 조회 완료", {"cameraIdentifier": camera_identifier, "running": running})


@router.get("/latest/{camera_identifier}", summary="특정 카메라 최신 분석 결과 조회")
def latest_analysis(camera_identifier: str, db: Session = Depends(get_db)) -> dict:
    latest = (
        db.query(AnalysisResult)
        .filter(AnalysisResult.camera_identifier == camera_identifier)
        .order_by(AnalysisResult.id.desc())
        .first()
    )
    if not latest:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_response("분석 결과가 없습니다.", "ANALYSIS_NOT_FOUND", None),
        )
    return success_response("최신 분석 결과 조회 완료", _serialize(latest))


@router.get("/results", summary="분석 결과 목록 조회")
def list_analysis_results(
    cameraIdentifier: str | None = Query(default=None),
    startAt: str | None = Query(default=None),
    endAt: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> dict:
    query = db.query(AnalysisResult)
    if cameraIdentifier:
        query = query.filter(AnalysisResult.camera_identifier == cameraIdentifier)
    if startAt:
        query = query.filter(AnalysisResult.analyzed_at >= datetime.fromisoformat(startAt))
    if endAt:
        query = query.filter(AnalysisResult.analyzed_at <= datetime.fromisoformat(endAt))
    rows = query.order_by(AnalysisResult.id.desc()).all()
    return success_response("분석 결과 목록 조회 완료", [_serialize(row) for row in rows])


@router.get("/results/{result_id}", summary="분석 결과 상세 조회")
def get_analysis_result(result_id: int, db: Session = Depends(get_db)) -> dict:
    entity = db.query(AnalysisResult).filter(AnalysisResult.id == result_id).first()
    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_response("분석 결과를 찾을 수 없습니다.", "ANALYSIS_NOT_FOUND", None),
        )
    return success_response("분석 결과 조회 완료", _serialize(entity))
