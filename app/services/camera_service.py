from __future__ import annotations

from urllib.parse import urlparse

from sqlalchemy.orm import Session

from app.models.camera import Camera
from app.schemas.camera_schema import CameraCreate, CameraUpdate


class CameraService:
    def __init__(self, db: Session) -> None:
        self.db = db

    @staticmethod
    def validate_stream_url(stream_url: str) -> bool:
        parsed = urlparse(stream_url)
        return parsed.scheme in {"rtsp", "http", "https"} and bool(parsed.netloc)

    def create(self, payload: CameraCreate) -> Camera:
        entity = Camera(
            camera_no=payload.cameraNo,
            identifier=payload.identifier,
            name=payload.name,
            stream_url=payload.streamUrl,
            stream_type=payload.streamType,
            target_user_id=payload.targetUserId,
            guardian_user_id=payload.guardianUserId,
            location_name=payload.locationName,
            is_active=payload.isActive,
        )
        self.db.add(entity)
        self.db.commit()
        self.db.refresh(entity)
        return entity

    def list(self) -> list[Camera]:
        return self.db.query(Camera).order_by(Camera.id.desc()).all()

    def get(self, camera_id: int) -> Camera | None:
        return self.db.query(Camera).filter(Camera.id == camera_id).first()

    def get_by_identifier(self, identifier: str) -> Camera | None:
        return self.db.query(Camera).filter(Camera.identifier == identifier).first()

    def update(self, entity: Camera, payload: CameraUpdate) -> Camera:
        updates = payload.model_dump(exclude_unset=True)
        mapping = {
            "streamUrl": "stream_url",
            "streamType": "stream_type",
            "targetUserId": "target_user_id",
            "guardianUserId": "guardian_user_id",
            "locationName": "location_name",
            "isActive": "is_active",
        }
        for key, value in updates.items():
            setattr(entity, mapping.get(key, key), value)
        self.db.add(entity)
        self.db.commit()
        self.db.refresh(entity)
        return entity

    def deactivate(self, entity: Camera) -> Camera:
        entity.is_active = False
        self.db.add(entity)
        self.db.commit()
        self.db.refresh(entity)
        return entity
