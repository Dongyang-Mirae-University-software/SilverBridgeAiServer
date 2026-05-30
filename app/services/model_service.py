from __future__ import annotations

from sqlalchemy.orm import Session

from app.models.ai_model import AIModel
from app.schemas.ai_model_schema import AIModelCreate, AIModelUpdate


class ModelService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(self, payload: AIModelCreate) -> AIModel:
        entity = AIModel(
            model_no=payload.modelNo,
            identifier=payload.identifier,
            name=payload.name,
            type=payload.type,
            file_path=payload.filePath,
            framework=payload.framework,
            version=payload.version,
            description=payload.description,
            threshold=payload.threshold,
            is_active=payload.isActive,
        )
        self.db.add(entity)
        self.db.commit()
        self.db.refresh(entity)
        return entity

    def list(self) -> list[AIModel]:
        return self.db.query(AIModel).order_by(AIModel.id.desc()).all()

    def get(self, model_id: int) -> AIModel | None:
        return self.db.query(AIModel).filter(AIModel.id == model_id).first()

    def get_by_identifier(self, identifier: str) -> AIModel | None:
        return self.db.query(AIModel).filter(AIModel.identifier == identifier).first()

    def get_by_model_no(self, model_no: str) -> AIModel | None:
        return self.db.query(AIModel).filter(AIModel.model_no == model_no).first()

    def update(self, entity: AIModel, payload: AIModelUpdate) -> AIModel:
        updates = payload.model_dump(exclude_unset=True)
        mapping = {
            "filePath": "file_path",
            "isActive": "is_active",
        }
        for key, value in updates.items():
            target = mapping.get(key, key)
            setattr(entity, target, value)
        self.db.add(entity)
        self.db.commit()
        self.db.refresh(entity)
        return entity

    def deactivate(self, entity: AIModel) -> AIModel:
        entity.is_active = False
        self.db.add(entity)
        self.db.commit()
        self.db.refresh(entity)
        return entity

    def activate(self, entity: AIModel) -> AIModel:
        entity.is_active = True
        self.db.add(entity)
        self.db.commit()
        self.db.refresh(entity)
        return entity
