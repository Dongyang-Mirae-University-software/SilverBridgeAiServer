from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter

from app.core.response import success_response

router = APIRouter(tags=["Health"])


@router.get("/health")
def health() -> dict:
    return success_response(
        message="AI 서버가 정상 동작 중입니다.",
        data={"status": "ok", "timestamp": datetime.utcnow().isoformat()},
    )
