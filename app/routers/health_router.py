from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.response import success_response
from app.services.clip_service import ffmpeg_available
from app.services.fall_detection_service import get_fall_detector
from app.services.fire_smoke_detection_service import get_fire_smoke_detector
from app.services.knife_detection_service import get_knife_detector

router = APIRouter(tags=["Health"])


def _detector_status() -> dict:
    # /health 는 인증 없이 열려 있다 — 켜짐·로드 여부만 내리고 load_error(파일 경로 포함)는 싣지 않는다.
    settings = get_settings()
    return {
        "fire": {"enabled": settings.fire_smoke_enabled, "loaded": get_fire_smoke_detector().loaded},
        "knife": {"enabled": settings.knife_enabled, "loaded": get_knife_detector().loaded},
        "fall": {"enabled": settings.fall_enabled, "loaded": get_fall_detector().loaded},
    }


@router.get("/health")
def health() -> dict:
    return success_response(
        message="AI 서버가 정상 동작 중입니다.",
        data={
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "ffmpeg": ffmpeg_available(),
            "detectors": _detector_status(),
        },
    )
