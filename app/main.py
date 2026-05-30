from __future__ import annotations

import gc
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.response import error_response
from app.core.security import require_api_key
from app.database.base import Base
from app.database.session import engine
from app.routers.analysis_router import router as analysis_router
from app.routers.camera_router import router as camera_router
from app.routers.chat_router import router as chat_router
from app.routers.game_router import router as game_router
from app.routers.health_router import router as health_router
from app.routers.live_stream_router import router as live_stream_router
from app.routers.live_ws_router import router as live_ws_router
from app.routers.model_router import router as model_router
from app.routers.reservation_credential_router import router as reservation_credential_router
import app.models.game  # noqa: F401
import app.models.reservation_credential  # noqa: F401
from app.services.fire_smoke_detection_service import get_fire_smoke_detector
from app.services.medical_llm_service import get_medgemma_loader
from app.utils.file_utils import ensure_directory
from app.utils.logger import setup_logging

settings = get_settings()
setup_logging(settings.log_level)
_LOG = logging.getLogger(__name__)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def log_gpu_status() -> None:
    if torch is None:
        raise RuntimeError("torch 모듈을 불러오지 못했습니다.")

    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda or "unknown"
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    gpu_name = torch.cuda.get_device_name(0) if cuda_available and gpu_count > 0 else "none"

    logger_message = (
        "[GPU]\n"
        f"CUDA Available: {cuda_available}\n"
        f"CUDA Version: {cuda_version}\n"
        f"GPU Count: {gpu_count}\n"
        f"GPU Name: {gpu_name}"
    )
    _LOG.info(logger_message)

    if settings.require_gpu and not cuda_available:
        raise RuntimeError("CUDA를 사용할 수 없습니다. docker compose에 GPU 할당이 필요합니다.")


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    for path in (
        settings.model_base_path,
        settings.upload_base_path,
        settings.snapshot_base_path,
        str(Path(settings.database_url.removeprefix("sqlite:///")).parent)
        if settings.database_url.startswith("sqlite:///")
        else "",
    ):
        if path:
            ensure_directory(path)
    log_gpu_status()
    medgemma_loader = get_medgemma_loader()
    medgemma_loader.load()
    medgemma_loader.warmup()
    get_fire_smoke_detector().try_load()
    yield
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="SilverBridge AI 서버 (모델 관리, 카메라 분석, 의료 챗 통합)",
    docs_url=settings.docs_path,
    openapi_url=settings.openapi_path,
    redoc_url=settings.redoc_path,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "success" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(str(exc.detail), "HTTP_ERROR", None),
    )


app.include_router(health_router)
app.include_router(model_router, dependencies=[Depends(require_api_key)])
app.include_router(camera_router, dependencies=[Depends(require_api_key)])
app.include_router(analysis_router, dependencies=[Depends(require_api_key)])
app.include_router(chat_router, dependencies=[Depends(require_api_key)])
app.include_router(game_router)
app.include_router(live_stream_router, dependencies=[Depends(require_api_key)])
app.include_router(live_ws_router)
app.include_router(reservation_credential_router, dependencies=[Depends(require_api_key)])
