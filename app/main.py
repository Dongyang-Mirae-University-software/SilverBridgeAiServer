from __future__ import annotations

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
from app.routers.health_router import router as health_router
from app.routers.model_router import router as model_router
from app.utils.file_utils import ensure_directory
from app.utils.logger import setup_logging

settings = get_settings()
setup_logging(settings.log_level)


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
    yield


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
