"""안부 말벗만 띄우는 개발용 앱 (DGX 등 AI 서버 전체를 올리지 않는 곳에서 시험).

운영(gosky)에서는 app.main 이 같은 라우터를 붙인다. 이 앱은 감지 모델·MedGemma 를 올리지 않는다.
실행: CHECKIN_ENABLED=true DATABASE_URL=sqlite:///./data/checkin_dev.db API_KEY=... \
      uvicorn scripts.checkin_dev_app:app --host 127.0.0.1 --port 6020
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.core.response import error_response
from app.core.security import require_api_key
from app.database.base import Base
from app.database.session import engine
from app.routers.checkin_router import router as checkin_router
from app.services.checkin.config import get_checkin_settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    if get_checkin_settings().enabled:
        from app.services.checkin.models import get_models

        get_models()  # 첫 대답이 모델 적재 시간을 기다리지 않게 미리 올린다
    yield


app = FastAPI(title="SilverBridge checkin (dev)", lifespan=lifespan)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "success" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content=error_response(str(exc.detail), "HTTP_ERROR", None))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "checkinEnabled": get_checkin_settings().enabled}


app.include_router(checkin_router, dependencies=[Depends(require_api_key)])
