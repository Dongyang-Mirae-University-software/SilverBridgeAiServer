from __future__ import annotations

import asyncio
import gc
import threading
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.companion_errors import register_companion_error_handler
from app.core.response import error_response
from app.core.security import require_api_key
from app.database.base import Base
from app.database.schema_upgrades import encrypt_plain_reservation_keys, ensure_indexes, upgrade_member_id_columns
from app.database.session import engine
from app.routers.analysis_router import router as analysis_router
from app.routers.camera_router import router as camera_router
from app.routers.chat_router import router as chat_router
from app.routers.companion_router import router as companion_router
from app.routers.game_router import router as game_router
from app.routers.health_router import router as health_router
from app.routers.live_stream_router import router as live_stream_router
from app.routers.live_ws_router import router as live_ws_router
from app.routers.medgemma_switch_router import router as medgemma_switch_router
from app.routers.medication_ocr_router import router as medication_ocr_router
from app.routers.model_router import router as model_router
from app.routers.reservation_credential_router import router as reservation_credential_router
import app.models.game  # noqa: F401
import app.models.reservation_credential  # noqa: F401
import app.models.reservation_token_claim  # noqa: F401
from app.services.fall_detection_service import get_fall_detector
from app.services.fire_smoke_detection_service import get_fire_smoke_detector
from app.services.knife_detection_service import get_knife_detector
from app.services.live_ws_manager import live_ws_manager
from app.services.medical_llm_service import get_medgemma_loader
from app.services.drug_name_list import get_drug_name_list
from app.services.medication_ocr_service import get_medication_ocr_service
from app.utils.file_utils import ensure_directory
from app.utils.logger import setup_logging

settings = get_settings()
setup_logging(settings.log_level)
_LOG = logging.getLogger(__name__)
if not settings.api_key:
    _LOG.warning("API_KEY 미설정 - 인증이 필요한 API·WS 는 모두 거절된다(.env 에 API_KEY 를 설정할 것)")

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
    # 동기 엔드포인트(세션 생성·종료)의 WS 방송이 이 루프로 넘어간다(QA AI-1).
    live_ws_manager.bind_loop(asyncio.get_running_loop())
    Base.metadata.create_all(bind=engine)
    upgrade_member_id_columns(engine)
    ensure_indexes(engine)
    try:
        sealed_rows = encrypt_plain_reservation_keys(engine, settings.reservation_account_secret)
        if sealed_rows:
            logging.getLogger(__name__).info("[예약계정] 평문 예약 키 %d건을 암호화했습니다", sealed_rows)
    except Exception as exc:  # noqa: BLE001 - 예약 키 정리 실패로 감지·상담 기동을 막지 않는다(다음 기동·다음 사용 때 다시 한다)
        logging.getLogger(__name__).error("[예약계정] 평문 예약 키 암호화 실패 type=%s", type(exc).__name__)
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
    if settings.medgemma_enabled:
        medgemma_loader = get_medgemma_loader()
        medgemma_loader.load()
        medgemma_loader.warmup()
    else:
        _LOG.info("MEDGEMMA_ENABLED=false - MedGemma 를 올리지 않는다")
    get_fire_smoke_detector().try_load()
    get_knife_detector().try_load()
    get_fall_detector().try_load()
    get_medication_ocr_service()  # 모델 설정 확인(haiku 면 기동 시 경고 후 sonnet 대체)
    get_drug_name_list()  # 약 이름 허가 목록을 한 번 메모리에 올린다(없으면 WARN 후 대조 생략)
    _start_companion_warmup()
    yield
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _start_companion_warmup() -> None:
    """말벗이 켜져 있으면(키 있음) EXAONE·어댑터를 백그라운드에서 미리 올린다 (11/10 점검 M-1).

    올리지 않으면 재시작 뒤 첫 턴이 요청 안에서 모델을 올리고(약 22초), 그동안 위험 판정이 낱말 규칙으로만 된다.
    서버 기동은 기다리지 않는다. 실패해도 첫 턴에서 다시 시도한다 (오류 내용은 클래스명만 남긴다).
    """
    if not (settings.companion_api_key and settings.companion_llm_warmup):
        return

    def run() -> None:
        from app.services.companion.llm_client import llm_client

        try:
            llm_client.warm_up()
            _LOG.info("[COMPANION-LLM] warm-up done")
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("[COMPANION-LLM] warm-up failed (%s)", type(exc).__name__)

    threading.Thread(target=run, name="companion-llm-warmup", daemon=True).start()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="SilverBridge AI 서버 (모델 관리, 카메라 분석, 의료 챗 통합)",
    docs_url=settings.docs_path if settings.docs_enabled else None,
    openapi_url=settings.openapi_path if settings.docs_enabled else None,
    redoc_url=settings.redoc_path if settings.docs_enabled else None,
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
# 옛 카메라 등록·주소 접속 분석 경로(2026-11-10 전체 점검 M-8): 지금 감지는 프레임 업로드(stream-sessions)로 하고 이 경로를 부르는
# 곳이 없다. 요청이 준 주소로 서버가 접속하는 경로라 기본은 등록하지 않는다(LEGACY_CAMERA_API_ENABLED=true 일 때만).
if settings.legacy_camera_api_enabled:
    app.include_router(camera_router, dependencies=[Depends(require_api_key)])
    app.include_router(analysis_router, dependencies=[Depends(require_api_key)])
app.include_router(chat_router, dependencies=[Depends(require_api_key)])
app.include_router(game_router)  # 키 요구는 game_router 가 경로별로 붙인다
app.include_router(live_stream_router, dependencies=[Depends(require_api_key)])
app.include_router(live_ws_router)
app.include_router(reservation_credential_router, dependencies=[Depends(require_api_key)])
app.include_router(medication_ocr_router, dependencies=[Depends(require_api_key)])
app.include_router(medgemma_switch_router, dependencies=[Depends(require_api_key)])  # MedGemma 켜기·끄기 (재시작 없음)
app.include_router(companion_router)  # 말벗: 자체 키(X-API-Key, COMPANION_API_KEY)를 라우터 안에서 붙인다
register_companion_error_handler(app)
