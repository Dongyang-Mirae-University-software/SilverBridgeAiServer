"""MedGemma 관리자 스위치: 컨테이너를 다시 만들지 않고 MedGemma 만 켜고 끈다.

- GET  /api/admin/medgemma/status   : 켜짐·올라감·양자화·오류 상태
- POST /api/admin/medgemma/enable   : 켜고 메모리에 올린다 (몇 분 걸릴 수 있다)
- POST /api/admin/medgemma/disable  : 끄고 메모리(GPU)에서 내린다
모든 경로는 X-API-Key 가 필요하다. 다른 기능(말벗·받아쓰기·탐지기)에는 영향을 주지 않는다.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool

from app.core.response import success_response
from app.services.medical_llm_service import get_medgemma_loader

router = APIRouter(prefix="/api/admin/medgemma", tags=["Admin"])


def get_loader():
    return get_medgemma_loader()


def _status_body(loader) -> dict:
    return {
        "enabled": loader.enabled,
        "loaded": loader.state.loaded,
        "modelName": loader.state.model_name,
        "quant": loader.quant,
        "error": loader.state.load_error,
    }


@router.get("/status", summary="MedGemma 켜짐·올라감 상태")
def medgemma_status(loader=Depends(get_loader)) -> dict:
    return success_response("MedGemma 상태", _status_body(loader))


def _enable(loader) -> None:
    loader.set_enabled(True)
    loader.load()
    if loader.state.loaded:
        loader.warmup()


@router.post("/enable", summary="MedGemma 켜기 (메모리에 올림)")
async def medgemma_enable(loader=Depends(get_loader)) -> dict:
    await run_in_threadpool(_enable, loader)
    return success_response("MedGemma 를 켰습니다.", _status_body(loader))


@router.post("/disable", summary="MedGemma 끄기 (메모리에서 내림)")
async def medgemma_disable(loader=Depends(get_loader)) -> dict:
    loader.set_enabled(False)
    await run_in_threadpool(loader.unload)
    return success_response("MedGemma 를 껐습니다.", _status_body(loader))
