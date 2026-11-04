"""말벗 개발용 실행 앱. main 서버(app.main)는 MedGemma 게이트 때문에 로컬에서 뜨지 않아 따로 띄운다 (CH-031).

실행: .venv/Scripts/python.exe -m uvicorn scripts.companion.dev_app:app --port 9100
main.py는 건드리지 않는다. main 등록(S1-3)은 MedGemma 접근이 해결된 뒤로 미룬다.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.companion_errors import register_companion_error_handler
from app.core.response import success_response
from app.routers.companion_router import router as companion_router
from app.services.companion.face_service import FaceUnavailable, face_service
from app.services.companion.stt_engine import get_stt_engine
from app.services.companion.stt_service import SttUnavailable
from app.services.companion.text_emotion_service import TextEmotionUnavailable, text_emotion_service


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # 모델을 서버 시작 때 미리 올린다. 첫 분석이 시간 제한(S9-1)에 걸리지 않게 하려는 것이다.
    # 모델 파일이 없으면 그 기능만 빠지고, 대화는 계속된다 (대화 우선 원칙).
    for service, unavailable in ((text_emotion_service, TextEmotionUnavailable),
                                 (face_service, FaceUnavailable),
                                 (get_stt_engine(), SttUnavailable)):
        try:
            service.warm_up()
        except unavailable:
            pass
    yield


app = FastAPI(title="SilverBridge Companion (dev)", docs_url="/api/docs", lifespan=lifespan)
register_companion_error_handler(app)
app.include_router(companion_router)


@app.get("/health", tags=["Health"])
def health() -> dict:
    return success_response("말벗 개발 서버 정상", {"mode": "companion-dev"})
