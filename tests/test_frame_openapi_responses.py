"""최종 QA CAM-N59: 프레임 수신 경로의 openapi 에 실제로 나는 오류 코드가 적혀 있어야 한다."""
from __future__ import annotations

from fastapi import FastAPI

from app.routers.live_stream_router import router

FRAME_PATH = "/api/v1/stream-sessions/{session_id}/frame"


def _frame_responses() -> dict:
    app = FastAPI()
    app.include_router(router)
    return app.openapi()["paths"][FRAME_PATH]["post"]["responses"]


def test_frame_openapi_lists_real_error_codes() -> None:
    responses = _frame_responses()
    assert {"200", "400", "401", "404", "413", "422"} <= set(responses)


def test_frame_openapi_error_examples_use_api_error_shape() -> None:
    responses = _frame_responses()
    codes = {
        "400": "STREAM_FRAME_INVALID_TYPE",
        "401": "AUTH_INVALID_KEY",
        "404": "STREAM_SESSION_NOT_FOUND",
        "413": "STREAM_FRAME_TOO_LARGE",
    }
    for status_code, error_code in codes.items():
        example = responses[status_code]["content"]["application/json"]["example"]
        # FastAPI 는 openapi 를 만들 때 None 값을 빼므로 data 는 없거나 None 이다.
        assert example["success"] is False and example.get("data") is None
        assert example["errorCode"] == error_code


def test_frame_request_body_still_documented() -> None:
    app = FastAPI()
    app.include_router(router)
    op = app.openapi()["paths"][FRAME_PATH]["post"]
    assert op["requestBody"]["content"]["multipart/form-data"]["schema"]["required"] == ["frame"]
