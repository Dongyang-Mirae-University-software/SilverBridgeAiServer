from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class CompanionError(Exception):
    """말벗 API 오류. 응답 본문에 code 와 errorCode 를 같은 값으로 함께 넣는다 (S2 객관식 답: 둘 다)."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def companion_error_body(code: str, message: str) -> dict:
    return {"success": False, "message": message, "code": code, "errorCode": code, "data": None}


def register_companion_error_handler(app: FastAPI) -> None:
    @app.exception_handler(CompanionError)
    async def _handle(_: Request, exc: CompanionError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=companion_error_body(exc.code, exc.message))
