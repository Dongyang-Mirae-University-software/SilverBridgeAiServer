from __future__ import annotations

from typing import Any


def success_response(message: str, data: Any = None) -> dict[str, Any]:
    return {
        "success": True,
        "message": message,
        "data": {} if data is None else data,
    }


def error_response(message: str, error_code: str, data: Any = None) -> dict[str, Any]:
    return {
        "success": False,
        "message": message,
        "errorCode": error_code,
        "data": data,
    }
