from __future__ import annotations

import hmac

from fastapi import Depends, Header

from app.core.companion_errors import CompanionError
from app.core.config import Settings, get_settings


def require_companion_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    """말벗 전용 키 검사 (D4). 명세 오류 코드 AUTH_INVALID_KEY (401). 설정 키가 비면 모두 거부한다."""
    expected = settings.companion_api_key
    if not expected or not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise CompanionError(401, "AUTH_INVALID_KEY", "키가 없거나 틀렸습니다.")
