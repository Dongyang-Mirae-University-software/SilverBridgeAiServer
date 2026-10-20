from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status

from app.core.config import Settings, get_settings


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    if not x_api_key or x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "success": False,
                "message": "유효하지 않은 API Key 입니다.",
                "errorCode": "AUTH_INVALID_KEY",
                "data": None,
            },
        )


def require_game_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    """게임 라우터 중 서버 간(보호자 FE 프록시) 호출에만 붙인다. GAME_REQUIRE_API_KEY 가 꺼져 있으면(기본) 통과.

    피보호자 게임 iframe 과 그 안의 JS 가 브라우저에서 키 없이 직접 부르는 /embed·/state·/answer·/reset 에는
    붙이지 않는다 - 거기에 키를 요구하면 게임 화면이 401 로 멈춘다.
    """
    if settings.game_require_api_key:
        require_api_key(x_api_key, settings)
