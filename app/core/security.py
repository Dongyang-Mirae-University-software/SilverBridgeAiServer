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


def game_router_dependencies(settings: Settings) -> list:
    """게임 라우터에 붙일 의존성. GAME_REQUIRE_API_KEY 가 꺼져 있으면(기본) 지금처럼 키 없이 연다.

    피보호자 게임 iframe 과 그 안의 JS(/state·/answer·/reset)가 브라우저에서 키 없이 직접 부르므로,
    FE 가 이 호출들을 프록시 경유로 바꾸기 전에 켜면 게임 화면이 401 로 멈춘다.
    """
    return [Depends(require_api_key)] if settings.game_require_api_key else []
