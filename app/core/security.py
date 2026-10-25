from __future__ import annotations

import hmac

from fastapi import Depends, Header, HTTPException, status

from app.core.config import Settings, get_settings


def _invalid_key() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail={
            "success": False,
            "message": "유효하지 않은 API Key 입니다.",
            "errorCode": "AUTH_INVALID_KEY",
            "data": None,
        },
    )


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    if not x_api_key or x_api_key != settings.api_key:
        raise _invalid_key()


def require_stream_reader_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    """최신 프레임·상태 조회 전용. 메인 API_KEY 또는 말벗 전용 읽기 키(COMPANION_STREAM_API_KEY)만 통과한다.

    말벗 키는 이 의존성이 붙은 GET 두 경로 외에는 쓸 수 없다(송출·분석·MJPEG 등은 require_api_key 만 허용).
    쿼리스트링 키는 받지 않는다(헤더 전용).
    """
    if not x_api_key:
        raise _invalid_key()
    if hmac.compare_digest(x_api_key.encode(), settings.api_key.encode()) and settings.api_key:
        return
    if settings.companion_stream_api_key and hmac.compare_digest(
        x_api_key.encode(), settings.companion_stream_api_key.encode()
    ):
        return
    raise _invalid_key()


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
