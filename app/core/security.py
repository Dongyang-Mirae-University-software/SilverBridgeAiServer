from __future__ import annotations

import hmac

from fastapi import Depends, Header, HTTPException, status

from app.core.config import Settings, get_settings


def api_key_matches(provided: str | None, expected: str | None) -> bool:
    """X-API-Key 상수 시간 비교. 서버 키가 비어 있으면 모두 거절한다. 비ASCII 값도 예외 없이 bytes 로 비교한다."""
    if not expected or not provided:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    if not api_key_matches(x_api_key, settings.api_key):
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
    """게임 라우터 중 서버 간(보호자 FE 프록시) 호출에만 붙인다. GAME_REQUIRE_API_KEY 가 꺼져 있을 때만(명시적으로 끈 경우) 통과.

    피보호자 게임 iframe 과 그 안의 JS 가 브라우저에서 키 없이 직접 부르는 /embed·/state·/answer·/reset 에는
    붙이지 않는다 - 거기에 키를 요구하면 게임 화면이 401 로 멈춘다.
    """
    if settings.game_require_api_key:
        require_api_key(x_api_key, settings)


def admin_api_key_warning(settings: Settings) -> str | None:
    """ADMIN_API_KEY 가 공용 API_KEY 와 같으면 경고 문구, 아니면 None. 값은 문구에 넣지 않는다. 기동 때 main.py 가 부른다."""
    admin_key = settings.admin_api_key.strip()
    if admin_key and admin_key == settings.api_key.strip():
        return "ADMIN_API_KEY 가 API_KEY 와 같습니다 - 관리 키를 나눈 의미가 없습니다. 다른 값으로 바꾸세요."
    return None


def require_admin_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    """관리 경로(/api/admin/*) 키. ADMIN_API_KEY 가 설정돼 있으면 그 키만 받는다(공용 키는 거절). 비어 있으면 공용 키로 받는다.

    공용 키는 BE·FE 서버가 함께 쓰므로, 모델을 내리는 조작까지 그 키로 되지 않게 나눌 수 있게 했다(2026-11-10 전체 점검 M-4).
    """
    admin_key = settings.admin_api_key.strip()
    if admin_key:
        if not api_key_matches(x_api_key, admin_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={"success": False, "message": "유효하지 않은 API Key 입니다.", "errorCode": "AUTH_INVALID_KEY", "data": None},
            )
        return
    require_api_key(x_api_key, settings)
