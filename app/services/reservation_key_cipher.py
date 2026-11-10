"""예약 서비스 사용자별 API 키의 저장용 암호화(2026-11-10 전체 점검 M-5).

DB(`reservation_credentials.api_key`)에 평문으로 두던 키를 `enc:v1:` 머리말 + Fernet 토큰으로 저장한다. 암호 키는
RESERVATION_ACCOUNT_SECRET 에서 용도 문자열로 갈라 만든다(계정 비밀번호 파생과 다른 키). 비밀값이 없으면 저장하지 않는다 -
그때는 예약 기능 자체가 꺼져 있다. 비밀값이 바뀌어 풀 수 없는 행은 키가 없는 것으로 보고 다시 발급받는다.
값(키·비밀값·토큰)은 어디에도 로그로 남기지 않는다.
"""
from __future__ import annotations

import base64
import hashlib
import hmac

from cryptography.fernet import Fernet, InvalidToken

PREFIX = "enc:v1:"
_PURPOSE = b"silverbridge-ai:reservation-credential:v1"


class CipherUnavailable(ValueError):
    """비밀값이 없어 암호화할 수 없다."""


def _fernet(secret: str) -> Fernet:
    raw = (secret or "").strip()
    if not raw:
        raise CipherUnavailable("예약 연동 비밀값이 설정되지 않아 예약 키를 저장할 수 없습니다.")
    return Fernet(base64.urlsafe_b64encode(hmac.new(raw.encode("utf-8"), _PURPOSE, hashlib.sha256).digest()))


def is_sealed(stored: str | None) -> bool:
    return (stored or "").startswith(PREFIX)


def seal(api_key: str, secret: str) -> str:
    return PREFIX + _fernet(secret).encrypt(api_key.strip().encode("utf-8")).decode("ascii")


def unseal(stored: str | None, secret: str) -> str | None:
    """저장된 값을 키로 되돌린다. 옛 평문 행은 그대로 돌려준다(기동 때 한 번에 암호화한다). 풀 수 없으면 None."""
    value = (stored or "").strip()
    if not value:
        return None
    if not value.startswith(PREFIX):
        return value
    try:
        return _fernet(secret).decrypt(value[len(PREFIX):].encode("ascii")).decode("utf-8").strip() or None
    except (InvalidToken, CipherUnavailable, ValueError):
        return None
