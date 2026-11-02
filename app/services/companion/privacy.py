from __future__ import annotations

import hashlib
import hmac
import re

from app.core.config import get_settings


class PseudonymSaltMissing(Exception):
    """비식별 ID 를 만들 salt 가 없음. 이 상태에서는 개선 DB 적재를 하지 않는다 (설정 주석 참고)."""


def pseudonym(value: str) -> str:
    """S8-3: 어르신 ID 같은 식별자를 HMAC-SHA256 으로 바꾼다. salt 가 없으면 만들지 않는다.

    같은 입력은 같은 결과가 나와야 기준선 같은 분석을 할 수 있다. salt 는 서버 설정(.env)에만 둔다.
    """
    salt = get_settings().companion_pseudonym_salt
    if not salt:
        raise PseudonymSaltMissing("COMPANION_PSEUDONYM_SALT 가 비어 있습니다")
    digest = hmac.new(salt.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[:24]


# S8-4: 적재 전에 가려야 할 식별 정보 (전화번호, 주민등록번호 형식, 계좌처럼 긴 숫자)
_PATTERNS = (
    (re.compile(r"01[016789][-\s]?\d{3,4}[-\s]?\d{4}"), "[전화번호]"),
    (re.compile(r"0\d{1,2}[-\s]\d{3,4}[-\s]\d{4}"), "[전화번호]"),
    (re.compile(r"\d{6}-[1-4]\d{6}"), "[주민번호]"),
    (re.compile(r"\d{10,}"), "[번호]"),
)


def strip_pii(text: str) -> str:
    """식별 정보로 보이는 숫자 묶음을 가린다. 완벽하지 않으므로 적재 전에 한 번 더 사람이 본다 (S8-4 확인 방식)."""
    out = text
    for pattern, mask in _PATTERNS:
        out = pattern.sub(mask, out)
    return out
