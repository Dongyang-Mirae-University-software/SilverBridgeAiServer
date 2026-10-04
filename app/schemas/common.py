from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, StringConstraints

MEMBER_ID_MAX_LENGTH = 64


def _to_member_id(value: Any) -> Any:
    # SilverBridge 회원 ID 는 문자열(예: "e2eg01")이다. 기존 FE 가 보내던 숫자(1)도 받아 문자열로 바꾼다(QA AI-3).
    if isinstance(value, bool):
        return value  # True/False 는 회원 ID 가 아니다 → 문자열 검증에서 거절
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value.strip()
    return value


MemberId = Annotated[
    str,
    BeforeValidator(_to_member_id),
    StringConstraints(min_length=1, max_length=MEMBER_ID_MAX_LENGTH),
]


class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Any | None = None
    errorCode: str | None = None
