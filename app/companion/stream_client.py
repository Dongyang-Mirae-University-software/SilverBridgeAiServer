"""AI 서버에서 피보호자 최신 프레임을 가져온다 (계획서 Phase 3, 3.x).

- 말벗 전용 읽기 키(COMPANION_STREAM_API_KEY)를 헤더 X-API-Key 로만 보낸다. 쿼리스트링에는 넣지 않는다.
- 프레임은 메모리에서만 다루고 파일로 저장하지 않는다. 분석이 끝나면 호출한 쪽이 버린다.
- 키 값은 로그나 예외 메시지에 넣지 않는다.

대상 경로 (AI 서버 live_stream_router.py):
    GET {base}/api/v1/live-streams/{session_id}/latest-frame  -> image/jpeg
    GET {base}/api/v1/live-streams/{session_id}/status        -> {success, message, data, code}
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

TIMEOUT_SEC = 5


class StreamError(Exception):
    """프레임·상태 조회 실패. code 는 AI 서버의 errorCode 또는 아래 로컬 코드."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _settings() -> tuple[str, str]:
    base = os.getenv("COMPANION_STREAM_BASE_URL", "").rstrip("/")
    key = os.getenv("COMPANION_STREAM_API_KEY", "")
    if not base:
        raise StreamError("CONFIG_MISSING", "COMPANION_STREAM_BASE_URL 이 .env 에 없습니다.")
    if not key:
        raise StreamError("CONFIG_MISSING", "COMPANION_STREAM_API_KEY 가 .env 에 없습니다.")
    return base, key


def _get(path: str) -> tuple[bytes, str]:
    base, key = _settings()
    req = urllib.request.Request(base + path, headers={"X-API-Key": key})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SEC) as resp:
            return resp.read(), resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        code, message = "HTTP_ERROR", f"AI 서버 응답 오류 ({e.code})"
        try:
            body = json.loads(e.read().decode("utf-8", "ignore"))
            detail = body.get("detail", body)
            if isinstance(detail, dict):
                code = detail.get("errorCode", code)
                message = detail.get("message", message)
        except (ValueError, AttributeError):
            pass
        raise StreamError(code, message) from None
    except (urllib.error.URLError, TimeoutError) as e:
        raise StreamError("UNREACHABLE", f"AI 서버에 연결하지 못했습니다: {type(e).__name__}") from None


def _session_path(session_id: str) -> str:
    return "/api/v1/live-streams/" + urllib.parse.quote(session_id, safe="")


def fetch_latest_frame(session_id: str) -> bytes:
    """최신 JPEG 프레임 바이트를 돌려준다. 저장하지 않는다."""
    body, content_type = _get(_session_path(session_id) + "/latest-frame")
    if not content_type.startswith("image/jpeg"):
        raise StreamError("BAD_CONTENT", f"예상과 다른 응답 형식입니다: {content_type}")
    return body


def fetch_status(session_id: str) -> dict:
    """세션 상태(status, lastFrameAt, fps 등)를 돌려준다."""
    body, _ = _get(_session_path(session_id) + "/status")
    payload = json.loads(body.decode("utf-8"))
    if not payload.get("success"):
        raise StreamError(payload.get("code", "STATUS_FAILED"), payload.get("message", "상태 조회 실패"))
    return payload["data"]
