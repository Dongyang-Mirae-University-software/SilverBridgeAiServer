from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from app.core.config import Settings


class BackendUnavailable(Exception):
    """백엔드 질문 조회 실패 (연결 오류, 시간 초과, 5xx, 주소 미설정)."""


def fetch_question(settings: Settings, ward_id: str, slot: str) -> tuple[str, str | None] | None:
    """GET /companion/question?wardId=&slot= 을 백엔드에 호출한다 (결정: AI가 질문을 조회).

    반환: (질문 문장, 질문 ID). 질문 ID는 백엔드가 주지 않으면 None.
    백엔드가 404(QUESTION_NOT_SET)면 None. 연결 실패는 BackendUnavailable.
    질문 ID는 백엔드가 질문마다 발급한 값을 그대로 되돌려 준다 (노션 요구: 응답률 계산용).
    """
    base = settings.companion_backend_base_url.rstrip("/")
    if not base:
        raise BackendUnavailable("백엔드 주소가 설정되지 않았습니다")
    query = urllib.parse.urlencode({"wardId": ward_id, "slot": slot})
    req = urllib.request.Request(f"{base}/companion/question?{query}", method="GET")
    if settings.companion_backend_api_key:
        req.add_header("X-API-Key", settings.companion_backend_api_key)
    try:
        with urllib.request.urlopen(req, timeout=settings.companion_backend_timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise BackendUnavailable(f"백엔드 응답 {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        raise BackendUnavailable("백엔드에 연결하지 못했습니다") from exc
    data = body.get("data") or {}
    question = data.get("question")
    if not (isinstance(question, str) and question.strip()):
        return None
    question_id = data.get("questionId")
    return question, (str(question_id) if question_id else None)
