from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections import Counter

from app.core.config import Settings
from app.services.companion.dialogue_store import Dialogue
from app.services.companion.reply_service import REPLY_PROMPT_VERSION
from app.services.companion.signals import PROMPT_VERSION as SIGNALS_PROMPT_VERSION

logger = logging.getLogger(__name__)

# 대화 1회 결과를 받는 백엔드 경로 (AI → 백엔드). 턴마다 보내지 않고 대화가 끝날 때 한 번 보낸다 (10/26 합의).
RESULT_PATH = "/companion/dialogue-results"

# 화면에 쓰는 기분 값 (노션 요구: 기쁨·평온·걱정·우울). 턴 라벨을 이 값으로 묶는다.
# 제안 매핑: 슬픔·화남 → 걱정. 팀이 정하면 이 표만 고치면 된다.
SCREEN_MOOD = {"기쁨": "기쁨", "평온": "평온", "우울": "우울", "슬픔": "걱정", "화남": "걱정"}


def build_dialogue_result(d: Dialogue) -> dict:
    """대화 1회 결과. 원문(대답 글)은 넣지 않는다 (10/26 합의: 원문은 AI 쪽에만 둔다).

    mood: 답한 턴 중 가장 많이 나온 기분. 판단 보류뿐이거나 답이 없으면 None (화면에서는 '확인할 수 없어요').
    alertLevel: 위험 플래그가 있었으면 URGENT, 없으면 INFO. 부정 비율 기반 NOTICE는 기준선이 필요해서
    백엔드가 계산한다 (AI는 대화 기록을 쌓지 않는다).
    """
    answered = [SCREEN_MOOD[t.label] for t in d.turns if t.label in SCREEN_MOOD]
    mood = Counter(answered).most_common(1)[0][0] if answered else None
    no_response = sum(1 for t in d.turns if t.label is None)
    return {
        # 어르신이 답한 턴 수 (무응답 제외, 판단 보류는 답한 것으로 센다). 응답률 = answeredCount ÷ 질문 수 (노션 요청 1-3)
        "answeredCount": d.turnNo - no_response,
        "dialogueId": d.dialogueId,
        "wardId": d.wardId,
        "questionId": d.questionId,  # 백엔드가 준 질문 ID를 그대로 돌려준다 (응답률 계산용)
        "slot": d.slot,
        "endedAt": d.endedAt,
        "endReason": d.endReason,
        "turnCount": d.turnNo,
        "noResponseCount": no_response,
        "mood": mood,
        "summary": None,  # 한 줄 요약은 LLM이 준비된 뒤에 채운다
        "riskFlag": d.riskSeen,
        "alertLevel": "URGENT" if d.riskSeen else "INFO",
        "promptVersion": {"reply": REPLY_PROMPT_VERSION, "signals": SIGNALS_PROMPT_VERSION},
    }


def send_dialogue_result(settings: Settings, result: dict) -> bool:
    """대화 결과를 백엔드로 보낸다. 응답을 기다리지 않는 백그라운드 작업으로 쓴다.

    주소가 비어 있으면 아무것도 하지 않는다. 실패해도 대화는 그대로 진행하고, 로그에는 내용을 남기지 않는다.
    """
    base = settings.companion_backend_base_url.rstrip("/")
    if not base:
        return False
    req = urllib.request.Request(
        f"{base}{RESULT_PATH}",
        data=json.dumps(result, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    if settings.companion_backend_api_key:
        req.add_header("X-API-Key", settings.companion_backend_api_key)
    try:
        with urllib.request.urlopen(req, timeout=settings.companion_record_timeout_sec) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        logger.warning("대화 결과 전송 실패: %s", type(exc).__name__)
        return False
