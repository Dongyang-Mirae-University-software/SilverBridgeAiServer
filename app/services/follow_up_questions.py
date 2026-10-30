"""답변 뒤 '이어서 물어볼 질문'(응답 suggestedQuestions, 2026-10-30).

FE 빠른 질문 칸이 대화 흐름대로 바뀌도록 답마다 사용자가 다음에 누를 만한 질문을 최대 3개 붙인다.
- 응급·자해·위기 후속·되묻기·생성 실패·예약 흐름(버튼 선택지) 답에는 붙이지 않는다 - 위기 안내를 흐리지 않게.
- 먼저 짧은 Claude 호출(FOLLOW_UP_MODEL)로 만들고, 실패하거나 FOLLOW_UP_TIMEOUT_SEC 안에 끝나지 않으면 규칙 문장으로 대신한다.
  라우터·MedGemma 프롬프트는 건드리지 않는다(안전 판정과 무관한 보조 값).
- 모델 입력은 이번 사용자 문장과 답 본문뿐이다(이름·전화·생년월일 등 프로필을 넣지 않는다, #88과 같은 기준).
- 로그에는 출처·개수·시간·실패 원인만 남긴다(질문 문장·상담 내용 금지).
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

from app.core.config import Settings
from app.services.claude_structured_client import ClaudeStructuredClient, failure_cause

_LOG = logging.getLogger(__name__)

MAX_QUESTIONS = 3
MAX_CHARS = 18
_MIN_CHARS = 2
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
# 전화번호·날짜 같은 숫자 덩어리는 개인정보일 수 있어 버린다.
_DIGIT_RUN = re.compile(r"\d{3,}")

_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["questions"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = """
너는 시니어 건강 상담 챗봇 화면의 '이어서 물어볼 질문' 버튼 문구를 만든다.
사용자의 질문과 챗봇의 답을 보고, 사용자가 다음에 누를 만한 질문 3개를 만든다.
규칙:
- 사용자가 챗봇에게 하는 말투의 짧은 질문 한 문장(18자 이내, 존댓말).
- 답에서 자연스럽게 이어지는 것: 병원에 가야 하는지, 집에서 할 일, 조심할 음식·행동, 어느 진료과인지, 예약 등.
- 사람 이름·전화번호·날짜·약 용량 숫자를 넣지 않는다. 진단을 단정하거나 약 복용량을 묻게 하지 않는다.
- 응급·자해 관련 문구는 만들지 않는다.
- 보호자 대리 상담(가족 대신 묻는 경우)이면 가족에 대해 묻는 말투로 쓴다(예: "병원에 모시고 가야 할까요?").
JSON {"questions": [...]}만 출력한다.
"""

# 규칙 문장(모델 실패·시간 초과 때). 상담 종류와 대리 상담 여부로만 고른다.
_RULE_SYMPTOM_SELF = ("병원에 가야 할까요?", "집에서 뭘 하면 좋아요?", "어느 진료과로 가요?")
_RULE_SYMPTOM_PROXY = ("병원에 모시고 가야 할까요?", "집에서 어떻게 돌봐요?", "어느 진료과로 가요?")
_RULE_GENERAL_SELF = ("몸이 좀 불편해요", "오늘 약 먹어야 해요?", "다음 병원 예약 알려줘")
_RULE_GENERAL_PROXY = ("증상을 상담하고 싶어요", "약을 같이 먹어도 되나요?", "다음 병원 예약 알려줘")

_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="follow-up")
_busy = threading.BoundedSemaphore(2)
_client_lock = threading.Lock()
_client: ClaudeStructuredClient | None = None


def should_suggest(result: dict, route_info: dict | None, message: str, *, emergency_text: bool, self_harm_text: bool) -> bool:
    """이 답에 이어서 물어볼 질문을 붙여도 되는가."""
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    route_info = route_info or {}
    if result.get("riskLevel") == "high" or result.get("intent") in ("emergency_guidance", "hospital_reservation"):
        return False
    if any(meta.get(flag) for flag in ("selfHarm", "crisisFollowup", "emergencyFollowup", "followupAsked", "generationFailed")):
        return False
    if route_info.get("emergency") or route_info.get("route") == "reservation":
        return False
    # 버튼 선택지(예약 흐름)·도구 결과가 떠 있는 동안은 그 선택이 먼저다.
    if result.get("ui") or result.get("type") == "tool_result":
        return False
    if emergency_text or self_harm_text:
        return False
    return isinstance(result.get("reply"), str) and bool(result["reply"].strip()) and bool(message)


def sanitize(items: Any, *, exclude: str = "") -> list[str]:
    """문자열만, 제어문자 제거, 길이 2~18자, 숫자 덩어리 없음, 중복·사용자 문장과 같은 것 제외, 최대 3개."""
    if not isinstance(items, list):
        return []
    out: list[str] = []
    seen = {exclude.strip()} if exclude.strip() else set()
    for item in items:
        if not isinstance(item, str):
            continue
        text = " ".join(_CONTROL.sub(" ", item).split())
        if not (_MIN_CHARS <= len(text) <= MAX_CHARS) or _DIGIT_RUN.search(text) or text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) == MAX_QUESTIONS:
            break
    return out


def rule_questions(route: str | None, *, proxy: bool) -> list[str]:
    if route == "symptom":
        return list(_RULE_SYMPTOM_PROXY if proxy else _RULE_SYMPTOM_SELF)
    return list(_RULE_GENERAL_PROXY if proxy else _RULE_GENERAL_SELF)


def _get_client(settings: Settings) -> ClaudeStructuredClient:
    global _client
    with _client_lock:
        if _client is None:
            # 보조 값이라 출력 상한을 줄이고, 호출 시간은 아래 대기 시간으로 끊는다(재시도해도 사용자를 붙들지 않는다).
            narrow = settings.model_copy(update={"claude_max_tokens": 1024})
            _client = ClaudeStructuredClient(narrow, settings.follow_up_model)
        return _client


def _model_questions(settings: Settings, message: str, reply: str, proxy: bool) -> list[str]:
    who = "보호자가 가족(피보호자) 대신 묻는 대리 상담" if proxy else "본인 상담"
    user_prompt = f"[상담 종류] {who}\n[사용자 질문]\n{message[:500]}\n[챗봇 답]\n{reply[:1500]}"
    raw = _get_client(settings).generate_structured_text(
        _SYSTEM_PROMPT, user_prompt, _SCHEMA, stage="follow_up"
    )
    data = json.loads(raw)
    return sanitize(data.get("questions") if isinstance(data, dict) else None, exclude=message)


def suggested_questions(
    settings: Settings, result: dict, route_info: dict | None, message: str, *, proxy: bool,
    emergency_text: bool = False, self_harm_text: bool = False,
) -> list[str]:
    """응답 suggestedQuestions 값. 붙이지 않는 답이면 빈 목록."""
    # 설정 칸이 없는 객체(테스트의 가짜 설정)는 끈 것으로 본다 - 실제 Settings 에는 항상 있다.
    if not getattr(settings, "follow_up_enabled", False):
        return []
    if not should_suggest(result, route_info, message, emergency_text=emergency_text, self_harm_text=self_harm_text):
        return []
    route = (route_info or {}).get("route")
    started = time.perf_counter()
    cause = None
    if _busy.acquire(blocking=False):
        released = False
        try:
            ctx = contextvars.copy_context()  # 챗 전체 마감(contextvar)을 호출 스레드에도 넘긴다
            future = _EXECUTOR.submit(ctx.run, _model_questions, settings, message, result["reply"], proxy)
            future.add_done_callback(lambda _f: _busy.release())
            released = True
            questions = future.result(timeout=settings.follow_up_timeout_sec)
            if questions:
                _LOG.info("[CHAT-SUGGEST] source=model count=%s ms=%s", len(questions), int((time.perf_counter() - started) * 1000))
                return questions
            cause = "empty"
        except FutureTimeout:
            cause = "timeout"  # 호출은 뒤에서 끝나도록 두고 결과는 버린다
        except Exception as exc:  # 호출 실패·파싱 실패 - 고정 코드나 클래스명만(원문을 남기지 않는다)
            cause = failure_cause(exc)
        finally:
            if not released:
                _busy.release()
    else:
        cause = "busy"
    questions = rule_questions(route, proxy=proxy)
    _LOG.info("[CHAT-SUGGEST] source=rule count=%s cause=%s ms=%s", len(questions), cause, int((time.perf_counter() - started) * 1000))
    return questions
