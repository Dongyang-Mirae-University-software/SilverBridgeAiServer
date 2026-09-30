from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from app.core.config import get_settings
from app.services.gpt_structured_client import GptStructuredClient
from app.utils.conversation import format_recent_history, normalize_history
from app.utils.json_extract import extract_json_object

_LOG = logging.getLogger(__name__)

ROUTES = ("reservation", "symptom", "general")

SYSTEM_PROMPT = """너는 노인 돌봄 챗봇의 1차 분류기다. 사용자 문장과 최근 대화를 읽고 JSON 하나만 출력한다.

route 판단:
- "reservation": 병원 예약·병원 검색·진료과 찾기·예약 조회/변경/취소. 직전 대화가 예약 흐름이면 짧은 후속 문장("내과 찾아 줘", "내일로")도 reservation.
- "symptom": 증상·통증·복약·건강 상태 질문. 의료 안내가 필요한 경우.
- "general": 인사, 감사, 작별, 잡담, 그 외. 직전이 예약 흐름이어도 "고마워", "알겠어", "잘 지내"처럼 예약 정보가 전혀 없는 문장은 general이다.

normalized_message (route=reservation일 때): 예약 로직에 넘길 한 문장. 형식 "병원 예약: 지역 / 진료과 / 날짜 / 시간 / 병원명 / 환자명 / 전화". 모르는 항목은 생략.
- 지역은 사용자가 직접 말한 경우에만 채우고(프로필 주소를 추측해 넣지 않는다), 반드시 "시도 구군" 표준형으로 바꾼다. 예: 강남역→서울 강남구, 판교→경기 성남시, 구로디지털단지→서울 구로구, 송도→인천 연수구. 알 수 없으면 생략.
- 진료과는 내과, 안과, 이비인후과, 정형외과, 피부과, 소아청소년과, 산부인과, 신경과, 정신건강의학과, 가정의학과, 응급의학과, 외과 중 하나. 증상만 있으면 가장 적절한 과로 추론.
- [현재 예약 상태]가 있으면 그 값을 기본으로 두고 현재 문장이 말한 항목만 갱신한다. 상태가 없으면 최근 대화에서 가장 나중에 나온 값을 이어받는다. 현재 문장에서 새로 말한 항목은 항상 우선한다(예: 이비인후과 예약 중에 "내과 찾아 줘" → 진료과는 내과, 그 다음 "내일 2시로" → 진료과는 여전히 내과).
- 날짜·시간은 절대 계산하지 말고 사용자가 말한 표현 그대로 쓴다("내일", "다음 주 월요일", "오후 2시"). YYYY-MM-DD로 바꾸지 마라. 오늘 날짜를 모르기 때문이다.

symptom_summary (route=symptom일 때): 의료 LLM에 줄 규격화된 증상 문장. "주증상 / 시작 시점·유발 요인 / 동반 증상 / 사용자가 알고 싶은 것" 순서로 2~3문장, 사용자 표현을 의학적으로 정리하되 새 정보를 지어내지 않는다.

emergency: 호흡곤란·의식 저하·심한 가슴 통증·대량 출혈·경련 등 즉시 119가 필요한 신호가 있으면 true.

출력 형식(키 고정, JSON 외 텍스트 금지):
{"route":"reservation|symptom|general","normalized_message":"","symptom_summary":"","emergency":false,"reason":""}
"""


class ChatRouterService:
    def __init__(self, loader: GptStructuredClient, keep_turns: int) -> None:
        self._loader = loader
        self._keep_turns = keep_turns

    def route(
        self,
        message: str,
        history: list[dict[str, str]],
        user_context: dict[str, Any] | None,
        prev_reservation: str | None = None,
    ) -> dict[str, Any]:
        recent, summary = normalize_history(history, keep_turns=self._keep_turns)
        history_text = format_recent_history(recent)
        if summary:
            history_text = f"(이전 요약) {summary}\n{history_text}"
        user_prompt = (
            f"[사용자 문장]\n{message}\n\n[현재 예약 상태(직전 턴의 규격화 문장)]\n{prev_reservation or '(없음)'}\n\n"
            f"[최근 대화]\n{history_text or '(없음)'}"
        )
        raw = self._loader.generate_structured_text(SYSTEM_PROMPT, user_prompt)
        parsed = extract_json_object(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError("router json parse failed")
        route = str(parsed.get("route") or "").strip().lower()
        if route not in ROUTES:
            raise RuntimeError(f"router unknown route: {route}")
        normalized = str(parsed.get("normalized_message") or "").strip()
        if route == "reservation" and "예약" not in normalized:
            normalized = f"병원 예약: {normalized or message}"
        return {
            "route": route,
            "normalized_message": normalized or message,
            "symptom_summary": str(parsed.get("symptom_summary") or "").strip() or message,
            "emergency": bool(parsed.get("emergency")),
            "reason": str(parsed.get("reason") or "").strip(),
            "engine": "gpt",
            "modelName": self._loader.state.model_name,
        }


@lru_cache
def get_chat_router_service() -> ChatRouterService:
    settings = get_settings()
    return ChatRouterService(GptStructuredClient(settings), settings.chat_history_keep_turns)
