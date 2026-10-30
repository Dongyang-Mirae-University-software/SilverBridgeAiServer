from __future__ import annotations

import contextlib
import json
import re
import logging
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo
from uuid import uuid4
from typing import Any

from sqlalchemy.orm import Session  # pyright: ignore[reportMissingImports]

from app.models.chat_log import ChatLog
from app.core.config import get_settings
from app.schemas.chat_schema import ChatHistoryItem, ChatRequest
from app.services.reservation_api_client import get_reservation_api_client
from app.services.reservation_credential_service import get_reservation_credential_service, recently_purged
from app.services.chat_deadline import ChatDeadlineExceeded, chat_deadline, tighter_deadline
from app.services.chat_router_service import ROUTER_RESERVE_SEC, get_chat_router_service
from app.services.claude_structured_client import failure_cause
from app.services.medical_llm_service import MedicalLlmService, get_medical_llm_service
from app.services.reservation_confirm_flow import CONFIRM_FIELDS
from app.services.reservation_intake import CHANGE_DATE_OPTION, _find_department
from app.services.reservation_orchestrator import ReservationOrchestrator, get_reservation_orchestrator

_KST = ZoneInfo("Asia/Seoul")
_MEDGEMMA_GENDERS = frozenset({"MALE", "FEMALE"})


def _age_from_birth_date(birth_date: str) -> int | None:
    """YYYY-MM-DD(또는 YYYYMMDD) 생년월일 -> KST 오늘 기준 만 나이. 형식이 다르거나 범위 밖이면 None."""
    digits = re.sub(r"\D", "", birth_date or "")
    if len(digits) != 8:
        return None
    try:
        born = date(int(digits[:4]), int(digits[4:6]), int(digits[6:]))
    except ValueError:
        return None
    today = datetime.now(_KST).date()
    age = today.year - born.year - ((today.month, today.day) < (born.month, born.day))
    return age if 0 <= age <= 130 else None


_LOG = logging.getLogger(__name__)


def _elapsed_ms(since: float) -> int:
    return int((time.perf_counter() - since) * 1000)


def _origin(exc: BaseException) -> str:
    """예외가 난 마지막 프레임의 파일:줄. 예외 문구(값이 섞일 수 있음) 없이 원인 위치만 남긴다."""
    tb = exc.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    if tb is None:
        return "?"
    return f"{tb.tb_frame.f_code.co_filename.rsplit('/', 1)[-1]}:{tb.tb_lineno}"


# 오케스트레이터에 넘긴 입력값이 응답 meta 로 되돌아온다. 챗 기록(context_json)에는 남기지 않는다:
# 예약 API 키(자격 증명), 직전 턴 상태 사본(storedFlow - 같은 기록이 이미 직전 행에 있음), 사용자 원문(message 컬럼에 있음),
# 버튼으로 입력했던 환자 이름·전화(storedContact - 원래 기록의 message 에 있음).
_UNSTORED_META_KEYS = ("reservationApiKey", "storedFlow", "rawMessage", "storedContact")
# 위기 고정 응답 직후 첫 일반 턴(예약 재개·증상·일반 대화) 답 맨 앞에 붙이는 한 줄(2026-10-25 예약, 2026-10-26 점검 H-1로 확장).
_CRISIS_FOLLOWUP = "힘드시면 언제든 자살예방 상담전화 109(24시간)에 이야기해 보세요."
# 라우터 실패 시 앞 턴 위기에 이어지는 짧은 말로 볼 최대 길이(공백 제외, 점검 M-1).
_CRISIS_CONTINUE_MAX_CHARS = 12
# 그 짧은 말 중 감사 표현은 위기가 이어지는 말로 보지 않는다(일반 폴백 + 109 한 줄, 2026-10-27).
# 작별 인사("안녕"·"잘 있어")는 위기 신호일 수 있어 넣지 않는다.
_CRISIS_CONTINUE_EXCLUDED = re.compile(r"고마|고맙|감사")
# 응급 키워드가 든 문장은 라우터를 이만큼만 기다린다(초, 2026-10-29 점검 L-4). 넘기면 라우터 실패 경로(키워드 분류 → 119 안내)로 간다.
# 라우터를 건너뛰지 않는 이유: 응급 키워드 문장도 라우터가 자해로 판정할 수 있다(예: 과다 복용).
_EMERGENCY_ROUTER_WAIT_SEC = 15.0
# 대화 이력은 요청의 history(클라이언트 값)가 아니라 서버 챗 기록에서 만든다(2026-10-28 S-7).
# 길이는 FE가 보내던 최대 24개(사용자·assistant 항목)와 같고, 각 단계는 지금처럼 그 안에서 다시 자른다.
_SERVER_HISTORY_ITEMS = 24
# 같은 사용자의 최근 기록 몇 건을 훑을지 - 다른 세션·다른 피보호자 상담이 사이에 끼어도 이번 세션 12턴을 찾을 만큼.
_SERVER_HISTORY_SCAN_ROWS = 60
# 버튼 턴은 FE 화면처럼 선택값을 사용자 말로 넣는다. 확인·취소·목록 버튼(토큰·ID 값)과 "다른 지역"·"다른 날짜"는 뺀다.
# 환자 이름·전화(patient_name·phone)는 넣지 않는다 - 저장 초안에 이미 있고, 이력은 라우터·추출(Claude)로 나간다(2026-10-29 점검 L-1).
_HISTORY_UI_FIELDS = frozenset({"hospital", "hospital_confirm", "department", "date", "time", "location", "location_confirm"})
_HISTORY_SKIP_VALUES = frozenset({"__other__", CHANGE_DATE_OPTION})
_CONTACT_UI_FIELDS = frozenset({"patient_name", "phone"})
_UI_LOG_RE = re.compile(r"^\[선택\] ([^:]+): (.*)$", re.S)
# 직전 턴이 응급(high)이었는데 이번 턴은 일반 답일 때 끝에 붙이는 한 줄(2026-10-29, 시험지 mt.01 "이제 좀 괜찮아졌어요" - 사용자 결정 (b)).
_EMERGENCY_FOLLOWUP = "증상이 다시 생기거나 심해지면 바로 119에 전화하세요."
# 되묻기 다음 턴이라도 인사·마무리 말이면 앞 증상과 잇지 않는다(C-1).
_FOLLOWUP_CLOSE_WORDS = re.compile(r"고마|고맙|감사|괜찮아요|됐어요|안녕|그만")
# 직전 기록 확인 함수에 "아직 안 읽음"을 알리는 값 - 요청 처음에 한 번 읽은 기록을 나눠 쓴다(점검 L-3).
_FETCH = object()


def _persistable_meta(meta: Any) -> dict:
    if not isinstance(meta, dict):
        return {}
    return {k: v for k, v in meta.items() if k not in _UNSTORED_META_KEYS}


def _mark_generation_failed(result: dict) -> dict:
    """생성 실패로 고정 안내가 나간 턴 표시(챗 기록 upstreamMeta.generationFailed). 자해 고정 응답처럼 engine 이 fallback 이
    아니어도 집계(/chat/stats fallbackReplies)가 실패 턴으로 셀 수 있게 한다. 응답 필드는 바뀌지 않는다."""
    meta = result.get("meta")
    result["meta"] = {**meta, "generationFailed": True} if isinstance(meta, dict) else {"generationFailed": True}
    return result


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in value]
    return value


class ChatService:
    """
    의료 챗 상담 서비스.
    - 1순위: MedGemma LLM (대화 이력 반영)
    - 2순위: 키워드 기반 fallback (모델 미로드/오류 시)
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    EMERGENCY_KEYWORDS = MedicalLlmService.EMERGENCY_KEYWORDS  # 단일 출처
    HIGH_RISK_KEYWORDS = ("가슴", "호흡곤란", "숨이", "의식", "실신", "출혈", "마비")
    MEDIUM_RISK_KEYWORDS = (
        "어지럽", "어지러", "어지럼", "복통", "발열", "기침", "두통", "구토", "아파", "아프", "통증", "더부룩", "속이", "소화",
        "무릎", "허리", "관절", "저리", "붓", "당뇨", "혈압", "혈당", "불면", "잠이", "수면", "약을", "약 먹", "복용",
        "설사", "변비", "가려", "피부", "기운", "피곤", "식욕",
    )
    RESERVATION_KEYWORDS = ("예약", "병원 찾아", "근처 병원", "진료 가능한", "응급실", "잡아", "접수", "병원 좀", "병원 추천", "진료 받")

    def process_message(self, db: Session, payload: ChatRequest) -> dict:
        # 요청 하나의 전체 마감(CHAT_DEADLINE_SEC, O-3). 라우터·예약 추출·예약 API·생성 대기가 남은 시간만큼만 기다린다.
        with chat_deadline():
            return self._process_message(db, payload)

    def _process_message(self, db: Session, payload: ChatRequest) -> dict:
        started = time.perf_counter()
        # 구간 시간(ms). 거치지 않은 구간은 None. 챗 기록 upstreamMeta.timing 과 [CHAT-TIMING] 로그에만 남는다(응답 필드 아님).
        timing: dict[str, int | None] = {"routerMs": None, "llmMs": None, "reservationMs": None}
        message = (payload.message or "").strip()
        ui_selection = None
        if payload.uiSelection:
            ui_selection = {
                "field": payload.uiSelection.field.strip(),
                "value": payload.uiSelection.value.strip(),
            }
        if not message and not ui_selection:
            raise ValueError("메시지를 입력해 주세요.")

        session_id = (payload.sessionId or "").strip() or f"chat-{uuid4().hex[:12]}"
        user_context = self._build_user_context(payload)
        ward_id = (payload.context.wardId or "").strip() if payload.context and payload.context.wardId else ""
        # 요청의 history 는 읽지 않는다 - 조작한 assistant 발화·지운 자해 발화로 경로·위기 응답·예약 값이 바뀌지 않게(S-7).
        # 아래 모든 단계(라우터·예약 추출·규칙 초안·MedGemma·위기 근거·대체 응답)가 payload.history 를 쓰므로 여기서 바꿔 끼운다.
        recent_logs = self._recent_logs(db, payload.userId)
        latest_log = recent_logs[0] if recent_logs else None
        history = self._server_history(db, payload.userId, session_id, ward_id, rows=recent_logs)
        # 버튼으로 입력했던 환자 이름·전화 - 이력(Claude)에는 넣지 않고, 저장 초안이 없을 때 예약 흐름이 빈 칸만 채운다(2026-10-29 점검 L-2).
        stored_contact = self._stored_contact(recent_logs, session_id, ward_id)
        # 직전 턴이 같은 상담(세션·피보호자)의 응급 답(high)이었는가 - 위기 고정 응답은 109 안내가 따로 붙어 제외한다.
        after_emergency = self._previous_turn_was_emergency(latest_log, session_id, ward_id)
        # 직전 턴이 같은 상담(세션·피보호자)의 되묻기였는가(C-1) - 이번 턴은 되묻지 않고 앞 질문과 이어 답한다.
        after_followup = self._previous_turn_was_followup(latest_log, session_id, ward_id)
        payload = payload.model_copy(update={"history": [ChatHistoryItem(**item) for item in history]})
        log_message = message
        # 위기 응답 직후 첫 일반 턴이면 답 앞에 109 안내 한 줄을 붙인다(한 번만 - 이 턴 기록엔 selfHarm이 없다).
        crisis_followup = False
        # 위기 뒤 해제된 턴인가. 해제된 턴은 생성이 실패해도 폴백이 history 근거로 위기 응답을 되살리지 않는다(2026-10-27 점검 M-1).
        crisis_released = False
        if not log_message and ui_selection:
            log_message = f"[선택] {ui_selection['field']}: {ui_selection['value']}"

        try:
            reservation_credential_service = get_reservation_credential_service()
            reservation_api_client = get_reservation_api_client()
            reservation_ui_fields = {
                "hospital",
                "hospital_confirm",
                "department",
                "date",
                "time",
                "patient_name",
                "phone",
                "location",
                "location_confirm",
                "reservation_followup",
                "reservation_id",
            } | CONFIRM_FIELDS
            # 1차 감별: GPT 라우터. 실패하면 키워드 분류로 내려간다.
            route_info: dict = {}
            # 자해·자살 신호는 진료과 답변·예약·MedGemma보다 먼저 고정 응답으로 보낸다(키워드 판정, 라우터 결과와 무관).
            self_harm_keyword = bool(message) and bool(MedicalLlmService.self_harm_signals(message))
            department_answer = self._department_question_reply(message, history) if message and not self_harm_keyword else None
            if department_answer:
                result = department_answer
            elif self_harm_keyword:
                # 위기 문장은 라우터(외부 API)로 보내지 않는다 - 라우터 결과와 상관없이 고정 응답이 나간다.
                route_info = {"route": "symptom", "engine": "rule", "normalized_message": message, "symptom_summary": "",
                              "emergency": True, "reason": "self-harm-keyword"}
            elif message:
                router_started = time.perf_counter()
                # 응급 키워드 문장은 라우터 호출만 15초 안으로(뒤 단계 몫 ROUTER_RESERVE_SEC 는 그대로 남긴다).
                router_wait = (
                    tighter_deadline(_EMERGENCY_ROUTER_WAIT_SEC + ROUTER_RESERVE_SEC)
                    if MedicalLlmService.emergency_signals(message)
                    else contextlib.nullcontext()
                )
                try:
                    with router_wait:
                        route_info = get_chat_router_service().route(
                            message, history, user_context,
                            # 새 예약 요청이면 직전 턴의 병원·시간을 끌어오지 않는다(지난 요청이 새 요청에 붙는 것 방지).
                            prev_reservation=None
                            if ReservationOrchestrator._is_new_request(message)
                            else self._last_reservation_state(db, payload.userId, session_id, ward_id, latest=latest_log),
                        )
                except Exception as exc:  # noqa: BLE001
                    # 원인은 고정 코드만(max_tokens·parse·api_<상태>…) - 입력·모델 출력은 남기지 않는다.
                    _LOG.warning("[CHAT-ROUTER-FAILED] cause=%s fallback=keywords", failure_cause(exc))
                timing["routerMs"] = _elapsed_ms(router_started)
                route_info = self._apply_router_overrides(route_info, message)
            if department_answer:
                route_info = {"route": "general", "engine": "rule", "normalized_message": message, "symptom_summary": "", "emergency": False}
            elif not route_info and not message and ui_selection and ui_selection["field"] in reservation_ui_fields:
                # UI 블록 선택 턴: 직전 규격화 문장에 선택값을 붙여 상태를 이어간다.
                prev = self._last_reservation_state(db, payload.userId, session_id, ward_id, latest=latest_log) or "병원 예약:"
                value = ui_selection["value"]
                carry = ui_selection["field"] in {"location", "location_confirm", "department", "hospital", "hospital_confirm", "date", "time"} and value != "__other__"
                route_info = {
                    "route": "reservation",
                    "normalized_message": f"{prev} / {value}" if carry else prev,
                    "symptom_summary": "",
                    "emergency": False,
                    "engine": "ui",
                }
            if not route_info and not department_answer:
                kw_intent = self.classify_intent(message)
                route_info = {
                    "route": {"hospital_reservation": "reservation", "medical_advice": "symptom", "emergency_guidance": "symptom"}.get(kw_intent, "general"),
                    "normalized_message": message,
                    "symptom_summary": message,
                    "emergency": kw_intent == "emergency_guidance",
                    "engine": "keyword",
                }
            route_info = self._force_emergency(route_info, message)
            self_harm_trigger = self._self_harm_trigger(self_harm_keyword, route_info, history)
            router_failed = route_info.get("engine") == "keyword"
            if router_failed and self_harm_trigger is None and self._crisis_continues_without_router(message, history):
                # 라우터 실패(키워드 분류): 앞 턴 위기 뒤 짧은 말("어떻게 해야 할지 모르겠어")은 라우터 판단 없이도 위기 응답으로.
                _LOG.info("[CHAT-SAFETY] crisis_continues_fallback=true text_len=%s", len(message))
                self_harm_trigger = "history"
                route_info = {**route_info, "route": "symptom", "emergency": True, "reason": "crisis-continues-fallback"}
            prev_meta = (
                self._previous_turn_meta(db, payload.userId, session_id, latest=latest_log) if message and not self_harm_keyword else {}
            )
            after_crisis = bool(prev_meta.get("selfHarm"))
            # 위기 뒤 해제된 턴(서버 기록 crisisFollowup)의 다음 턴도 history(앞 턴 자해 키워드) 근거로 다시 갇히지 않게
            # 같은 해제 검사를 탄다(2026-10-27). 클라이언트 history만 있고 서버 기록이 없으면 풀지 않는다.
            # 라우터가 실패한 턴의 history 근거는 라우터 판단이 없어 그대로 둔다.
            history_after_release = (
                self_harm_trigger == "history" and not router_failed and not after_crisis and bool(prev_meta.get("crisisFollowup"))
            )
            release = (
                self._release_after_crisis(route_info, message, self_harm_trigger) if after_crisis or history_after_release else None
            )
            if release is not None:
                if release == "reservation":
                    # "응 예약해줘": 증상 단어가 없고 예약 의도가 분명하면 예약으로(새 예약으로 시작).
                    _LOG.info("[CHAT-SAFETY] resume_reservation_after_crisis=true trigger=%s", self_harm_trigger)
                    crisis_followup = crisis_released = True
                    self_harm_trigger = None
                    route_info = {**route_info, "route": "reservation", "emergency": False, "emergency_signals": [],
                                  "normalized_message": message, "reason": "resume-after-crisis"}
                else:
                    # 증상·일반 대화는 라우터 경로 그대로(라우터가 응급이라고 했으면 위기 고정 응답이 아닌 일반 응급 안내로).
                    _LOG.info("[CHAT-SAFETY] crisis_released=true route=%s trigger=%s", route_info["route"], self_harm_trigger)
                    crisis_followup = crisis_released = True
                    self_harm_trigger = None
                    route_info = {**route_info, "reason": "release-after-crisis"}
            elif self_harm_trigger is None and message and (
                after_crisis or (router_failed and MedicalLlmService.self_harm_in_history(history))
            ):
                crisis_followup = True
            self_harm = self_harm_trigger is not None
            # 상담 내용(증상 요약·예약 문장)은 로그에 남기지 않는다 - 길이만.
            _LOG.info(
                "[라우터] route=%s engine=%s text_len=%s",
                route_info["route"],
                route_info.get("engine"),
                len(str(route_info.get("normalized_message") if route_info["route"] == "reservation" else route_info.get("symptom_summary") or "")),
            )
            reservation_flow_hint = not self_harm and (
                route_info["route"] == "reservation" or bool(ui_selection and ui_selection["field"] in reservation_ui_fields)
            )
            reservation_api_key = None
            credential_unavailable = False
            deadline_flow = None
            reservation_started = time.perf_counter()
            # 예약 계정 비밀값이 없으면 예약 기능만 끈다(상담·이상감지 등 나머지는 그대로).
            reservation_disabled = reservation_flow_hint and (
                not self._settings.has_reservation_account_secret
                # 방금 회원 데이터를 지운 ID면 예약 계정을 되살리지 않는다(탈퇴 정리 점검 L-2) - 예약 꺼짐과 같은 안내로 끝낸다.
                or recently_purged(ward_id or payload.userId)
            )
            if department_answer:
                pass  # 이미 result가 있다
            elif self_harm:
                family = MedicalLlmService.about_family(message)
                # 상담 내용은 남기지 않는다 - 경로 플래그만.
                _LOG.info("[CHAT-SAFETY] self_harm=true path=main trigger=%s family=%s", self_harm_trigger, family)
                result = MedicalLlmService.self_harm_response(family=family, trigger=self_harm_trigger)
            elif reservation_flow_hint:
                stored_flow = self._last_flow_state(db, payload.userId, session_id, latest=latest_log)
                try:
                    # 보호자가 피보호자 대신이면 피보호자 계정으로 예약한다(피보호자 웹·보호자 웹과 같은 목록).
                    reservation_api_key = reservation_credential_service.ensure_credential(
                        db,
                        user_id=ward_id or payload.userId,
                        client=reservation_api_client,
                        name=str(user_context.get("name") or ""),
                        phone=str(user_context.get("phone") or ""),
                    )
                    if reservation_api_key is None and not reservation_disabled:
                        # 회원 예약 계정 키를 못 받았다(로그인·가입·발급 실패, 예약 서비스 장애) - 키 없이 예약 서비스를 부르지 않고
                        # 예약 꺼짐 안내로 끝낸다(2026-10-30, 공용 키·공용 계정 대체 제거). 진행 중이던 초안은 다음 턴에 넘긴다.
                        _LOG.info("[CHAT-RESERVATION] credential_unavailable")
                        reservation_disabled = credential_unavailable = True
                        reservation_result = None
                        deadline_flow = {k: stored_flow[k] for k in ("flowState", "pendingAction") if isinstance(stored_flow.get(k), dict)}
                    else:
                        reservation_result = self._run_reservation(
                            payload, route_info, message, reservation_api_key, stored_flow, reservation_disabled, stored_contact
                        )
                except ChatDeadlineExceeded as exc:
                    # 마감에 걸려 새 예약 호출을 시작하지 않았다 - 아래 기존 실패 문구로 끝내고 직전 초안은 다음 턴에 넘긴다.
                    _LOG.warning("[CHAT-DEADLINE] stage=%s action=reservation_fallback", exc.stage)
                    reservation_result = None
                    # 확인 대기(pendingAction)도 넘긴다 - 마감은 토큰 선점 전에만 여기로 오므로 같은 버튼으로 다시 시도할 수 있다(점검 L-1).
                    deadline_flow = {k: stored_flow[k] for k in ("flowState", "pendingAction") if isinstance(stored_flow.get(k), dict)}
                else:
                    if not credential_unavailable:
                        deadline_flow = None
                timing["reservationMs"] = _elapsed_ms(reservation_started)
                if reservation_result is not None:
                    result = reservation_result
                else:
                    result = {
                        "reply": (
                            "지금은 병원 예약 기능을 사용할 수 없어요. 병원에 직접 전화해 예약해 주세요."
                            if reservation_disabled
                            else "예약 정보를 처리하지 못했습니다. 잠시 후 다시 시도해 주세요."
                        ),
                        "riskLevel": "medium",
                        "recommendedAction": "guardian_contact",
                        "reservationRequired": True,
                        "intent": "hospital_reservation",
                        "engine": "claude",
                        "modelName": self._settings.claude_model_name,
                        "type": "message",
                        "ui": None,
                        "tool": None,
                        "data": None,
                        "summary": None,
                        "possibleCauses": [],
                        "homeCare": [],
                        "visitHospitalIf": [],
                        "emergencyWarning": [],
                        "meta": {"reservationFlowHint": True, **(deadline_flow or {})},
                        "decisionTrace": [{"event": "reservation_credential_unavailable" if credential_unavailable
                                           else "reservation_disabled" if reservation_disabled else "reservation_flow_fallback"}],
                    }
            elif self._should_ask_followup(route_info, message, ui_selection, crisis_followup, after_emergency, after_followup):
                # 정보가 부족한 일반 증상 한마디 - 언제부터·얼마나·동반 증상을 한 번 묻는다(C-1, MedGemma 를 부르지 않는다).
                proxy = bool(ward_id) or MedicalLlmService.about_family(message)
                _LOG.info("[CHAT-QUALITY] followup_asked=true proxy=%s", proxy)
                result = MedicalLlmService.followup_question_response(message, proxy=proxy)
            else:
                local_llm = get_medical_llm_service()
                llm_started = time.perf_counter()
                try:
                    gen_message = message
                    if route_info["route"] == "symptom":
                        llm_intent = "emergency_guidance" if route_info.get("emergency") else "medical_advice"
                    else:
                        llm_intent = "general_chat"
                    processed_symptom = route_info.get("symptom_summary") if route_info["route"] == "symptom" else None
                    followup_answer = self._followup_answer_text(route_info, message, history, after_followup)
                    if followup_answer:
                        # 되묻기에 답한 턴: 짧은 답("어제부터요")만으로는 증상을 모르니 앞 질문과 이어서 상담한다.
                        _LOG.info("[CHAT-QUALITY] followup_answered=true route=%s", route_info["route"])
                        gen_message = followup_answer
                        llm_intent = "medical_advice"
                        processed_symptom = (route_info.get("symptom_summary") if route_info["route"] == "symptom" else None) or followup_answer
                    result = local_llm.generate(
                        gen_message,
                        self._medgemma_context(user_context),
                        history,
                        intent=llm_intent,
                        processed_symptom=processed_symptom,
                    )
                    timing["llmMs"] = _elapsed_ms(llm_started)
                except Exception as exc:
                    timing["llmMs"] = _elapsed_ms(llm_started)
                    # 옛 챗 서버로 넘기던 폴백은 없앴다(2026-10-22) - 고정 문구로 끝낸다. 로그는 예외 클래스명만(상담 내용 금지).
                    _LOG.warning("[CHAT-LLM-FAILED] %s", type(exc).__name__)
                    result = _mark_generation_failed(self._fallback_reply(
                        message, history, emergency=bool(route_info.get("emergency")), history_basis=not crisis_released
                    ))
        except Exception as exc:
            # 예외 문구·스택에는 SQL 파라미터·요청 값이 섞일 수 있어 클래스명과 발생 위치만 남긴다.
            _LOG.error("chat process_message fallback: %s at %s", type(exc).__name__, _origin(exc))
            # 직전 턴이 응급 답이면 대체 응답도 응급 쪽(119·경고)으로 - "안정 상태로 보입니다"가 나가지 않게(2026-10-29 점검 M-1).
            result = _mark_generation_failed(self._fallback_reply(message or log_message, history, emergency=after_emergency))
            route_info = {}
        if crisis_followup and route_info and isinstance(result.get("reply"), str):
            result["reply"] = f"{_CRISIS_FOLLOWUP}\n{result['reply']}"
            result.setdefault("meta", {})
            if isinstance(result["meta"], dict):
                result["meta"]["crisisFollowup"] = True
        if (
            after_emergency
            and route_info
            and route_info.get("route") != "reservation"
            and not crisis_followup
            and message
            and not MedicalLlmService.emergency_signals(message)
            and not MedicalLlmService.self_harm_signals(message)
            and result.get("riskLevel") != "high"
            and isinstance(result.get("reply"), str)
            # 답에 이미 119 안내가 있으면 되풀이하지 않는다(2026-10-29 점검 L-1, #97 위험 신호 문장과 같은 기준).
            and "119" not in result["reply"]
        ):
            result["reply"] = f"{result['reply']}\n{_EMERGENCY_FOLLOWUP}"
            result.setdefault("meta", {})
            if isinstance(result["meta"], dict):
                result["meta"]["emergencyFollowup"] = True
        post_started = time.perf_counter()
        if route_info:
            result.setdefault("meta", {})
            if isinstance(result["meta"], dict):
                result["meta"]["router"] = {k: route_info.get(k) for k in ("route", "engine", "reason", "normalized_message", "symptom_summary", "emergency", "emergency_signals")}
                # 라우터가 자해로 봤는지(점검 L-1). 라우터를 거치지 않은 턴(키워드·규칙·UI)은 None - 모르는 값을 false로 채우지 않는다.
                router_self_harm = route_info.get("self_harm")
                result["meta"]["router"]["self_harm"] = router_self_harm if isinstance(router_self_harm, bool) else None
                # 라우터 기억은 이 상담의 피보호자에게 묶는다 - 다른 피보호자 상담에서 앞 사람의 병원·날짜를 이어 쓰지 않게.
                result["meta"]["router"]["wardId"] = ward_id

        timing["postMs"] = _elapsed_ms(post_started)
        timing["totalMs"] = _elapsed_ms(started)  # 저장 직전까지. 저장 시간은 행을 쓴 뒤에야 알아 로그에만 남긴다.
        result.setdefault("meta", {})
        if isinstance(result["meta"], dict):
            result["meta"]["timing"] = dict(timing)

        context_payload = {
            "sessionId": session_id,
            "userContext": user_context,
            "historyLength": len(history),
            "engine": result.get("engine", "fallback"),
            "modelName": result.get("modelName"),
            "type": result.get("type", "message"),
            "ui": result.get("ui"),
            "tool": result.get("tool"),
            "data": result.get("data"),
            "summary": result.get("summary"),
            "possibleCauses": result.get("possibleCauses") or [],
            "homeCare": result.get("homeCare") or [],
            "visitHospitalIf": result.get("visitHospitalIf") or [],
            "emergencyWarning": result.get("emergencyWarning") or [],
            "decisionTrace": result.get("decisionTrace") or [],
            "upstreamMeta": _persistable_meta(result.get("meta")),
        }

        log = ChatLog(
            chat_no=f"CHAT-{uuid4().hex[:12]}",
            user_id=payload.userId,
            message=log_message,
            context_json=json.dumps(_json_safe(context_payload), ensure_ascii=False),
            reply=result["reply"],
            risk_level=result["riskLevel"],
            recommended_action=result["recommendedAction"],
            reservation_required=bool(result["reservationRequired"]),
        )
        save_started = time.perf_counter()
        db.add(log)
        db.commit()
        db.refresh(log)
        # 경로·엔진·구간 ms 만 남긴다(상담 내용·사용자 문장 금지).
        _LOG.info(
            "[CHAT-TIMING] route=%s engine=%s router_ms=%s llm_ms=%s reservation_ms=%s post_ms=%s save_ms=%s total_ms=%s",
            route_info.get("route") if route_info else None,
            result.get("engine", "fallback"),
            timing["routerMs"],
            timing["llmMs"],
            timing["reservationMs"],
            timing["postMs"],
            _elapsed_ms(save_started),
            _elapsed_ms(started),
        )

        return {
            "sessionId": session_id,
            "reply": result["reply"],
            "riskLevel": result["riskLevel"],
            "recommendedAction": result["recommendedAction"],
            "reservationRequired": bool(result["reservationRequired"]),
            "intent": result["intent"],
            "engine": result.get("engine", "fallback"),
            "modelName": result.get("modelName"),
            "type": result.get("type", "message"),
            "ui": result.get("ui"),
            "tool": result.get("tool"),
            "toolData": result.get("data"),
            "summary": result.get("summary"),
            "possibleCauses": result.get("possibleCauses") or [],
            "homeCare": result.get("homeCare") or [],
            "visitHospitalIf": result.get("visitHospitalIf") or [],
            "emergencyWarning": result.get("emergencyWarning") or [],
            "chatLogId": log.id,
            "chatNo": log.chat_no,
        }

    def _run_reservation(
        self,
        payload: ChatRequest,
        route_info: dict,
        message: str,
        reservation_api_key: str | None,
        stored_flow: dict,
        reservation_disabled: bool,
        stored_contact: dict | None = None,
    ) -> dict | None:
        """규격화된 문장("병원 예약: 서울 강남구 / 이비인후과 / ...")을 예약 오케스트레이터에 넘긴다. 처리하지 못하면 None."""
        if reservation_disabled:
            return None
        routed_context = payload.context
        if routed_context and routed_context.wardId:
            # 환자는 피보호자다 - 예약 초안의 환자명·연락처를 피보호자 것으로 바꿔 넘긴다.
            routed_context = routed_context.model_copy(
                update={"name": routed_context.wardName or routed_context.name, "phone": routed_context.wardPhone or routed_context.phone}
            )
        routed_payload = payload.model_copy(update={"message": route_info["normalized_message"], "context": routed_context})
        return get_reservation_orchestrator().handle(
            routed_payload,
            {
                "rawMessage": message,
                "model": self._settings.claude_model_name,
                "schemaVersion": "ai-reservation-v1",
                "reservationApiKey": reservation_api_key,
                # 직전 턴이 남긴 확인 대기·고치기 상태. 서버 DB 값이라 클라이언트가 바꿀 수 없다.
                "storedFlow": stored_flow,
                "storedContact": stored_contact or {},
            },
        )

    @staticmethod
    def _recent_logs(db: Session, user_id: str) -> list[ChatLog]:
        """같은 사용자의 최근 기록(최신 순). 요청 처음에 한 번 읽어 이력 조립과 직전 턴 확인이 나눠 쓴다(점검 L-3)."""
        return db.query(ChatLog).filter(ChatLog.user_id == user_id).order_by(ChatLog.id.desc()).limit(_SERVER_HISTORY_SCAN_ROWS).all()

    @staticmethod
    def _row_ward_id(ctx: dict) -> str:
        """기록이 어느 피보호자 상담인가. router.wardId 를 먼저 보고, router 칸이 없는 기록(내부 오류 대체 응답)은
        같은 기록의 userContext.wardId 로 본다(2026-10-29 점검 M-1 - 대리 상담 오류 턴이 본인 상담 이력으로 들어가던 것)."""
        meta = ctx.get("upstreamMeta") if isinstance(ctx.get("upstreamMeta"), dict) else {}
        router = meta.get("router") if isinstance(meta.get("router"), dict) else None
        if router is not None and "wardId" in router:
            return str(router.get("wardId") or "")
        user_context = ctx.get("userContext") if isinstance(ctx.get("userContext"), dict) else {}
        return str(user_context.get("wardId") or "")

    @staticmethod
    def _previous_turn_was_followup(latest: ChatLog | None, session_id: str, ward_id: str) -> bool:
        """직전 기록이 같은 세션·피보호자의 되묻기(upstreamMeta.followupAsked)였는가."""
        if latest is None:
            return False
        try:
            ctx = json.loads(latest.context_json or "{}")
        except json.JSONDecodeError:
            return False
        if not isinstance(ctx, dict) or ctx.get("sessionId") != session_id or ChatService._row_ward_id(ctx) != ward_id:
            return False
        meta = ctx.get("upstreamMeta") if isinstance(ctx.get("upstreamMeta"), dict) else {}
        return bool(meta.get("followupAsked"))

    @staticmethod
    def _should_ask_followup(
        route_info: dict, message: str, ui_selection: dict | None, crisis_followup: bool, after_emergency: bool, after_followup: bool
    ) -> bool:
        """되묻기 턴 조건(C-1): 응급이 아닌 증상 경로의 일반 문장, 위기 뒤·응급 다음 턴·되묻기 다음 턴이 아님 + 문장 조건."""
        if not message or ui_selection or crisis_followup or after_emergency or after_followup:
            return False
        if route_info.get("route") != "symptom" or route_info.get("emergency") or route_info.get("self_harm"):
            return False
        return MedicalLlmService.needs_followup_question(message)

    @staticmethod
    def _followup_answer_text(route_info: dict, message: str, history: list[dict[str, str]], after_followup: bool) -> str | None:
        """되묻기 다음 턴이면 "앞 질문 / 이번 답" 문장. 응급·예약·인사(고마워요 등)·앞 질문을 못 찾으면 None(이번 문장 그대로)."""
        if not after_followup or not message or route_info.get("emergency") or route_info.get("route") == "reservation":
            return None
        if _FOLLOWUP_CLOSE_WORDS.search(message):
            return None
        previous_user = next((h.get("content", "") for h in reversed(history or []) if h.get("role") == "user"), "")
        if not previous_user.strip():
            return None
        return f"{previous_user.strip()} / {message.strip()}"

    @staticmethod
    def _previous_turn_was_emergency(latest: ChatLog | None, session_id: str, ward_id: str) -> bool:
        """직전 기록이 같은 세션·피보호자의 응급 답(high)이고 위기 고정 응답이 아니었는가."""
        if latest is None or latest.risk_level != "high":
            return False
        try:
            ctx = json.loads(latest.context_json or "{}")
        except json.JSONDecodeError:
            return False
        if not isinstance(ctx, dict) or ctx.get("sessionId") != session_id or ChatService._row_ward_id(ctx) != ward_id:
            return False
        meta = ctx.get("upstreamMeta") if isinstance(ctx.get("upstreamMeta"), dict) else {}
        return not meta.get("selfHarm")

    @staticmethod
    def _server_history(
        db: Session, user_id: str, session_id: str, ward_id: str, rows: list[ChatLog] | None = None
    ) -> list[dict[str, str]]:
        """같은 사용자·세션·피보호자의 서버 챗 기록으로 대화 이력을 만든다(오래된 순, 최근 _SERVER_HISTORY_ITEMS개).
        피보호자 구분은 _row_ward_id. 기록이 없으면(첫 턴·새 세션) 빈 목록. rows 를 주면 다시 읽지 않는다."""
        if rows is None:
            rows = ChatService._recent_logs(db, user_id)
        items: list[dict[str, str]] = []
        for log in reversed(rows):
            try:
                ctx = json.loads(log.context_json or "{}")
            except json.JSONDecodeError:
                continue
            if not isinstance(ctx, dict) or ctx.get("sessionId") != session_id:
                continue
            if ChatService._row_ward_id(ctx) != ward_id:
                continue
            user_text = ChatService._history_user_text(log.message or "")
            if user_text is None:
                continue
            items.append({"role": "user", "content": user_text})
            if (log.reply or "").strip():
                items.append({"role": "assistant", "content": log.reply})
        return items[-_SERVER_HISTORY_ITEMS:]

    @staticmethod
    def _stored_contact(rows: list[ChatLog], session_id: str, ward_id: str) -> dict[str, str]:
        """같은 세션·피보호자에서 버튼으로 마지막에 입력한 환자 이름·전화(최근 기록부터). 없으면 빈 dict.
        이력에는 넣지 않는 값이라(Claude로 나가지 않게) 예약 흐름에 따로 넘긴다(점검 L-2)."""
        found: dict[str, str] = {}
        for log in rows:
            matched = _UI_LOG_RE.match(log.message or "")
            if not matched or matched.group(1).strip() not in _CONTACT_UI_FIELDS:
                continue
            field, value = matched.group(1).strip(), matched.group(2).strip()
            if field in found or not value:
                continue
            try:
                ctx = json.loads(log.context_json or "{}")
            except json.JSONDecodeError:
                continue
            if not isinstance(ctx, dict) or ctx.get("sessionId") != session_id or ChatService._row_ward_id(ctx) != ward_id:
                continue
            found[field] = value
        return found

    @staticmethod
    def _history_user_text(message: str) -> str | None:
        """기록의 사용자 문장. 버튼 턴("[선택] field: value")은 선택값만, 이력에 넣지 않는 버튼 턴은 None."""
        matched = _UI_LOG_RE.match(message)
        if not matched:
            return message.strip() or None
        field, value = matched.group(1).strip(), matched.group(2).strip()
        if field not in _HISTORY_UI_FIELDS or not value or value in _HISTORY_SKIP_VALUES:
            return None
        return value

    @classmethod
    def _release_after_crisis(cls, route_info: dict, message: str, trigger: str | None) -> str | None:
        """위기 응답 직후 턴을 위기 고정 응답에서 풀어 줄지. "reservation" = 예약으로 재개 / "release" = 라우터 경로 그대로 / None = 위기 유지.
        라우터가 앞 턴 위기를 이번 턴 신호로 옮겨 평범한 말까지 같은 고정 응답에 갇히던 것(점검 H-1): 근거가 router·history이고
        이번 문장에 자해·응급 키워드가 없으며 라우터가 이번 문장 기준으로 자해가 아니라고 하면(self_harm=false) 푼다.
        라우터가 이번 문장도 자해라고 하면(self_harm=true) 예약 의도가 있어도 위기 응답을 유지한다(점검 M-2).
        평가 스크립트(tests/eval/crisis_followup_router.py)도 같은 함수로 잰다."""
        if trigger not in {"router", "history"} or not message:
            return None
        if MedicalLlmService.emergency_signals(message) or MedicalLlmService.self_harm_signals(message):
            return None
        if route_info.get("self_harm") is not True and cls._reservation_resume_intent(message):
            return "reservation"
        if route_info.get("self_harm") is False:
            return "release"
        return None

    @staticmethod
    def _reservation_resume_intent(message: str) -> bool:
        """위기 응답 뒤 예약을 다시 이어가려는 문장인가 - 증상 단어가 없고 예약 의도가 분명해야 한다.
        응급·자해 키워드("응급실"도 응급 키워드)는 호출 전에 걸러진다."""
        if not message or any(word in message for word in MedicalLlmService.MEDICAL_KEYWORDS):
            return False
        return ReservationOrchestrator._looks_like_reservation(message)

    @staticmethod
    def _crisis_continues_without_router(message: str, history: list[dict[str, str]]) -> bool:
        """라우터가 실패했을 때 위기가 이어지는 말로 볼 것인가(점검 M-1): 직전 사용자 턴 2개에 자해 키워드가 있고,
        이번 문장이 짧고(공백 빼고 12자 이하) 증상·예약 의도·감사 표현이 없다. 라우터 판단이 없어 안전 쪽으로 넓게 잡는다."""
        if not message or not MedicalLlmService.self_harm_in_history(history):
            return False
        if len(re.sub(r"\s", "", message)) > _CRISIS_CONTINUE_MAX_CHARS:
            return False
        if any(word in message for word in MedicalLlmService.MEDICAL_KEYWORDS) or _CRISIS_CONTINUE_EXCLUDED.search(message):
            return False
        return not ReservationOrchestrator._looks_like_reservation(message)

    @staticmethod
    def _previous_turn_meta(db: Session, user_id: str, session_id: str, latest: Any = _FETCH) -> dict:
        """같은 세션 직전 턴의 위기 표시(selfHarm = 위기 고정 응답 / crisisFollowup = 위기 뒤 해제되어 109 한 줄이 붙음).
        클라이언트 history가 아니라 서버 챗 기록(upstreamMeta)으로 본다. 다른 세션이거나 기록이 없으면 빈 dict."""
        log = ChatService._latest_log(db, user_id) if latest is _FETCH else latest
        if not log:
            return {}
        try:
            ctx = json.loads(log.context_json or "{}")
        except json.JSONDecodeError:
            return {}
        if ctx.get("sessionId") != session_id:
            return {}
        meta = ctx.get("upstreamMeta") or {}
        return {k: bool(meta.get(k)) for k in ("selfHarm", "crisisFollowup")}

    @staticmethod
    def _last_flow_state(db: Session, user_id: str, session_id: str, latest: Any = _FETCH) -> dict:
        """직전 턴(같은 세션)이 남긴 예약 확인 대기(pendingAction)·고치기 상태(flowState). 그 사이에 다른 대화가 끼면 비어 있다."""
        log = ChatService._latest_log(db, user_id) if latest is _FETCH else latest
        if not log:
            return {}
        try:
            ctx = json.loads(log.context_json or "{}")
        except json.JSONDecodeError:
            return {}
        if ctx.get("sessionId") != session_id:
            return {}
        meta = ctx.get("upstreamMeta") or {}
        return {k: meta[k] for k in ("pendingAction", "flowState") if isinstance(meta.get(k), dict)}

    @staticmethod
    def _last_reservation_state(
        db: Session, user_id: str, session_id: str, ward_id: str = "", latest: Any = _FETCH
    ) -> str | None:
        """같은 세션의 직전 턴이 예약 흐름이었으면 그때의 규격화 문장을 돌려준다 (라우터의 상태 기억).
        직전 턴이 다른 피보호자의 상담이었으면 비운다. wardId 를 남기기 전의 기록은 본인 상담("")으로 본다."""
        log = ChatService._latest_log(db, user_id) if latest is _FETCH else latest
        if not log:
            return None
        try:
            ctx = json.loads(log.context_json or "{}")
        except json.JSONDecodeError:
            return None
        if ctx.get("sessionId") != session_id:
            return None
        router = (ctx.get("upstreamMeta") or {}).get("router") or {}
        # 피보호자 구분은 대화 이력과 같은 기준(router.wardId, 없으면 userContext.wardId - 2026-10-29 점검 L-3).
        if ChatService._row_ward_id(ctx) != ward_id:
            return None
        return router.get("normalized_message") if router.get("route") == "reservation" else None

    @staticmethod
    def _latest_log(db: Session, user_id: str) -> ChatLog | None:
        return db.query(ChatLog).filter(ChatLog.user_id == user_id).order_by(ChatLog.id.desc()).first()

    @staticmethod
    def _medgemma_context(user_context: dict[str, object]) -> dict[str, object]:
        """MedGemma 프롬프트에는 나이·성별만 넘긴다 - 이름·연락처·주소·생년월일·ID 는 답에 쓸 곳이 없다.
        원본(예약 환자 정보·챗 기록 피보호자 구분)은 그대로 둔다. 생년월일만 있으면 만 나이로 바꿔 넘긴다."""
        ctx: dict[str, object] = {}
        age = user_context.get("age")
        # 나이는 0~130 정수, 성별은 정해진 값만(그 밖이면 칸을 뺀다 - 2026-10-29 #107 점검 L-3).
        if not (isinstance(age, int) and not isinstance(age, bool) and 0 <= age <= 130):
            age = _age_from_birth_date(str(user_context.get("birthDate") or ""))
        if age is not None:
            ctx["age"] = age
        gender = str(user_context.get("gender") or "").strip().upper()
        if gender in _MEDGEMMA_GENDERS:
            ctx["gender"] = gender
        return ctx

    @staticmethod
    def _build_user_context(payload: ChatRequest) -> dict[str, object]:
        if not payload.context:
            return {}
        ctx: dict[str, object] = {}
        if payload.context.age is not None:
            ctx["age"] = payload.context.age
        if payload.context.email:
            ctx["email"] = payload.context.email.strip()
        if payload.context.name:
            ctx["name"] = payload.context.name.strip()
        if payload.context.phone:
            ctx["phone"] = payload.context.phone.strip()
        if payload.context.wardId:
            # 보호자 상담은 피보호자가 환자다 - 예약 환자명·연락처를 피보호자로 둔다.
            ctx["wardId"] = payload.context.wardId.strip()
            if payload.context.wardName:
                ctx["name"] = payload.context.wardName.strip()
            if payload.context.wardPhone:
                ctx["phone"] = payload.context.wardPhone.strip()
        if payload.context.gender:
            ctx["gender"] = payload.context.gender
        if payload.context.birthDate:
            ctx["birthDate"] = payload.context.birthDate.strip()
        if payload.context.postcode:
            ctx["postcode"] = payload.context.postcode.strip()
        if payload.context.address:
            ctx["address"] = payload.context.address.strip()
        if payload.context.addressDetail:
            ctx["addressDetail"] = payload.context.addressDetail.strip()
        if payload.context.guardianId is not None:
            ctx["guardianId"] = payload.context.guardianId
        if payload.context.location:
            ctx["location"] = payload.context.location.strip()
        if payload.context.role:
            ctx["role"] = payload.context.role.strip()
        return ctx

    @staticmethod
    def _force_emergency(route_info: dict, message: str) -> dict:
        """응급 신호가 있으면 라우트와 상관없이 증상 상담(119 안내)으로 보낸다.
        라우터가 emergency=True를 주고도 route를 reservation·general로 두거나, 라우터가 놓친 키워드가 있어도 응급이 묻히지 않게."""
        signals = MedicalLlmService.emergency_signals(message)
        self_harm = MedicalLlmService.self_harm_signals(message)
        if not route_info or not (signals or self_harm or route_info.get("emergency")):
            return route_info
        return {
            **route_info,
            "route": "symptom",
            "emergency": True,
            "emergency_signals": signals,
            "symptom_summary": route_info.get("symptom_summary") or message,
        }

    @staticmethod
    def _apply_router_overrides(route_info: dict, message: str) -> dict:
        """라우터 결과 뒤에 붙이는 예약 고정 규칙. 평가 스크립트(tests/eval/compare_router.py)도 같은 함수로 최종 경로를 잰다."""
        # "머리가 아파서 병원 예약하고 싶어요"처럼 증상과 예약이 섞이면 라우터가 증상 상담으로 보내기도 한다 - 예약 요청이 분명하면 예약으로 고정.
        if (
            route_info
            and route_info.get("route") != "reservation"
            and not route_info.get("emergency")
            and ReservationOrchestrator._is_new_request(message)
        ):
            route_info = {**route_info, "route": "reservation", "normalized_message": message, "reason": "new-request-override"}
        # "예약 취소할래"·"예약 바꿔줘"도 라우터가 다른 경로로 보내면 예약 흐름으로 고정한다(응급 신호가 있으면 그대로 상담).
        if (
            route_info
            and route_info.get("route") != "reservation"
            and not route_info.get("emergency")
            and (
                ReservationOrchestrator._is_cancel_request(message)
                or ReservationOrchestrator._is_change_request(message)
                or ReservationOrchestrator._is_cancel_change_question(message)
            )
        ):
            route_info = {**route_info, "route": "reservation", "normalized_message": message, "reason": "cancel-change-override"}
        if route_info and route_info.get("route") == "reservation" and ReservationOrchestrator._is_new_request(message):
            # 새 요청은 라우터 규격화 문장(이전 병원·시간이 딸려옴) 대신 원문을 넘긴다. 다음 턴의 상태 기억도 원문 기준.
            route_info = {**route_info, "normalized_message": message}
        return route_info

    @staticmethod
    def _self_harm_trigger(keyword: bool, route_info: dict, history: list[dict[str, str]]) -> str | None:
        """위기 고정 응답을 낼 근거. keyword = 이번 문장 키워드 / router = 라우터 self_harm 칸이 true /
        history = 이번 턴이 응급이고 직전 사용자 턴 2개에 자해 신호. 응급이 아니면 앞 턴 때문에 위기 응답에 가두지 않는다.
        라우터 요약 문구(자살·자해)로 보던 판정은 뺐다(2026-10-27) - 측정에서 칸 true 밖으로 잡은 건이 0건이었다."""
        if keyword:
            return "keyword"
        if route_info and route_info.get("self_harm") is True:
            return "router"
        if not route_info or not route_info.get("emergency"):
            return None
        if MedicalLlmService.self_harm_in_history(history):
            return "history"
        return None

    @classmethod
    def classify_intent(cls, message: str) -> str:
        text = (message or "").strip().lower()
        if MedicalLlmService.emergency_signals(text):
            return "emergency_guidance"
        if any(keyword in text for keyword in cls.RESERVATION_KEYWORDS):
            return "hospital_reservation"
        if any(keyword in text for keyword in cls.MEDIUM_RISK_KEYWORDS):
            return "medical_advice"
        return "general_chat"

    @classmethod
    def infer_risk(cls, message: str, intent: str) -> tuple[str, str, bool]:
        text = (message or "").strip().lower()
        if intent == "emergency_guidance" or any(keyword in text for keyword in cls.HIGH_RISK_KEYWORDS):
            return "high", "hospital_visit", True
        if intent in {"hospital_reservation", "medical_advice"} or any(keyword in text for keyword in cls.MEDIUM_RISK_KEYWORDS):
            return "medium", "guardian_contact", intent == "hospital_reservation"
        return "low", "home_monitoring", False

    @staticmethod
    def _department_question_reply(message: str, history: list[dict[str, str]]) -> dict | None:
        """"무슨 과로 가야 해?"에는 직전 상담에서 권한 진료과를 한 줄로 답한다(같은 상담 답변을 반복하지 않게)."""
        if not re.search(r"(무슨|어느|어떤)\s*과", message or ""):
            return None
        last_assistant = next((h.get("content", "") for h in reversed(history) if h.get("role") == "assistant"), "")
        department = _find_department(last_assistant)
        if not department:
            return None
        return {
            "reply": f"{department}로 가시면 돼요. 가까운 {department}를 찾아드릴까요? \"병원 찾아줘\"라고 말씀해 주세요.",
            "riskLevel": "medium",
            "recommendedAction": "guardian_contact",
            "reservationRequired": False,
            "intent": "medical_advice",
            "engine": "rule",
            "modelName": None,
            "summary": None,
            "possibleCauses": [],
            "homeCare": [],
            "visitHospitalIf": [],
            "emergencyWarning": [],
        }

    def _fallback_reply(
        self, message: str, history: list[dict[str, str]], emergency: bool = False, history_basis: bool = True
    ) -> dict:
        """생성 실패 폴백. history_basis=False(위기 뒤 해제된 턴)면 앞 턴 자해 키워드로 위기 응답을 되살리지 않는다 -
        응급이면 일반 119 폴백. 이번 문장의 자해 키워드는 그대로 위기 응답이다."""
        trigger = "keyword" if MedicalLlmService.self_harm_signals(message) else (
            "history" if history_basis and emergency and MedicalLlmService.self_harm_in_history(history) else None
        )
        if trigger:
            family = MedicalLlmService.about_family(message)
            _LOG.info("[CHAT-SAFETY] self_harm=true path=fallback trigger=%s family=%s", trigger, family)
            return MedicalLlmService.self_harm_response(family=family, trigger=trigger)
        intent = self.classify_intent(message)
        if emergency:
            # 라우터가 응급으로 본 문장은 키워드에 안 걸려도 119 안내가 빠지면 안 된다.
            intent = "emergency_guidance"
        risk_level, action, reservation_required = self.infer_risk(message, intent)
        prior_user = [item["content"] for item in history if item.get("role") == "user"][-2:]
        context_hint = ""
        if prior_user:
            context_hint = f" 이전에 말씀하신 내용({prior_user[0][:30]})도 함께 고려해 주세요."

        if risk_level == "high":
            reply = (
                "현재 증상은 즉시 확인이 필요할 수 있습니다. 119 또는 가까운 응급실 방문을 권장합니다."
                f"{context_hint} "
                "이 답변은 일반적인 건강 정보 안내이며, 정확한 진단과 치료는 의료진 상담이 필요합니다."
            )
        elif risk_level == "medium":
            reply = (
                # 보호자·피보호자 모두에게 쓰이는 문구다 - "보호자"로 못 박으면 보호자 본인이나 연결된 보호자가 없는
                # 피보호자에게 맞지 않는다(백엔드 점검 L-4, 2026-10-09).
                "증상이 지속되면 진료가 필요할 수 있습니다. 가족이나 가까운 분과 상태를 공유하고 경과를 관찰해 주세요."
                f"{context_hint} "
                "이 답변은 일반적인 건강 정보 안내이며, 정확한 진단과 치료는 의료진 상담이 필요합니다."
            )
        else:
            reply = (
                f"전달해 주신 내용을 기준으로 우선 안정 상태로 보입니다.{context_hint} "
                "이상 징후가 있으면 증상이 언제부터 시작됐는지, 다른 동반 증상이 있는지 알려주세요. "
                "이 답변은 일반적인 건강 정보 안내이며, 정확한 진단과 치료는 의료진 상담이 필요합니다."
            )

        return {
            "reply": reply,
            "riskLevel": risk_level,
            "recommendedAction": action,
            "reservationRequired": reservation_required,
            "intent": intent,
            "engine": "fallback",
            "modelName": None,
            "summary": None,
            "possibleCauses": [],
            "homeCare": [],
            "visitHospitalIf": [],
            # MedGemma 응급 경로와 같은 경고 박스 - 모델이 실패해도 응급 안내가 빨간 박스에서 빠지지 않게(#33).
            "emergencyWarning": [MedicalLlmService.EMERGENCY_WARNING] if intent == "emergency_guidance" else [],
        }

    @staticmethod
    def delete_logs(db: Session, user_id: str) -> tuple[int, int]:
        """해당 회원의 상담 기록을 모두 지운다(회원 탈퇴 정리용). (본인 기록 수, 대리 상담 기록 수)를 돌려준다.
        보호자가 이 회원(피보호자)을 대신해 상담한 기록(보호자 ID로 저장, userContext.wardId == 이 회원)도 지운다 -
        그 기록에 피보호자 이름·전화·생년월일·예약 초안 환자 정보가 들어 있다(2026-10-30 탈퇴 정리 점검 M-1, 기록째 삭제는 사용자 결정).
        그 밖의 다른 회원 기록은 건드리지 않는다."""
        own = db.query(ChatLog).filter(ChatLog.user_id == user_id).delete(synchronize_session=False)
        proxy_ids = [
            log.id
            for log in db.query(ChatLog).filter(ChatLog.context_json.contains(user_id)).all()
            if user_id in ChatService._proxy_ward_ids(log.context_json)
        ]
        proxy = db.query(ChatLog).filter(ChatLog.id.in_(proxy_ids)).delete(synchronize_session=False) if proxy_ids else 0
        db.commit()
        return int(own), int(proxy)

    @staticmethod
    def _proxy_ward_ids(context_json: str | None) -> set[str]:
        """기록이 대리 상담한 피보호자 ID(router.wardId·userContext.wardId). 형식이 이상하면 빈 집합."""
        try:
            ctx = json.loads(context_json or "{}")
        except ValueError:
            return set()
        if not isinstance(ctx, dict):
            return set()
        meta = ctx.get("upstreamMeta") if isinstance(ctx.get("upstreamMeta"), dict) else {}
        router = meta.get("router") if isinstance(meta.get("router"), dict) else {}
        user_context = ctx.get("userContext") if isinstance(ctx.get("userContext"), dict) else {}
        return {str(v) for v in (router.get("wardId"), user_context.get("wardId")) if v}

    @staticmethod
    def list_logs(db: Session, user_id: str | None = None) -> list[dict]:
        query = db.query(ChatLog)
        if user_id is not None:
            query = query.filter(ChatLog.user_id == user_id)
        logs = query.order_by(ChatLog.id.desc()).all()
        items: list[dict] = []
        for log in logs:
            context = {}
            try:
                context = json.loads(log.context_json or "{}")
            except json.JSONDecodeError:
                context = {}
            items.append(
                {
                    "id": log.id,
                    "chatNo": log.chat_no,
                    "userId": log.user_id,
                    "message": log.message,
                    "contextJson": log.context_json,
                    "sessionId": context.get("sessionId"),
                    "engine": context.get("engine"),
                    "intent": context.get("conversationState", {}).get("intent") if isinstance(context.get("conversationState"), dict) else context.get("intent"),
                    "modelName": context.get("modelName"),
                    "type": context.get("type"),
                    "tool": context.get("tool"),
                    "toolData": context.get("data"),
                    "ui": context.get("ui"),
                    "reply": log.reply,
                    "riskLevel": log.risk_level,
                    "recommendedAction": log.recommended_action,
                    "reservationRequired": log.reservation_required,
                    # 기록을 다시 열 때 빨간 경고 박스를 복원하려고 내보낸다(BE 기록 허용 목록 emergencyWarning 과 짝).
                    "emergencyWarning": context.get("emergencyWarning") or [],
                    "decisionTrace": context.get("decisionTrace") or [],
                    "upstreamMeta": context.get("upstreamMeta") or {},
                    "createdAt": log.created_at.isoformat(),
                },
            )
        return items

    @staticmethod
    def get_log(db: Session, chat_id: int, user_id: str | None = None) -> dict | None:
        log = db.query(ChatLog).filter(ChatLog.id == chat_id).first()
        if not log:
            return None
        if user_id is not None and log.user_id != user_id:
            return None
        context = {}
        try:
            context = json.loads(log.context_json or "{}")
        except json.JSONDecodeError:
            context = {}
        return {
            "id": log.id,
            "chatNo": log.chat_no,
            "userId": log.user_id,
            "message": log.message,
            "contextJson": context,
            "sessionId": context.get("sessionId"),
            "engine": context.get("engine"),
            "intent": context.get("conversationState", {}).get("intent") if isinstance(context.get("conversationState"), dict) else context.get("intent"),
            "modelName": context.get("modelName"),
            "type": context.get("type"),
            "tool": context.get("tool"),
            "toolData": context.get("data"),
            "ui": context.get("ui"),
            "reply": log.reply,
            "riskLevel": log.risk_level,
            "recommendedAction": log.recommended_action,
            "reservationRequired": log.reservation_required,
            "emergencyWarning": context.get("emergencyWarning") or [],
            "decisionTrace": context.get("decisionTrace") or [],
            "upstreamMeta": context.get("upstreamMeta") or {},
            "createdAt": log.created_at.isoformat(),
        }
