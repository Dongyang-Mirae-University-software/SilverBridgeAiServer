from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _as_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _opt_float(name: str) -> float | None:
    """값이 없거나 숫자가 아니면 None (쓰는 곳에서 다른 설정에 맞춰 기본값을 정한다). 숫자가 아니면 경고한다 (값은 남기지 않음)."""
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else None
    except ValueError:
        logging.getLogger("app.config").warning("[CONFIG] %s 값이 숫자가 아니어서 기본값을 씁니다", name)
        return None


def _opt_int(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    try:
        return int(raw) if raw else None
    except ValueError:
        logging.getLogger("app.config").warning("[CONFIG] %s 값이 정수가 아니어서 기본값을 씁니다", name)
        return None


def _as_choice(name: str, default: str, choices: tuple[str, ...]) -> str:
    """선택형 설정. 앞뒤 공백·대소문자는 무시하고, 모르는 값이면 경고를 남기고 기본값을 쓴다 (값 자체는 로그에 남기지 않는다)."""
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    if raw in choices:
        return raw
    logging.getLogger("app.config").warning("[CONFIG] %s 값을 알 수 없어 기본값 %s 을(를) 씁니다 (허용: %s)",
                                            name, default, "|".join(choices))
    return default


def _as_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logging.getLogger("app.config").warning("[CONFIG] %s 값이 정수가 아니어서 기본값을 씁니다", name)
        return default


def _as_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logging.getLogger("app.config").warning("[CONFIG] %s 값이 숫자가 아니어서 기본값을 씁니다", name)
        return default


def _as_float_in_range(name: str, default: float, low: float, high: float) -> float:
    """범위 밖(또는 숫자가 아닌) 값은 기본값으로 대체한다 - 잘못 적은 환경변수가 판정을 무력화하지 않게."""
    value = _as_float(name, default)
    return value if low <= value <= high else default


def _first_env(names: tuple[str, ...], default: str) -> str:
    """앞에서부터 비어 있지 않은 첫 환경변수 값, 모두 비면 default."""
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return default


def _as_int_in_range(name: str, default: int, low: int, high: int) -> int:
    """범위 밖(또는 정수가 아닌) 값은 기본값으로 대체한다."""
    value = _as_int(name, default)
    return value if low <= value <= high else default


def _as_int_min(name: str, default: int, low: int) -> int:
    """low 미만(또는 정수가 아닌) 값은 경고를 남기고 기본값으로 대체한다 (값은 로그에 남기지 않는다)."""
    value = _as_int(name, default)
    if value < low:
        logging.getLogger("app.config").warning("[CONFIG] %s 값이 %d 미만이어서 기본값을 씁니다", name, low)
        return default
    return value


def _as_int_capped(name: str, default: int, cap: int) -> int:
    """cap 을 넘는 값은 cap 으로 자르고, 0 이하(또는 정수가 아닌) 값은 기본값으로 대체한다."""
    value = _as_int(name, default)
    if value <= 0:
        return default
    return min(value, cap)


class Settings(BaseModel):
    app_name: str = os.getenv("APP_NAME", "SilverBridge AI Server")
    app_env: str = os.getenv("APP_ENV", "development")
    app_version: str = os.getenv("APP_VERSION", "0.1.0")
    app_port: int = _as_int("APP_PORT", 9000)
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    database_url: str = os.getenv(
        "DATABASE_URL",
        f"postgresql+psycopg2://"
        f"{os.getenv('POSTGRES_USER', 'silverbridge_ai')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'silverbridge_ai_pw')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{_as_int('POSTGRES_PORT', 5432)}/"
        f"{os.getenv('POSTGRES_DB', 'silverbridge_ai')}",
    )
    # 기본값 없음 — 비어 있으면 인증이 필요한 API 는 모두 401 이다(키를 코드에 두지 않는다, QA AI-2).
    api_key: str = os.getenv("API_KEY", "")

    model_base_path: str = os.getenv("MODEL_BASE_PATH", str(_PROJECT_ROOT / "models"))
    upload_base_path: str = os.getenv("UPLOAD_BASE_PATH", str(_PROJECT_ROOT / "uploads"))
    snapshot_base_path: str = os.getenv(
        "SNAPSHOT_BASE_PATH",
        str(_PROJECT_ROOT / "uploads" / "snapshots"),
    )
    default_detection_threshold: float = _as_float("DEFAULT_DETECTION_THRESHOLD", 0.75)
    fire_smoke_enabled: bool = _as_bool("FIRE_SMOKE_ENABLED", True)
    fire_smoke_model_path: str = os.getenv("FIRE_SMOKE_MODEL_PATH", "fire_smoke.pt")
    fire_smoke_conf_threshold: float = _as_float("FIRE_SMOKE_CONF_THRESHOLD", 0.35)
    fire_smoke_iou_threshold: float = _as_float("FIRE_SMOKE_IOU_THRESHOLD", 0.45)
    # 위험 판정(danger) 임계. 추론 필터(conf 0.35)는 "박스를 그릴 후보"를 고르는 값이라
    # 그대로 알람 기준으로 쓰면 오탐이 그대로 보호자 알림이 된다 → 판정용을 따로 둔다.
    fire_smoke_danger_threshold: float = _as_float("FIRE_SMOKE_DANGER_THRESHOLD", 0.6)
    # 흉기·낙상 — 라이브 분석에 화재와 함께 돈다. 기본 OFF: 모델 교체 후 환경변수로 켠다.
    knife_enabled: bool = _as_bool("KNIFE_ENABLED", False)
    knife_model_path: str = os.getenv("KNIFE_MODEL_PATH", "knife.pt")
    knife_conf_threshold: float = _as_float("KNIFE_CONF_THRESHOLD", 0.25)
    knife_iou_threshold: float = _as_float("KNIFE_IOU_THRESHOLD", 0.45)
    knife_danger_threshold: float = _as_float("KNIFE_DANGER_THRESHOLD", 0.5)
    fall_enabled: bool = _as_bool("FALL_ENABLED", False)
    fall_model_path: str = os.getenv("FALL_MODEL_PATH", "fall.pt")
    fall_conf_threshold: float = _as_float("FALL_CONF_THRESHOLD", 0.25)
    fall_iou_threshold: float = _as_float("FALL_IOU_THRESHOLD", 0.45)
    # 낙상 danger 는 한 장 판정이 아니다: 최근 FALL_HOLD_SEC 초 동안 분석된 장면 중 FALL_HOLD_RATIO 이상에서
    # 점수가 FALL_DANGER_THRESHOLD 이상이어야 한다(임계 0.4 는 검토 중 — 환경변수로 조정).
    fall_danger_threshold: float = _as_float("FALL_DANGER_THRESHOLD", 0.4)
    fall_hold_sec: float = _as_float("FALL_HOLD_SEC", 1.5)
    fall_hold_ratio: float = _as_float("FALL_HOLD_RATIO", 0.7)
    # 화면 면적 대비 이 비율 이상인 fallen 상자는 낙상 판정에서 무시한다(0 이하 = 끔). 화면 공유·TV 영상처럼 화면
    # 대부분을 덮는 상자가 낮은 점수로 유지 조건을 채우는 오탐을 막는다. 방 카메라 실측 최대는 31%(99%가 24% 이하).
    fall_max_box_area: float = _as_float("FALL_MAX_BOX_AREA", 0.35)
    # 라이브 추론 시간 요약 로그 주기(초, 0 이하 = 끔). 감지기를 여러 개 켜면 프레임당 추론이 길어진다 —
    # 너무 느리면 STREAM_SAMPLE_EVERY_N_FRAMES 를 늘린다.
    stream_infer_log_interval_sec: float = _as_float("STREAM_INFER_LOG_INTERVAL_SEC", 60.0)

    default_chat_model: str = os.getenv("DEFAULT_CHAT_MODEL", "google/medgemma-1.5-4b-it")
    chat_model_path: str = os.getenv("CHAT_MODEL_PATH", "")
    hf_token: str = os.getenv("HF_TOKEN", "")
    chat_enable_llm: bool = _as_bool("CHAT_ENABLE_LLM", True)
    # 라우터·예약 추출용 Claude. 키는 .env의 ANTHROPIC_API_KEY.
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY") or ""
    claude_model_name: str = os.getenv("CLAUDE_MODEL_NAME", "claude-opus-5-5")
    # 용도별 모델(2026-10-24 O-2). 비면 CLAUDE_MODEL_NAME, 그것도 비면 claude-opus-5-5 - 두 줄을 지우면 원래대로.
    chat_router_model: str = _first_env(("CHAT_ROUTER_MODEL", "CLAUDE_MODEL_NAME"), "claude-opus-5-5")
    reservation_extract_model: str = _first_env(("RESERVATION_EXTRACT_MODEL", "CLAUDE_MODEL_NAME"), "claude-opus-5-5")
    claude_timeout_sec: int = _as_int_min("CLAUDE_TIMEOUT_SEC", 60, 1)
    # 라우터·예약 추출 Claude 호출의 출력 상한(생각 토큰 포함, 2026-10-25). MedGemma MAX_NEW_TOKENS(512)와 따로 둔다 -
    # opus 5.5는 생각을 끌 수 없어 위기 문장에서 생각 블록이 512를 다 써 답이 잘렸다. 범위 밖 값은 기본값으로 대체한다.
    claude_max_tokens: int = _as_int_in_range("CLAUDE_MAX_TOKENS", 2048, 256, 8192)
    # 라우터·이어서 물어볼 질문·예약 추출의 고정 지시문(system)에 프롬프트 캐시 표시(5분). 같은 지시문 호출이 5분 안에 다시 오면
    # 입력 단가의 약 5%로 읽는다(첫 기록은 1.25배). 상담이 아주 드물어 적중률이 낮으면 false 로 끈다(2026-11-05).
    claude_prompt_cache: bool = _as_bool("CLAUDE_PROMPT_CACHE", True)
    # 약 봉투 OCR(POST /api/v1/medication/ocr). 키는 위 ANTHROPIC_API_KEY 를 같이 쓴다.
    # 모델은 2026-10-22 측정으로 sonnet 확정 - haiku 는 없는 약을 지어내 금지(기동 시 sonnet 으로 대체).
    medication_ocr_enabled: bool = _as_bool("MEDICATION_OCR_ENABLED", True)
    medication_ocr_model: str = os.getenv("MEDICATION_OCR_MODEL", "claude-sonnet-5-5")
    # 답변 뒤 '이어서 물어볼 질문'(응답 suggestedQuestions, 2026-10-30). 안전 판정과 무관한 보조 값이라 가벼운 모델·짧은 대기.
    # 대기를 넘기거나 실패하면 규칙 문장으로 대신한다. FOLLOW_UP_ENABLED=false 면 항상 빈 목록.
    follow_up_enabled: bool = _as_bool("FOLLOW_UP_ENABLED", True)
    follow_up_model: str = os.getenv("FOLLOW_UP_MODEL", "claude-haiku-5-5")
    follow_up_timeout_sec: float = _as_float_in_range("FOLLOW_UP_TIMEOUT_SEC", 3.0, 0.5, 10.0)
    reservation_api_base_url: str = os.getenv(
        "RESERVATION_API_BASE_URL",
        "http://127.0.0.1:6015/api/v1",
    )
    # 예약 서비스 호출 한 번의 대기 상한(초). 정상 응답은 0~1초라 기본 10초, 설정값이 커도 15초로 자른다
    # (예전 기본 1200·서버 120이면 예약 서비스가 멈췄을 때 챗 마감 110초를 다 쓰고서야 끝났다).
    reservation_api_timeout_sec: int = _as_int_capped("RESERVATION_API_TIMEOUT_SEC", 10, 15)
    reservation_api_key: str = (
        os.getenv("RESERVATION_API_KEY", "").strip() or os.getenv("RESERVATION_API_TOKEN", "").strip()
    )
    reservation_user_jwt: str = os.getenv("RESERVATION_USER_JWT", "")
    reservation_admin_jwt: str = os.getenv("RESERVATION_ADMIN_JWT", "")
    # 사용자별 예약 계정(sb-<userId>@silverbridge.local) 비밀번호 파생 비밀. 프론트 BFF(RESERVATION_ACCOUNT_SECRET)와 같아야 한다.
    # 코드에 기본값을 두지 않는다(저장소를 보는 누구나 계정 비밀번호를 계산할 수 있다). 비어 있으면 예약 기능만 꺼진다.
    reservation_account_secret: str = os.getenv("RESERVATION_ACCOUNT_SECRET", "").strip()
    # 옛 공용 예약 계정. 기본값 없음 - 둘 다 .env에 있을 때만 쓴다.
    reservation_service_email: str = os.getenv("RESERVATION_SERVICE_EMAIL", "")
    reservation_service_password: str = os.getenv("RESERVATION_SERVICE_PASSWORD", "")
    torch_dtype: str = os.getenv("TORCH_DTYPE", "float16")
    device_map: str = os.getenv("DEVICE_MAP", "auto")
    require_gpu: bool = _as_bool("REQUIRE_GPU", False)
    max_new_tokens: int = _as_int("MAX_NEW_TOKENS", 512)
    temperature: float = _as_float("TEMPERATURE", 0.4)
    # MedGemma 끄기. false면 기동 때 올리지 않고, 요청 때도 다시 올리지 않는다(메모리 확보). 나머지 기능에는 영향 없다.
    medgemma_enabled: bool = _as_bool("MEDGEMMA_ENABLED", True)
    top_p: float = _as_float("TOP_P", 0.9)
    repetition_penalty: float = _as_float("REPETITION_PENALTY", 1.1)
    do_sample: bool = _as_bool("DO_SAMPLE", True)
    load_in_8bit: bool = _as_bool("LOAD_IN_8BIT", False)
    # MedGemma 가중치 양자화. none(기본, bf16 그대로) | nf4(bitsandbytes 4비트) | int8(bitsandbytes 8비트). 모르는 값은 none으로 본다.
    medgemma_quant: str = os.getenv("MEDGEMMA_QUANT", "none").strip().lower()
    # MedGemma 구조화(JSON) 생성 전용 상한(2026-10-26). 비면 MAX_NEW_TOKENS, 64~2048 밖이면 MAX_NEW_TOKENS로 대체한다.
    medgemma_json_max_tokens: int = _as_int_in_range("MEDGEMMA_JSON_MAX_TOKENS", max_new_tokens, 64, 2048)
    # 구조화 생성 반복 억제. off(기본, 지금과 같음) | gen_ngram:<2~20>(생성 구간만) | penalty:<1.0~1.5> | ngram:<2~20>. 모르는 값은 off.
    medgemma_json_repetition: str = os.getenv("MEDGEMMA_JSON_REPETITION", "off").strip().lower()
    # MedGemma 동시 생성 수(2026-10-27). 생성이 겹치면 같은 GPU의 감지(화재·흉기·낙상) 처리 시간이 2배가 된다. 0 이하면 제한 없음.
    medgemma_max_concurrent_generate: int = _as_int("MEDGEMMA_MAX_CONCURRENT_GENERATE", 1)
    # 생성 슬롯 대기 상한(초). 넘기면 생성하지 않고 고정 문구로 답한다 - BE 상담 제한 130초 안에 끝나게.
    # 생성 자체의 시간 상한(초, HF generate max_time). 슬롯 대기와 별개 (2차 점검 R-3)
    medgemma_generate_max_time_sec: float = _as_float("MEDGEMMA_GENERATE_MAX_TIME_SEC", 90.0)
    # 실시간 분석 최신성·WS 송신 제한·끊긴 세션 정리 (2차 점검 R-2 등)
    live_latest_analysis_max_age_sec: float = _as_float("LIVE_LATEST_ANALYSIS_MAX_AGE_SEC", 30.0)
    live_ws_send_timeout_sec: float = _as_float("LIVE_WS_SEND_TIMEOUT_SEC", 5.0)
    live_stale_session_purge_sec: float = _as_float("LIVE_STALE_SESSION_PURGE_SEC", 1800.0)
    # DB 연결·풀 대기 제한(초, PostgreSQL 만 적용)
    db_connect_timeout_sec: int = _as_int("DB_CONNECT_TIMEOUT_SEC", 5)
    db_pool_timeout_sec: int = _as_int("DB_POOL_TIMEOUT_SEC", 10)
    medgemma_gen_wait_timeout_sec: int = _as_int_in_range("MEDGEMMA_GEN_WAIT_TIMEOUT_SEC", 60, 1, 120)
    # 응급 턴은 줄을 오래 서지 않는다 - 넘기면 바로 119 안내 고정 문구로 답한다.
    medgemma_gen_wait_timeout_emergency_sec: int = _as_int_in_range("MEDGEMMA_GEN_WAIT_TIMEOUT_EMERGENCY_SEC", 3, 1, 30)
    chat_history_keep_turns: int = _as_int("CHAT_HISTORY_KEEP_TURNS", 8)
    # 답 줄 머리 이모지(2026-11-01 가독성). 2026-11-09 기본 끔 - 어르신 화면에서 문장만 보이게(응급 🚨도 끈다, 핵심 줄은 119 전화 문장).
    # 판정은 원래 이모지 없는 문장 기준이라 이 값과 무관하다. 다시 켜려면 CHAT_LAYOUT_EMOJI=true.
    chat_layout_emoji: bool = _as_bool("CHAT_LAYOUT_EMOJI", False)
    # Claude(라우터) 호출이 실패한 턴의 답 끝에 안내 한 줄(2026-11-11). 응급·위기·되묻기 질문 턴에는 붙이지 않는다. 끄려면 false.
    chat_degraded_notice: bool = _as_bool("CHAT_DEGRADED_NOTICE", True)
    # 라우터 실패 누적 알림(2026-11-08): 창(초) 안에서 실패가 기준 횟수에 닿으면 ERROR 한 번(창마다 1회).
    # 크레딧 소진·API 장애처럼 실패가 이어지는 상황을 사람이 알게 한다. 코드 기본값으로 동작(.env 불필요).
    chat_router_failure_alert_window_sec: int = _as_int_in_range("CHAT_ROUTER_FAILURE_ALERT_WINDOW_SEC", 600, 60, 86400)
    chat_router_failure_alert_threshold: int = _as_int_in_range("CHAT_ROUTER_FAILURE_ALERT_THRESHOLD", 5, 1, 1000)
    # 크레딧·결제 오류는 처음 생길 때 바로 ERROR 한 번, 이어지면 이 간격(초)마다 1회. 라우터가 다시 성공하면 풀린다.
    chat_router_credit_alert_repeat_sec: int = _as_int_in_range("CHAT_ROUTER_CREDIT_ALERT_REPEAT_SEC", 3600, 60, 86400)
    # 챗 요청 하나의 전체 마감(초, 2026-10-28 O-3). BE 중계 전체 제한 130초에 응답 전송 여유 20초를 둔 값.
    # 각 호출(Claude·예약 API·생성 대기)은 남은 시간만큼만 기다리고, 모자라면 새 호출을 시작하지 않는다.
    chat_deadline_sec: int = _as_int_in_range("CHAT_DEADLINE_SEC", 110, 30, 125)
    # 되돌릴 수 없는 호출(예약 생성·취소)을 시작하는 최소 남은 시간(초). 모자라면 확인 카드를 다시 보여 준다.
    chat_irreversible_min_remaining_sec: int = _as_int_in_range("CHAT_IRREVERSIBLE_MIN_REMAINING_SEC", 25, 5, 60)
    # MedGemma 생성을 시작하는 최소 남은 시간(초). 생성은 중간에 끊을 수 없어 이만큼 남았을 때만 슬롯을 잡는다.
    chat_gen_min_remaining_sec: int = _as_int_in_range("CHAT_GEN_MIN_REMAINING_SEC", 30, 5, 60)

    def resolved_chat_model_id(self) -> str:
        path = (self.chat_model_path or "").strip()
        if path:
            return path
        return self.default_chat_model

    @property
    def has_reservation_account_secret(self) -> bool:
        return bool(self.reservation_account_secret.strip())

    @property
    def has_reservation_service_credentials(self) -> bool:
        return bool(self.reservation_service_email.strip() and self.reservation_service_password.strip())

    # 문서 경로(/api/docs·/api/openapi.json·/api/redoc) 노출. false 면 세 경로가 404.
    docs_enabled: bool = _as_bool("DOCS_ENABLED", True)
    # 옛 카메라 등록·주소 접속 분석 API(/api/v1/cameras·/api/v1/analysis). 쓰는 곳이 없어 기본은 끈다(전체 점검 M-8)
    legacy_camera_api_enabled: bool = _as_bool("LEGACY_CAMERA_API_ENABLED", False)
    # 회원 데이터 삭제(탈퇴·연결 종료) 뒤 같은 ID 로 상담 기록·예약 계정을 새로 만들지 않는 시간(초). 목록은 프로세스 메모리라
    # 재시작하면 비워진다 - 워커 1개 기준(전체 점검 L-1)
    member_purge_hold_sec: float = _as_float("MEMBER_PURGE_HOLD_SEC", 150.0)
    docs_path: str = os.getenv("DOCS_PATH", "/api/docs")
    openapi_path: str = os.getenv("OPENAPI_PATH", "/api/openapi.json")
    redoc_path: str = os.getenv("REDOC_PATH", "/api/redoc")

    stream_frame_max_bytes: int = _as_int("STREAM_FRAME_MAX_BYTES", 4 * 1024 * 1024)  # 프레임 업로드 바이트 상한 (초과 413 STREAM_FRAME_TOO_LARGE)
    stream_frame_max_side: int = _as_int("STREAM_FRAME_MAX_SIDE", 4096)  # 프레임 한 변 픽셀 상한 (JPEG 헤더로 검사)
    stream_sample_every_n_frames: int = _as_int("STREAM_SAMPLE_EVERY_N_FRAMES", 15)
    stream_fallback_interval_sec: int = _as_int("STREAM_FALLBACK_INTERVAL_SEC", 2)
    save_normal_results: bool = _as_bool("SAVE_NORMAL_RESULTS", True)
    live_stream_disconnect_timeout_sec: int = _as_int("LIVE_STREAM_DISCONNECT_TIMEOUT_SEC", 10)
    stream_state_backend: str = os.getenv("STREAM_STATE_BACKEND", "memory")
    mediamtx_enabled: bool = _as_bool("MEDIAMTX_ENABLED", False)
    mediamtx_webrtc_ingest_base: str = os.getenv("MEDIAMTX_WEBRTC_INGEST_BASE", "")
    mediamtx_webrtc_view_base: str = os.getenv("MEDIAMTX_WEBRTC_VIEW_BASE", "")
    mediamtx_hls_view_base: str = os.getenv("MEDIAMTX_HLS_VIEW_BASE", "")
    # WS 의 ?apiKey= 쿼리 인증 허용 여부. 쿼리 키는 브라우저 번들·접속 로그에 남는다(QA AI-2).
    # 기본은 거부(닫힘). 쿼리 접속이 필요한 환경만 WS_ALLOW_QUERY_API_KEY=true 로 명시적으로 연다.
    ws_allow_query_api_key: bool = _as_bool("WS_ALLOW_QUERY_API_KEY", False)
    # 챗 로그 상세(GET /api/v1/chat/logs/{id})에 userId 필수 + 소유자 일치. 기본 ON(닫힘) —
    # BE 는 항상 userId 를 붙인다. userId 없는 호출이 필요한 환경만 CHAT_REQUIRE_USER_ID=false 로 명시적으로 끈다.
    chat_require_user_id: bool = _as_bool("CHAT_REQUIRE_USER_ID", True)
    # 게임 API(/api/v1/games/**)에 X-API-Key 필수. 기본 ON(닫힘). 키 없이 부르는 환경만
    # GAME_REQUIRE_API_KEY=false 로 명시적으로 끈다(/embed·/state·/answer·/reset 은 원래 키 대상이 아님).
    game_require_api_key: bool = _as_bool("GAME_REQUIRE_API_KEY", True)

    # 이상감지 클립(POST /api/v1/live-streams/{sessionId}/clips). 영구 저장은 백엔드가 한다 — 여기선 임시 파일만.
    # 동시 상한·대기열은 방어선이다: FE 프록시를 거치면 이 API도 외부에서 호출될 수 있다.
    clip_enabled: bool = _as_bool("CLIP_ENABLED", True)
    clip_buffer_seconds: float = _as_float("CLIP_BUFFER_SECONDS", 10.0)
    clip_max_frames: int = _as_int("CLIP_MAX_FRAMES", 100)
    # 링버퍼 바이트 상한(0 이하 = 끔). 장수 상한만으로는 큰 프레임이 오면 메모리가 무한정 는다.
    # 세션 64MB 는 FHD JPEG 5초 클립(약 50장)을 넉넉히 담고, 전체 512MB 를 넘으면 큰 세션부터 줄인다.
    clip_session_max_bytes: int = _as_int("CLIP_SESSION_MAX_BYTES", 64 * 1024 * 1024)
    clip_total_max_bytes: int = _as_int("CLIP_TOTAL_MAX_BYTES", 512 * 1024 * 1024)
    clip_min_frames: int = _as_int("CLIP_MIN_FRAMES", 2)
    clip_bitrate: str = os.getenv("CLIP_BITRATE", "3M")
    clip_max_concurrency: int = _as_int("CLIP_MAX_CONCURRENCY", 2)
    clip_queue_max: int = _as_int("CLIP_QUEUE_MAX", 10)
    clip_timeout_sec: float = _as_float("CLIP_TIMEOUT_SEC", 20.0)
    clip_tmp_dir: str = os.getenv("CLIP_TMP_DIR", "")

    # 화재·연기 알림용 화면 품질 검사. 초점 나간·하얗게 날아간·가려진 화면은 분석 불가로 보고 화재 danger 를 내지 않는다
    # (흉기·낙상은 대상 아님). 기준값은 10/12 실측(analysis/frame_gate_calib.json): 불량 규칙 =
    # white > WHITE_RATIO 또는 black > BLACK_RATIO 또는 std < STD_MIN 또는 (lap < LAP_MAX 그리고 std < BLUR_STD_MAX).
    # 킬 스위치 FRAME_QUALITY_GATE_ENABLED=false 면 기존 동작과 같다. 범위 밖 값은 기본값으로 대체한다.
    # 2026-11-11 완화(A-4 후속, DGX analysis/frame_quality_a4): 가상 장면에서 품질 검사가 막은 오탐 58장보다 지운 화재가
    # 492장으로 많았다. black 0.95→0.98, std 12→5, blur_flat 끔(BLUR_STD_MAX=0 - std<0 은 없어 규칙이 꺼진다),
    # 화재 최고 점수가 BYPASS_CONF 이상이면 품질 불량이어도 지우지 않는다(1.01 = 우회 끔, 점수는 1.0 을 넘지 않는다).
    # 옛 동작: BLACK_RATIO=0.95 STD_MIN=12 BLUR_STD_MAX=30 BYPASS_CONF=1.01.
    frame_quality_gate_enabled: bool = _as_bool("FRAME_QUALITY_GATE_ENABLED", True)
    frame_quality_white_ratio: float = _as_float_in_range("FRAME_QUALITY_WHITE_RATIO", 0.5, 0.05, 1.0)
    frame_quality_black_ratio: float = _as_float_in_range("FRAME_QUALITY_BLACK_RATIO", 0.98, 0.05, 1.0)
    frame_quality_std_min: float = _as_float_in_range("FRAME_QUALITY_STD_MIN", 5.0, 0.0, 60.0)
    frame_quality_lap_max: float = _as_float_in_range("FRAME_QUALITY_LAP_MAX", 10.0, 0.0, 500.0)
    frame_quality_blur_std_max: float = _as_float_in_range("FRAME_QUALITY_BLUR_STD_MAX", 0.0, 0.0, 100.0)
    frame_quality_bypass_conf: float = _as_float_in_range("FRAME_QUALITY_BYPASS_CONF", 0.6, 0.0, 1.01)

    # 화재·연기 유지 조건: 한 장짜리 양성(흔들림·철판 같은 순간적 오탐)으로는 알리지 않는다. 최근 FIRE_HOLD_SEC 초 동안의
    # 분석 중 FIRE_HOLD_RATIO 이상이 양성(한 장 기준 danger)이고, 그 구간에 분석이 FIRE_HOLD_MIN_SCENES 회 이상 있으며,
    # 연속 관찰이 FIRE_HOLD_SEC 초 이상일 때만 danger 다(알림이 대략 SEC x RATIO 초 늦다).
    # 시간 기준이다: 분석은 정상 상태에서 초당 약 10회라 횟수 기준은 0.3초에 불과했다. 분석이 느려져 창 안에 MIN_SCENES 회가
    # 안 들어오면 최근 MIN_SCENES 회를 본다. FIRE_HOLD_ENABLED=false 면 한 장 판정 그대로(킬 스위치).
    # 마지막 분석 후 FIRE_HOLD_MAX_GAP_SEC 가 지나면 다시 센다.
    fire_hold_enabled: bool = _as_bool("FIRE_HOLD_ENABLED", True)
    fire_hold_sec: float = _as_float_in_range("FIRE_HOLD_SEC", 2.0, 0.5, 30.0)
    fire_hold_ratio: float = _as_float_in_range("FIRE_HOLD_RATIO", 0.6, 0.3, 1.0)
    fire_hold_min_scenes: int = _as_int_in_range("FIRE_HOLD_MIN_SCENES", 3, 1, 20)
    fire_hold_max_gap_sec: float = _as_float_in_range("FIRE_HOLD_MAX_GAP_SEC", 10.0, 1.0, 120.0)
    # 흉기 유지 조건: 화재와 같은 구조(한 장짜리 양성으로는 알리지 않는다). KNIFE_HOLD_ENABLED=false 면 한 장 판정 그대로(킬 스위치).
    knife_hold_enabled: bool = _as_bool("KNIFE_HOLD_ENABLED", True)
    knife_hold_sec: float = _as_float_in_range("KNIFE_HOLD_SEC", 2.0, 0.5, 30.0)
    knife_hold_ratio: float = _as_float_in_range("KNIFE_HOLD_RATIO", 0.6, 0.3, 1.0)
    knife_hold_min_scenes: int = _as_int_in_range("KNIFE_HOLD_MIN_SCENES", 3, 1, 20)
    knife_hold_max_gap_sec: float = _as_float_in_range("KNIFE_HOLD_MAX_GAP_SEC", 10.0, 1.0, 120.0)

    # 실시간 영상(MJPEG·latest-frame)의 감지 박스. 보여주는 복사본에만 그리고 원본·분석 입력·클립은 건드리지 않는다.
    # LIVE_DRAW_BBOX=false 면 원본 바이트를 그대로 내보낸다(비상 스위치). 박스 표시 기준은 기본 "danger" =
    # 종류별 위험 기준(*_DANGER_THRESHOLD)이라 박스가 보이면 알림 기준을 넘은 것이다(낙상은 유지 조건 별도).
    # 숫자(0~1)를 주면 모든 종류에 그 값을 쓰는 재정의(시연·디버깅용).
    live_draw_bbox: bool = _as_bool("LIVE_DRAW_BBOX", True)
    live_bbox_hold_seconds: float = _as_float("LIVE_BBOX_HOLD_SECONDS", 1.0)
    live_bbox_min_confidence: str = os.getenv("LIVE_BBOX_MIN_CONFIDENCE", "danger").strip() or "danger"
    live_bbox_draw_timeout_ms: int = _as_int("LIVE_BBOX_DRAW_TIMEOUT_MS", 150)

    # 학습용 프레임 수집(집 안 영상이다). 기본 꺼짐이고, 켜도 COLLECT_SESSION_IDS 에 적은 세션만 저장한다(비면 0).
    # 실제 사용자 카메라는 넣지 말 것 - 허용한 팀 시험 세션만. 내려받는 API 는 없다(서버에서 직접 옮긴다).
    collect_enabled: bool = _as_bool("COLLECT_ENABLED", False)
    collect_session_ids: str = os.getenv("COLLECT_SESSION_IDS", "")
    collect_dir: str = os.getenv("COLLECT_DIR", "").strip() or "/app/data/collect"
    collect_min_confidence: float = _as_float("COLLECT_MIN_CONFIDENCE", 0.2)
    collect_daily_max_per_session: int = _as_int("COLLECT_DAILY_MAX_PER_SESSION", 2000)
    collect_max_total_gb: float = _as_float("COLLECT_MAX_TOTAL_GB", 20.0)
    collect_min_free_gb: float = _as_float("COLLECT_MIN_FREE_GB", 5.0)
    collect_retention_days: int = _as_int("COLLECT_RETENTION_DAYS", 14)
    # 무작위 장면 수집(모델 비교용). 점수·위험과 무관하게 세션별로 간격마다 1장(±30% 흔들림). 간격 0 = 끔.
    # 대상은 COLLECT_RANDOM_SESSION_IDS 의 정확한 세션 ID 만(와일드카드 거부). 비면 저장 0. COLLECT_ENABLED 도 켜져 있어야 한다.
    collect_random_interval_sec: float = _as_float("COLLECT_RANDOM_INTERVAL_SEC", 0.0)
    collect_random_session_ids: str = os.getenv("COLLECT_RANDOM_SESSION_IDS", "")
    collect_random_daily_max: int = _as_int("COLLECT_RANDOM_DAILY_MAX", 300)

    # ---- 말벗·정서 분석 (companion). 결정: docs/ai-part/AI파트-가이드.html 의 D1~D24 ----
    companion_api_key: str = os.getenv("COMPANION_API_KEY", "")  # D4: 말벗 전용 키 (빈 값이면 모든 요청 거부)
    companion_session_limit_sec: int = _as_int("COMPANION_SESSION_LIMIT_SEC", 0)  # D1: 0 = 시간 제한 없음
    companion_max_turns: int = _as_int("COMPANION_MAX_TURNS", 8)  # D1: 답변 8번이면 종료
    companion_frame_interval_sec: float = _as_float("COMPANION_FRAME_INTERVAL_SEC", 2.0)  # D5 (사용자 결정 2026-11-01)
    companion_frame_max: int = _as_int("COMPANION_FRAME_MAX", 5)  # D5 (말풍선당 최대 5장)
    companion_face_min_rate: float = _as_float("COMPANION_FACE_MIN_RATE", 0.5)  # D6
    companion_confidence_min: float = _as_float("COMPANION_CONFIDENCE_MIN", 0.6)  # 확신도 0.6 기준
    # 발화 최소 글자 수. 비우면 기분 엔진에 맞춘다: e900 = 2자(짧은 대답도 판단, 11/10) · ggara02 = 10자(재희)
    companion_min_text_chars: int | None = _opt_int("COMPANION_MIN_TEXT_CHARS")
    companion_upload_max_mb: int = _as_int("COMPANION_UPLOAD_MAX_MB", 5)  # BE 기준 5MB (2026-11-09 맞춤, 명세 Q12는 10MB였음)
    companion_reply_max_chars: int = _as_int("COMPANION_REPLY_MAX_CHARS", 140)  # D21 (사용자 결정 140자 이내)
    # 말·표정 합치기: prob_add(글 기분 4코드 확률 + 표정 코드 쪽으로 COMPANION_FACE_WEIGHT, 11/10 기본) | plan41(재희 fuse_turn: 방향 반대면 보류)
    # 근거: 무표정(= 평온) 상황에서 걱정·우울 찾음 plan41 0.3% · prob_add 81% (노션 "AI 안부말벗·정서분석 모델 학습 기록", fuse_cmp)
    companion_fusion_rule: str = _as_choice("COMPANION_FUSION_RULE", "prob_add", ("prob_add", "plan41"))
    companion_face_weight: float = _as_float("COMPANION_FACE_WEIGHT", 0.2)
    # 글 기분 엔진: e900(EXAONE + 기분 LoRA, 4코드 JOY·CALM·WORRY·DEPRESSED, 11/10 기본) | ggara02(재희: KcELECTRA 6감정)
    companion_text_mood_engine: str = _as_choice("COMPANION_TEXT_MOOD_ENGINE", "e900", ("e900", "ggara02"))
    # 표정 판정: head3(vgaf 특징 + 한국인 3분류 층 웃음·평온·부정, 부정은 0.8 이상만, 11/10 기본) | hsemotion8(재희: afew 8감정 최댓값)
    companion_face_head: str = _as_choice("COMPANION_FACE_HEAD", "head3", ("head3", "hsemotion8"))
    companion_face_neg_min: float = _as_float("COMPANION_FACE_NEG_MIN", 0.8)
    companion_stt_subdir: str = os.getenv("COMPANION_STT_SUBDIR", "companion/stt")  # D19: MODEL_BASE_PATH 하위
    companion_stt_model: str = os.getenv("COMPANION_STT_MODEL", "large-v3")  # D23
    # 받아쓰기 엔진: turbo(Whisper large-v3-turbo GPU, 11/09 기본) | sensevoice(재희 버전 기본) | whisper(faster-whisper, 비교용)
    # 근거: 어르신 200문장 글자 오류 turbo 2.4% · SenseVoice 4.7% (노션 "AI 안부말벗·정서분석 모델 학습 기록")
    companion_stt_engine: str = _as_choice("COMPANION_STT_ENGINE", "turbo", ("turbo", "sensevoice", "whisper"))
    # 챗봇 받아쓰기 API(/companion/stt) 엔진: turbo(기본, 2026-11-11) | sensevoice(되돌리기용)
    # 11/09에는 챗봇을 그대로 두기로 했으나, 11/11 실기기에서 "테스트 테스트"를 "데 투 데 투"로 받아써 바꿨다.
    # DGX 재현: 글자 오류 SenseVoice 4.6% · turbo 2.4%, 웅성거림 10dB 15.4% · 9.5%
    companion_stt_api_engine: str = _as_choice("COMPANION_STT_API_ENGINE", "turbo", ("sensevoice", "turbo"))
    # 받아쓰기 엔진 중 하나라도 turbo 면 기동 때 백그라운드에서 미리 올린다 (첫 호출이 모델 올리기로 20초 제한을 넘지 않게)
    companion_stt_warmup: bool = _as_bool("COMPANION_STT_WARMUP", True)
    companion_sensevoice_subdir: str = os.getenv("COMPANION_SENSEVOICE_SUBDIR", "companion/stt-sensevoice")
    companion_turbo_subdir: str = os.getenv("COMPANION_TURBO_SUBDIR", "companion/stt-turbo")  # openai/whisper-large-v3-turbo 폴더
    companion_turbo_device: str = os.getenv("COMPANION_TURBO_DEVICE", "cuda")  # cuda 는 fp16, cpu 는 fp32
    # 30초 창 하나에서 만들 최대 토큰 수. 한국어는 음절당 약 0.83토큰이라 128이면 빠르게 긴 대답이 잘린다 (11/09 점검 M-1). 디코더 한도 448 안
    companion_turbo_max_new_tokens: int = _as_int("COMPANION_TURBO_MAX_NEW_TOKENS", 224)
    companion_turbo_retry_sec: float = _as_float("COMPANION_TURBO_RETRY_SEC", 60.0)  # 모델 로드 실패 뒤 다시 시도할 때까지 (0이면 재시작 전까지 안 함)
    # 받아쓰기 결과에서 같은 1~4어절이 3번 이상 연달아 나오면 1번만 남긴다 (turbo 결과에만. false 면 그대로)
    companion_stt_collapse_repeats: bool = _as_bool("COMPANION_STT_COLLAPSE_REPEATS", True)
    # 말소리 감지: silero(말소리 합계가 기준 초 이상일 때만 받아쓰기, 11/09 기본) | rms(재희: 전체 소리 크기 0.01 기준)
    companion_vad: str = _as_choice("COMPANION_VAD", "silero", ("silero", "rms"))
    companion_vad_min_speech_sec: float = _as_float("COMPANION_VAD_MIN_SPEECH_SEC", 0.3)
    # 되묻기: logprob(turbo 토큰 평균 로그확률이 기준보다 낮으면 되묻기, 11/09 기본) | lexicon(재희: 형태소·어휘 목록 규칙)
    # 확률이 없는 턴(글 대답, SenseVoice)은 logprob 모드에서도 재희 규칙을 쓴다
    companion_confirm_mode: str = _as_choice("COMPANION_CONFIRM_MODE", "logprob", ("logprob", "lexicon"))
    companion_confirm_logprob: float = _as_float("COMPANION_CONFIRM_LOGPROB", -0.3)
    # 확신도가 낮아 되묻는 턴이 이 횟수만큼 연달아 나오면 무응답처럼 대화를 끝낸다 (TV·옆 사람 말소리, 11/09 점검 M-2). 0이면 끝내지 않는다
    companion_confirm_streak_limit: int = _as_int("COMPANION_CONFIRM_STREAK_LIMIT", 3)
    # 되묻는 문장·안전 안내 문장은 같은 대화에서 반복돼도 다른 문장으로 바꾸지 않는다 (11/09 점검 H-1·H-2). false 면 재희 동작
    companion_keep_fixed_replies: bool = _as_bool("COMPANION_KEEP_FIXED_REPLIES", True)
    companion_stt_api_max_sec: float = _as_float("COMPANION_STT_API_MAX_SEC", 30.0)  # 챗봇 공유 음성 인식 API 길이 상한
    companion_stt_api_max_mb: int = _as_int("COMPANION_STT_API_MAX_MB", 10)  # 받아쓰기 API(챗봇 공유) 음성 크기 상한 (2026-11-09 결정: 10MB)
    companion_stt_api_concurrency: int = _as_int_min("COMPANION_STT_API_CONCURRENCY", 2, 1)  # 동시에 돌릴 음성 인식 수
    companion_stt_api_wait_sec: float = _as_float("COMPANION_STT_API_WAIT_SEC", 5.0)  # 자리가 날 때까지 기다리는 시간
    companion_stt_api_timeout_sec: float = _as_float("COMPANION_STT_API_TIMEOUT_SEC", 20.0)  # 받아쓰기 한 번의 제한 시간 (넘으면 503 STT_UNAVAILABLE)
    companion_stt_device: str = os.getenv("COMPANION_STT_DEVICE", "cpu")
    companion_stt_compute: str = os.getenv("COMPANION_STT_COMPUTE", "int8")
    companion_text_emotion_subdir: str = os.getenv("COMPANION_TEXT_EMOTION_SUBDIR", "companion/text-emotion")
    companion_face_model_subdir: str = os.getenv("COMPANION_FACE_MODEL_SUBDIR", "companion/face-emotion")  # D24
    companion_face_detect_subdir: str = os.getenv("COMPANION_FACE_DETECT_SUBDIR", "companion/face-detect")  # YuNet (검출)
    companion_llm_base_url: str = os.getenv("COMPANION_LLM_BASE_URL", "http://127.0.0.1:11434")  # D7: Ollama
    companion_llm_model: str = os.getenv("COMPANION_LLM_MODEL", "exaone3.5:2.4b")  # 표시용 이름 (D7: transformers 직접 로드)
    companion_llm_subdir: str = os.getenv("COMPANION_LLM_SUBDIR", "companion/llm/exaone-3.5-2.4b")  # MODEL_BASE_PATH 아래 폴더 (D19)
    companion_llm_timeout_sec: int = _as_int("COMPANION_LLM_TIMEOUT_SEC", 25)  # 한 턴 60초 예산 안에 맞춘다 (tests/test_turn_budget.py)
    companion_llm_dtype: str = os.getenv("COMPANION_LLM_DTYPE", "bfloat16")  # 메모리 절약 (가중치는 float32 저장)
    # GPU가 있으면 GPU에 올린다. int8 이면 8비트로 줄인다 (CPU 메모리 부족 방지, 11/05·11/08 강제 종료 원인)
    companion_llm_quant: str = os.getenv("COMPANION_LLM_QUANT", "int8").strip().lower()
    # 위험 판정 프롬프트: v9(노인 자살 경고 신호, 11/10 기본) | v4(재희)
    companion_risk_prompt: str = _as_choice("COMPANION_RISK_PROMPT", "v9", ("v9", "v4"))
    # 위험 판정 기준 (위험 '있' 확률). 비우면 프롬프트에 맞춘다: v9 = 0.99 (평범한 안부 통화 강제 알림 11% → 1% 미만) · v4 = 0.5 (재희 11/09 측정)
    companion_risk_threshold: float | None = _opt_float("COMPANION_RISK_THRESHOLD")
    # 위험 낱말 규칙: kw4('자살'·'자해'는 앞이 한글이면 제외 - '혼자 살다', 관용 '아파 죽겠' 제외, 11/10 기본) | jaehee(띄어쓰기만 지우고 찾음)
    companion_risk_keyword_rule: str = _as_choice("COMPANION_RISK_KEYWORD_RULE", "kw4", ("kw4", "jaehee"))
    # LoRA 어댑터 (MODEL_BASE_PATH 아래). 폴더가 없으면 그 단계만 기본 모델로 돌린다
    companion_mood_lora_subdir: str = os.getenv("COMPANION_MOOD_LORA_SUBDIR", "companion/llm/lora-mood-e900")
    companion_reply_lora_subdir: str = os.getenv("COMPANION_REPLY_LORA_SUBDIR", "companion/llm/lora-reply-cc5")
    # 공감 대답(꼬리질문 턴) 생성을 대답 어댑터로: true(11/10 기본) | false(재희: 기본 모델). 프롬프트·규칙 검사는 같다
    companion_reply_lora: bool = _as_bool("COMPANION_REPLY_LORA", True)
    # 한 줄 요약 앞의 "요약:"·"기록:" 머리말을 지운다 (false 면 재희 그대로)
    companion_summary_strip_prefix: bool = _as_bool("COMPANION_SUMMARY_STRIP_PREFIX", True)
    # 11/10 판단 단계 점검(AI #209) 반영 - 끄면 #206 동작
    # 위험·기분 확률(각 0.1초 안팎)을 생성(신호·요약·대답)보다 먼저 계산한다. 끄면 넷을 함께 보낸다 (H-1)
    companion_risk_first: bool = _as_bool("COMPANION_RISK_FIRST", True)
    # 글 기분이 평온·기쁨인 턴은 '대답과 무관' 검사를 빼고, 권유 문장 검사(모든 꼬리질문 턴)를 더한다 (11/12).
    # 끄면 전과 같다. 기분은 risk_first 가 켜져 있을 때만 생성 전에 있다 (꺼져 있으면 전 규칙)
    companion_reply_relax_calm: bool = _as_bool("COMPANION_REPLY_RELAX_CALM", True)
    # 말을 피하는 대답("말하기 싫어", "묻지 마")에는 모델 대답 대신 물러나는 문장으로 답한다 (11/12). 끄면 전과 같다
    companion_avoid_reply: bool = _as_bool("COMPANION_AVOID_REPLY", True)
    # 한 대화에서 이 횟수만큼 말을 피하면 마무리 문장으로 끝낸다 (endReason AVOIDED - BE V72, 끝내기 USER_STOP과 구별). 0이면 일찍 끝내지 않는다
    companion_avoid_end_count: int = _as_int("COMPANION_AVOID_END_COUNT", 2)
    # 몸 응급 낱말(숨·가슴·쓰러짐·뇌졸중 징후)만 걸린 위험 턴은 몸 응급용 안내 문장으로 답한다 (11/12). 끄면 전과 같다
    companion_body_safety_reply: bool = _as_bool("COMPANION_BODY_SAFETY_REPLY", True)
    # 단어 확인(되묻기) 턴에서도 위험 모델을 돌리고, 위험이면 안전 안내 문장으로 답한다 (H-3). 끄면 낱말 규칙만
    companion_confirm_risk_model: bool = _as_bool("COMPANION_CONFIRM_RISK_MODEL", True)
    # 말벗 생성(신호·요약·대답) 동시 실행 상한. 0이면 상한 없음. 위험·기분 확률 계산은 상한 밖 (H-5, GPU 자원)
    companion_llm_gen_concurrency: int = _as_int("COMPANION_LLM_GEN_CONCURRENCY", 2)
    # 말벗이 켜져 있으면(키 있음) 서버를 켤 때 EXAONE·어댑터를 백그라운드로 미리 올린다 (M-1)
    companion_llm_warmup: bool = _as_bool("COMPANION_LLM_WARMUP", True)
    # 실시간 감지가 GPU 메모리 부족(CUDA OOM)이면 캐시를 비우고 그 프레임을 한 번 더 본다 (GPU 자원 ⑤)
    live_detect_oom_retry: bool = _as_bool("LIVE_DETECT_OOM_RETRY", True)
    # 한 턴 전체 상한 (백엔드 요청: 60초). 넘으면 미리 정한 문장으로 답한다
    companion_turn_cap_sec: float = _as_float("COMPANION_TURN_CAP_SEC", 60.0)
    companion_llm_num_predict: int = _as_int("COMPANION_LLM_NUM_PREDICT", 256)
    companion_llm_temperature: float = _as_float("COMPANION_LLM_TEMPERATURE", 0.1)
    companion_llm_retry: int = _as_int("COMPANION_LLM_RETRY", 0)  # D20: 재시도 0회(2026-11-09 결정), 실패하면 바로 대체 문장. 1로 올리면 1회 재시도
    companion_pseudonym_salt: str = os.getenv("COMPANION_PSEUDONYM_SALT", "")  # 비면 개선 DB 적재 끔
    companion_log_transcript: bool = _as_bool("COMPANION_LOG_TRANSCRIPT", False)  # D10
    companion_log_database_url: str = os.getenv("COMPANION_LOG_DATABASE_URL", "")  # D11 결정(2026-11-02): 턴 기록은 백엔드 DB. 이 값은 비워 둔다
    # 검수 목록 전용 (명세 9-1 5·6번). AI가 만든 문장과 질문만 담고 대답 원문은 담지 않는다
    companion_review_database_url: str = os.getenv("COMPANION_REVIEW_DATABASE_URL", "sqlite:///./data/companion_review.db")
    companion_log_retention_days: int = _as_int("COMPANION_LOG_RETENTION_DAYS", 30)  # D18 (사용자 결정 30일)
    companion_no_response_limit: int = _as_int("COMPANION_NO_RESPONSE_LIMIT", 3)  # D15 (사용자 결정 2026-11-02: 3번 연속이면 종료)
    # D16 (사용자 결정 2026-11-02): 중간에 그만두면 그때까지의 턴을 그대로 기록한다. 최소 턴 기준 없음.
    companion_stt_timeout_sec: float = _as_float("COMPANION_STT_TIMEOUT_SEC", 20.0)  # S9-1 (한 턴 60초 예산)
    companion_face_timeout_sec: float = _as_float("COMPANION_FACE_TIMEOUT_SEC", 10.0)  # S9-1
    companion_text_timeout_sec: float = _as_float("COMPANION_TEXT_TIMEOUT_SEC", 5.0)  # S9-1
    companion_record_timeout_sec: float = _as_float("COMPANION_RECORD_TIMEOUT_SEC", 5.0)  # S8-5 턴 기록 전송
    companion_baseline_ratio: float = _as_float("COMPANION_BASELINE_RATIO", 2.0)  # D22
    companion_baseline_window_days: int = _as_int("COMPANION_BASELINE_WINDOW_DAYS", 30)  # D22 (사용자 결정 2026-11-02: 최근 30일)
    companion_baseline_min_days: int = _as_int("COMPANION_BASELINE_MIN_DAYS", 20)  # D22 (사용자 결정 2026-11-02: 30일 중 20일 이상)
    companion_backend_base_url: str = os.getenv("COMPANION_BACKEND_BASE_URL", "")  # 질문 조회 (GET /companion/question)
    companion_backend_api_key: str = os.getenv("COMPANION_BACKEND_API_KEY", "")  # 백엔드 호출용 키
    companion_backend_timeout_sec: int = _as_int("COMPANION_BACKEND_TIMEOUT_SEC", 5)


@lru_cache
def get_settings() -> Settings:
    return Settings()
