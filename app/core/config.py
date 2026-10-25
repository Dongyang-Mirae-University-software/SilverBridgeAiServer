from __future__ import annotations

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


def _as_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _as_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
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
    # 말벗(companion) 전용 읽기 키. 최신 프레임·상태 조회 두 경로에만 쓰인다(헤더 전용). 비어 있으면 비활성.
    companion_stream_api_key: str = os.getenv("COMPANION_STREAM_API_KEY", "")

    model_base_path: str = os.getenv("MODEL_BASE_PATH", str(_PROJECT_ROOT / "models"))
    upload_base_path: str = os.getenv("UPLOAD_BASE_PATH", str(_PROJECT_ROOT / "uploads"))
    snapshot_base_path: str = os.getenv(
        "SNAPSHOT_BASE_PATH",
        str(_PROJECT_ROOT / "uploads" / "snapshots"),
    )
    default_detection_threshold: float = float(os.getenv("DEFAULT_DETECTION_THRESHOLD", "0.75"))
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
    claude_timeout_sec: int = _as_int("CLAUDE_TIMEOUT_SEC", 60)
    # 라우터·예약 추출 Claude 호출의 출력 상한(생각 토큰 포함, 2026-10-25). MedGemma MAX_NEW_TOKENS(512)와 따로 둔다 -
    # opus 5.5는 생각을 끌 수 없어 위기 문장에서 생각 블록이 512를 다 써 답이 잘렸다. 범위 밖 값은 기본값으로 대체한다.
    claude_max_tokens: int = _as_int_in_range("CLAUDE_MAX_TOKENS", 2048, 256, 8192)
    # 약 봉투 OCR(POST /api/v1/medication/ocr). 키는 위 ANTHROPIC_API_KEY 를 같이 쓴다.
    # 모델은 2026-10-22 측정으로 sonnet 확정 - haiku 는 없는 약을 지어내 금지(기동 시 sonnet 으로 대체).
    medication_ocr_enabled: bool = _as_bool("MEDICATION_OCR_ENABLED", True)
    medication_ocr_model: str = os.getenv("MEDICATION_OCR_MODEL", "claude-sonnet-5-5")
    reservation_api_base_url: str = os.getenv(
        "RESERVATION_API_BASE_URL",
        "http://127.0.0.1:6015/api/v1",
    )
    reservation_api_timeout_sec: int = _as_int("RESERVATION_API_TIMEOUT_SEC", 1200)
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
    top_p: float = _as_float("TOP_P", 0.9)
    repetition_penalty: float = _as_float("REPETITION_PENALTY", 1.1)
    do_sample: bool = _as_bool("DO_SAMPLE", True)
    load_in_8bit: bool = _as_bool("LOAD_IN_8BIT", False)
    # MedGemma 가중치 양자화. none(기본, bf16 그대로) | nf4(bitsandbytes 4비트). 모르는 값은 none으로 본다.
    medgemma_quant: str = os.getenv("MEDGEMMA_QUANT", "none").strip().lower()
    chat_history_keep_turns: int = _as_int("CHAT_HISTORY_KEEP_TURNS", 8)

    def resolved_chat_model_id(self) -> str:
        path = (self.chat_model_path or "").strip()
        if path:
            return path
        return self.default_chat_model

    @property
    def reservation_headers(self) -> dict[str, str] | None:
        if not self.reservation_api_key.strip():
            return None
        return {"Authorization": f"Bearer {self.reservation_api_key.strip()}"}

    @property
    def has_reservation_account_secret(self) -> bool:
        return bool(self.reservation_account_secret.strip())

    @property
    def has_reservation_service_credentials(self) -> bool:
        return bool(self.reservation_service_email.strip() and self.reservation_service_password.strip())

    docs_path: str = os.getenv("DOCS_PATH", "/api/docs")
    openapi_path: str = os.getenv("OPENAPI_PATH", "/api/openapi.json")
    redoc_path: str = os.getenv("REDOC_PATH", "/api/redoc")

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
    # FE 이상감지 모니터가 아직 쿼리로 접속하므로 기본은 허용 — FE 가 헤더/백엔드 경유로 바꾸면 false 로 끈다.
    ws_allow_query_api_key: bool = _as_bool("WS_ALLOW_QUERY_API_KEY", True)
    # 챗 로그 상세(GET /api/v1/chat/logs/{id})에 userId 필수 + 소유자 일치. 기본 OFF —
    # FE 의 상세 호출 함수가 userId 없이 부르는 형태라, FE 가 userId 를 붙인 뒤 켠다.
    chat_require_user_id: bool = _as_bool("CHAT_REQUIRE_USER_ID", False)
    # 게임 API(/api/v1/games/**)에 X-API-Key 필수. 기본 OFF — 피보호자 게임 iframe 과 그 안의 JS 가
    # 브라우저에서 키 없이 직접 부른다. FE 가 iframe·embed 호출을 프록시 경유로 바꾼 뒤 켠다.
    game_require_api_key: bool = _as_bool("GAME_REQUIRE_API_KEY", False)

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
    frame_quality_gate_enabled: bool = _as_bool("FRAME_QUALITY_GATE_ENABLED", True)
    frame_quality_white_ratio: float = _as_float_in_range("FRAME_QUALITY_WHITE_RATIO", 0.5, 0.05, 1.0)
    frame_quality_black_ratio: float = _as_float_in_range("FRAME_QUALITY_BLACK_RATIO", 0.95, 0.05, 1.0)
    frame_quality_std_min: float = _as_float_in_range("FRAME_QUALITY_STD_MIN", 12.0, 0.0, 60.0)
    frame_quality_lap_max: float = _as_float_in_range("FRAME_QUALITY_LAP_MAX", 10.0, 0.0, 500.0)
    frame_quality_blur_std_max: float = _as_float_in_range("FRAME_QUALITY_BLUR_STD_MAX", 30.0, 0.0, 100.0)

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
