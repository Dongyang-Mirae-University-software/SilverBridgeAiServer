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
    # 라이브 추론 시간 요약 로그 주기(초, 0 이하 = 끔). 감지기를 여러 개 켜면 프레임당 추론이 길어진다 —
    # 너무 느리면 STREAM_SAMPLE_EVERY_N_FRAMES 를 늘린다.
    stream_infer_log_interval_sec: float = _as_float("STREAM_INFER_LOG_INTERVAL_SEC", 60.0)

    default_chat_model: str = os.getenv("DEFAULT_CHAT_MODEL", "google/medgemma-1.5-4b-it")
    chat_model_path: str = os.getenv("CHAT_MODEL_PATH", "")
    hf_token: str = os.getenv("HF_TOKEN", "")
    chat_enable_llm: bool = _as_bool("CHAT_ENABLE_LLM", True)
    gpt_api_key: str = os.getenv("GPT_API_KEY", "")
    gpt_model_name: str = os.getenv("GPT_MODEL_NAME", "gpt-4o-mini")
    gpt_timeout_sec: int = _as_int("GPT_TIMEOUT_SEC", 60)
    chat_upstream_url: str = os.getenv("CHAT_UPSTREAM_URL", "http://127.0.0.1:6012")
    chat_upstream_timeout_sec: int = _as_int("CHAT_UPSTREAM_TIMEOUT_SEC", 1200)
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
    reservation_service_email: str = os.getenv("RESERVATION_SERVICE_EMAIL", "user1@example.com")
    reservation_service_password: str = os.getenv("RESERVATION_SERVICE_PASSWORD", "User123456!")
    torch_dtype: str = os.getenv("TORCH_DTYPE", "float16")
    device_map: str = os.getenv("DEVICE_MAP", "auto")
    require_gpu: bool = _as_bool("REQUIRE_GPU", False)
    max_new_tokens: int = _as_int("MAX_NEW_TOKENS", 512)
    temperature: float = _as_float("TEMPERATURE", 0.4)
    top_p: float = _as_float("TOP_P", 0.9)
    repetition_penalty: float = _as_float("REPETITION_PENALTY", 1.1)
    do_sample: bool = _as_bool("DO_SAMPLE", True)
    load_in_8bit: bool = _as_bool("LOAD_IN_8BIT", False)
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
