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


class Settings(BaseModel):
    app_name: str = os.getenv("APP_NAME", "SilverBridge AI Server")
    app_env: str = os.getenv("APP_ENV", "development")
    app_version: str = os.getenv("APP_VERSION", "0.1.0")
    app_port: int = _as_int("APP_PORT", 9000)
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    database_url: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{(_PROJECT_ROOT / 'data' / 'ai_server.db').as_posix()}",
    )
    api_key: str = os.getenv("API_KEY", "change-me")

    model_base_path: str = os.getenv("MODEL_BASE_PATH", str(_PROJECT_ROOT / "models"))
    upload_base_path: str = os.getenv("UPLOAD_BASE_PATH", str(_PROJECT_ROOT / "uploads"))
    snapshot_base_path: str = os.getenv(
        "SNAPSHOT_BASE_PATH",
        str(_PROJECT_ROOT / "uploads" / "snapshots"),
    )
    default_detection_threshold: float = float(os.getenv("DEFAULT_DETECTION_THRESHOLD", "0.75"))

    default_chat_model: str = os.getenv("DEFAULT_CHAT_MODEL", "google/medgemma-1.5-4b-it")
    chat_model_path: str = os.getenv("CHAT_MODEL_PATH", "")
    hf_token: str = os.getenv("HF_TOKEN", "")

    docs_path: str = os.getenv("DOCS_PATH", "/docs")
    openapi_path: str = os.getenv("OPENAPI_PATH", "/openapi.json")
    redoc_path: str = os.getenv("REDOC_PATH", "/redoc")

    stream_sample_every_n_frames: int = _as_int("STREAM_SAMPLE_EVERY_N_FRAMES", 15)
    stream_fallback_interval_sec: int = _as_int("STREAM_FALLBACK_INTERVAL_SEC", 2)
    save_normal_results: bool = _as_bool("SAVE_NORMAL_RESULTS", True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
    fire_smoke_enabled: bool = _as_bool("FIRE_SMOKE_ENABLED", True)
    fire_smoke_model_path: str = os.getenv("FIRE_SMOKE_MODEL_PATH", "fire_smoke.pt")
    fire_smoke_conf_threshold: float = _as_float("FIRE_SMOKE_CONF_THRESHOLD", 0.35)
    fire_smoke_iou_threshold: float = _as_float("FIRE_SMOKE_IOU_THRESHOLD", 0.45)

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
