"""안부 말벗(checkin) 설정. 다른 기능 설정과 섞이지 않게 이 모듈 안에서 환경 변수를 읽는다.

킬 스위치 CHECKIN_ENABLED 는 기본 꺼짐이다 - 켜지 않으면 모델을 올리지 않고 모든 경로가 503 CHECKIN_DISABLED 다
(gosky AI 서버 기동·GPU 메모리에 영향이 없게).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    return default if not raw else raw in {"1", "true", "yes", "on"}


def _int(name: str, default: int, low: int, high: int) -> int:
    try:
        value = int(os.getenv(name, "").strip())
    except ValueError:
        return default
    return value if low <= value <= high else default


@dataclass(frozen=True)
class CheckinSettings:
    enabled: bool
    exaone_model: str  # HF id 또는 로컬 경로 (transformers 5 에서는 remote code 를 고친 로컬 사본이 필요할 수 있다)
    # QLoRA 추가 가중치 폴더(비면 기본 모델 + 프롬프트만). 정리와 정서는 따로 둔다 - 하나로 함께 배우게 하면
    # 정서는 좋아졌지만 정리가 처음 보는 대답 97→94칸·헛경보 0→2로 나빠졌다(2026-11-01 DGX).
    summary_adapter: str
    emotion_adapter: str
    sensevoice_model: str
    # 표정(보조 신호): 셋 다 있어야 켜진다. 비면 face 는 null.
    face_model: str  # HSEmotion enet_b0_8_best_vgaf.pt
    face_head: str  # 한국인 표정으로 다시 학습한 3분류 층
    face_detector: str  # OpenCV YuNet face_detection_yunet_2023mar.onnx
    face_neg_pct: int  # 부정 확률이 이 % 이상인 사진만 부정으로 센다
    device: str
    max_audio_bytes: int
    max_audio_sec: int
    max_frames: int
    max_frame_bytes: int
    max_turns: int
    retention_days: int
    busy_wait_sec: int
    reply_max_tokens: int


@lru_cache
def get_checkin_settings() -> CheckinSettings:
    return CheckinSettings(
        enabled=_bool("CHECKIN_ENABLED", False),
        exaone_model=os.getenv("CHECKIN_EXAONE_MODEL", "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct").strip(),
        summary_adapter=os.getenv("CHECKIN_SUMMARY_ADAPTER", "").strip(),
        emotion_adapter=os.getenv("CHECKIN_EMOTION_ADAPTER", "").strip(),
        sensevoice_model=os.getenv("CHECKIN_SENSEVOICE_MODEL", "FunAudioLLM/SenseVoiceSmall").strip(),
        face_model=os.getenv("CHECKIN_FACE_MODEL", "").strip(),
        face_head=os.getenv("CHECKIN_FACE_HEAD", "").strip(),
        face_detector=os.getenv("CHECKIN_FACE_DETECTOR", "").strip(),
        face_neg_pct=_int("CHECKIN_FACE_NEG_PCT", 80, 50, 99),
        device=os.getenv("CHECKIN_DEVICE", "cuda:0").strip(),
        max_audio_bytes=_int("CHECKIN_MAX_AUDIO_BYTES", 5 * 1024 * 1024, 64 * 1024, 10 * 1024 * 1024),
        max_audio_sec=_int("CHECKIN_MAX_AUDIO_SEC", 60, 5, 120),
        max_frames=_int("CHECKIN_MAX_FRAMES", 120, 0, 240),
        max_frame_bytes=_int("CHECKIN_MAX_FRAME_BYTES", 300 * 1024, 16 * 1024, 2 * 1024 * 1024),
        max_turns=_int("CHECKIN_MAX_TURNS", 8, 1, 12),
        retention_days=_int("CHECKIN_RETENTION_DAYS", 30, 1, 30),
        busy_wait_sec=_int("CHECKIN_BUSY_WAIT_SEC", 10, 0, 60),
        reply_max_tokens=_int("CHECKIN_REPLY_MAX_TOKENS", 80, 20, 200),
    )
