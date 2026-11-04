from __future__ import annotations

from app.core.config import get_settings


def get_stt_engine():
    """설정(COMPANION_STT_ENGINE)에 맞는 받아쓰기 엔진을 돌려준다.

    sensevoice: 팀 합의 기본값. whisper: faster-whisper (비교·대체용). 둘 다 transcribe(bytes), warm_up() 을 가진다.
    """
    if get_settings().companion_stt_engine == "whisper":
        from app.services.companion.stt_service import stt_service

        return stt_service
    from app.services.companion.sensevoice_service import sensevoice_service

    return sensevoice_service
