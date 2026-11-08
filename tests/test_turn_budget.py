"""한 음성 턴의 최악 대기 시간이 백엔드 대기 예산(60초) 안에 드는지 고정한다.

순서: 받아쓰기 → 얼굴 → 글 감정 → 응답·신호(병렬, 각 1회 시도).
"""

from app.core.config import Settings

BUDGET_SEC = 60


def test_worst_case_voice_turn_fits_60_seconds():
    s = Settings()
    attempts = 1 + s.companion_llm_retry
    worst = (s.companion_stt_timeout_sec + s.companion_face_timeout_sec
             + s.companion_text_timeout_sec + s.companion_llm_timeout_sec * attempts)
    assert worst <= BUDGET_SEC, worst


def test_llm_defaults_are_single_attempt():
    s = Settings()
    assert s.companion_llm_retry == 0
    assert s.companion_llm_timeout_sec == 25
