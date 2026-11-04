from __future__ import annotations

# 사용자 객관식 답 (S2 질문): WAV·webm·mp3 만 받는다. 명세 Q12의 6종 목록과 다름 (C-27).
ALLOWED_FORMATS = ("wav", "webm", "mp3")


def detect_audio_format(data: bytes) -> str | None:
    """파일 머리(매직 바이트)로 형식을 판별한다. 확장자는 믿지 않는다. 허용 밖이면 None."""
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if len(data) >= 4 and data[:4] == b"\x1a\x45\xdf\xa3":  # EBML 머리 (webm)
        return "webm"
    if len(data) >= 3 and data[:3] == b"ID3":
        return "mp3"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:  # MPEG 프레임 동기 (mp3)
        return "mp3"
    return None
