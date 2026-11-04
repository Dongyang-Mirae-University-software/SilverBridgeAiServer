from __future__ import annotations

# 얼굴 사진은 JPEG·PNG만 받는다. 확장자나 보낸 종류(content-type)는 믿지 않고 파일 머리(매직 바이트)로 본다.
ALLOWED_IMAGE_FORMATS = ("jpeg", "png")


def detect_image_format(data: bytes) -> str | None:
    """사진 머리로 형식을 판별한다. JPEG·PNG가 아니면 None."""
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":  # JPEG (SOI + 다음 마커)
        return "jpeg"
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":  # PNG 시그니처
        return "png"
    return None
