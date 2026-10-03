from __future__ import annotations

import cv2
import numpy as np

WEBM_MAGIC = b"\x1a\x45\xdf\xa3"


def make_jpeg(width: int, height: int, index: int = 0) -> bytes:
    """프레임마다 색·위치가 바뀌는 합성 JPEG."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = ((index * 37) % 256, (index * 11) % 256, 128)
    x = (index * 17) % max(1, width - 20)
    cv2.rectangle(img, (x, height // 4), (x + 20, height // 2), (255, 255, 255), -1)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    return buf.tobytes()


def fake_webm(*_: object) -> bytes:
    return WEBM_MAGIC + b"fake"
