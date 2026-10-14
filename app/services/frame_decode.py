from __future__ import annotations

import numpy as np

try:  # opencv 가 없는 환경(일부 단위 테스트)에서도 import 는 되어야 한다 - 그때는 디코드하지 않는다.
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


def decode_bgr(frame_bytes: bytes) -> np.ndarray | None:
    """JPEG 바이트 → BGR 배열. 비었거나 디코드할 수 없으면(또는 opencv 미설치) None.

    라이브 분석은 프레임당 이것을 한 번만 부르고 감지기·품질 검사가 결과 배열을 함께 쓴다(읽기 전용).
    """
    if cv2 is None or not frame_bytes:
        return None
    frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        return None
    return frame
