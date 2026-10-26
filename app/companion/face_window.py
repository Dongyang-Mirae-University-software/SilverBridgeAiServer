"""대답하는 동안 얼굴이 보이는지 확인한다 (계획서 Phase 3).

- AI 서버 스트림의 최신 프레임을 일정 간격으로 받아 얼굴 검출만 한다. 표정 판단은 하지 않는다 (사용자 결정: 표정 보완은 추후).
- 프레임은 메모리에서 읽고 버린다. 파일로 저장하지 않는다.
- 얼굴이 기준 비율보다 적게 보이면 "판단 불가"로 기록한다. 얼굴 신호가 없으면 텍스트 결과만으로 판단한다.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np

import stream_client

EVERY_SEC = 2.5  # 프레임을 받는 간격 (계획서 3: 2~3초 간격)
FACE_MIN_RATE = 0.5  # 이 비율 이상의 프레임에서 얼굴이 보여야 "얼굴 확인"

STATUS_SEEN = "얼굴 확인"
STATUS_UNKNOWN = "판단 불가"
STATUS_NO_STREAM = "얼굴 확인 안 함 (말벗 스트림 설정 없음)"
STATUS_UNREACHABLE = "얼굴 확인 불가 (AI 서버 연결 실패)"


@lru_cache(maxsize=1)
def load_detector() -> cv2.CascadeClassifier:
    # OpenCV 는 한글 경로의 모델 파일을 읽지 못하므로, 영문 임시 폴더로 복사해서 쓴다 (face_capture_test.py 와 같은 방식).
    src = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    tmp_model = Path(tempfile.gettempdir()) / "haarcascade_frontalface_default.xml"
    shutil.copyfile(src, tmp_model)
    return cv2.CascadeClassifier(str(tmp_model))


def has_face(jpeg: bytes, detector=None) -> bool:
    """JPEG 한 장에 얼굴이 있는지 본다. 디코딩이 안 되면 얼굴 없음으로 센다. 파일은 쓰지 않는다."""
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    det = detector or load_detector()
    return len(det.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))) > 0


def summarize(flags: Iterable[bool], min_rate: float = FACE_MIN_RATE) -> dict:
    """프레임별 얼굴 여부(True/False)를 한 번의 결과로 묶는다."""
    flags = list(flags)
    frames = len(flags)
    with_face = sum(flags)
    if frames == 0:
        status = f"{STATUS_UNKNOWN} (프레임 없음)"
    elif with_face / frames >= min_rate:
        status = STATUS_SEEN
    else:
        status = f"{STATUS_UNKNOWN} (얼굴이 거의 보이지 않음)"
    return {"faceStatus": status, "frames": frames, "withFace": with_face}


def collect(session_id: str, seconds: float, every_sec: float = EVERY_SEC,
            fetch: Optional[Callable[[str], bytes]] = None,
            detector=None, clock=time.monotonic, sleep=time.sleep) -> dict:
    """seconds 동안 프레임을 받아 얼굴 여부를 센다. 스트림 설정이 없거나 연결이 안 되면 결과 문구로 알린다."""
    fetch = fetch or stream_client.fetch_latest_frame
    try:
        stream_client._settings()
    except stream_client.StreamError:
        return {"faceStatus": STATUS_NO_STREAM, "frames": 0, "withFace": 0}

    flags: list[bool] = []
    deadline = clock() + seconds
    while clock() < deadline:
        jpeg: Optional[bytes] = None
        try:
            jpeg = fetch(session_id)
        except stream_client.StreamError as err:
            if err.code == "UNREACHABLE":
                return {"faceStatus": STATUS_UNREACHABLE, "frames": len(flags), "withFace": sum(flags)}
            flags.append(False)  # 프레임 없음·오류는 얼굴 없음으로 센다
        else:
            flags.append(has_face(jpeg, detector))
        jpeg = None  # 프레임은 여기서 버린다 (파일로 쓰지 않음)
        sleep(every_sec)
    return summarize(flags)
