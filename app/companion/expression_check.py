"""한 장의 사진에서 얼굴을 찾고 표정을 읽는다 (테스트 화면·표정 인식 확인용).

- 얼굴 검출: OpenCV Haar (face_window.load_detector 와 같은 방식)
- 표정: HSEmotion (enet_b0_8_best_afew). 판단 규칙은 face_capture_test.decide 와 같다.
- 입력은 메모리에서만 읽고 버린다. 파일로 저장하지 않는다.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

import face_capture_test as fct
import face_window


@lru_cache(maxsize=1)
def _recognizer():
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer

    rec = EmotiEffLibRecognizer(engine="torch", model_name="enet_b0_8_best_afew")
    return rec, list(rec.idx_to_emotion_class.values())


def classify_jpeg(jpeg: bytes) -> dict:
    """얼굴이 있으면 표정을 판단하고, 없으면 face=False 를 돌려준다."""
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"face": False, "expression": None, "reason": "이미지를 읽을 수 없음"}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_window.load_detector().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return {"face": False, "expression": None, "reason": "얼굴 없음"}

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # 가장 큰 얼굴
    crop = cv2.cvtColor(img[y : y + h, x : x + w], cv2.COLOR_BGR2RGB)
    rec, names = _recognizer()
    _, probs = rec.predict_emotions(crop, logits=False)
    pv = np.asarray(probs).reshape(-1)
    order = np.argsort(-pv)[:3]
    return {
        "face": True,
        "expression": fct.decide(pv, names),
        "top": names[order[0]],
        "topScore": round(float(pv[order[0]]), 3),
        "top3": [{"label": names[i], "score": round(float(pv[i]), 3)} for i in order],
    }
