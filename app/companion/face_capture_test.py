"""Phase 3 시험: 노트북 웹캠에서 얼굴을 찾고 표정(HSEmotion)을 판단한다.

개인정보 원칙
- 프레임은 메모리에서만 처리하고 파일로 저장하지 않는다.
- 화면 창에는 얼굴 사각형과 현재 표정 라벨만 보여 준다.
- 결과는 집계 숫자만 출력한다.
- 본인 얼굴로만 테스트한다. 다른 사람은 본인 동의를 받은 뒤에만 찍는다.

판단 규칙 (2026-10-25 보완, 3가지 적용)
1. 가장 높은 확률(1위)이 CONF_MIN(0.3) 이상이어야 한다.
2. 1위와 2위의 차이가 MARGIN_MIN(0.05) 이상이어야 한다. 1위가 뚜렷하지 않으면 판단 보류.
3. 평온(Neutral)은 확률 0.2 이상, 차이 0.03 이상이면 인정한다 (무표정이 많이 놓치는 문제 보완).
- 프레임 결과는 최근 WINDOW(10)번 중 절반 이상 같을 때만 화면에 표시한다. 아니면 판단 보류.

실행 (companion-emotion 폴더에서, 카메라 앞에 앉아서):
    venv\\Scripts\\python.exe companion\\face_capture_test.py --seconds 30
창에서 q 를 누르면 일찍 끝난다. --no-preview 로 창을 끌 수 있다. --debug 는 확률 상위 3개를 출력한다.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

CONF_MIN = 0.3
MARGIN_MIN = 0.05
NEUTRAL_CONF_MIN = 0.2
NEUTRAL_MARGIN_MIN = 0.03
WINDOW = 10
HOLD = "판단 보류"


def load_detector() -> cv2.CascadeClassifier:
    # OpenCV 는 한글 경로의 모델 파일을 읽지 못하므로, 영문 임시 폴더로 복사해서 쓴다.
    src = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    tmp_model = Path(tempfile.gettempdir()) / "haarcascade_frontalface_default.xml"
    shutil.copyfile(src, tmp_model)
    return cv2.CascadeClassifier(str(tmp_model))


def decide(pv: np.ndarray, names: list[str]) -> str:
    """한 프레임의 확률로 표정을 정한다 (규칙 1~3)."""
    order = np.argsort(-pv)
    top, second = order[0], order[1]
    top_p, margin = float(pv[top]), float(pv[top] - pv[second])
    if names[top] == "Neutral":
        ok = top_p >= NEUTRAL_CONF_MIN and margin >= NEUTRAL_MARGIN_MIN
    else:
        ok = top_p >= CONF_MIN and margin >= MARGIN_MIN
    return names[top] if ok else HOLD


def stable_label(history: deque) -> str:
    """최근 결과 중 절반 이상이 같은 것이면 그 결과, 아니면 판단 보류."""
    if not history:
        return HOLD
    label, count = Counter(history).most_common(1)[0]
    return label if count * 2 > len(history) else HOLD


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=30.0, help="측정 시간(초)")
    ap.add_argument("--every", type=float, default=0.5, help="표정 분석 간격(초)")
    ap.add_argument("--no-preview", action="store_true", help="창을 띄우지 않음")
    ap.add_argument("--debug", action="store_true", help="확률 상위 3개와 얼굴 위치를 출력")
    args = ap.parse_args()

    from emotiefflib.facial_analysis import EmotiEffLibRecognizer

    rec = EmotiEffLibRecognizer(engine="torch", model_name="enet_b0_8_best_afew")
    names = list(rec.idx_to_emotion_class.values())
    detector = load_detector()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("웹캠을 열 수 없습니다. 카메라 권한과 연결을 확인하세요.")

    seen = 0
    with_face = 0
    raw_labels: Counter[str] = Counter()
    history: deque = deque(maxlen=WINDOW)
    last_box = None
    shown = "얼굴 찾는 중"
    start = time.time()
    next_at = start
    print(f"{args.seconds:.0f}초 동안 창을 보세요. q 를 누르면 일찍 끝납니다. (저장하지 않음)", flush=True)

    while time.time() - start < args.seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        now = time.time()
        if now >= next_at:
            next_at = now + args.every
            seen += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
            if len(faces) == 0:
                last_box = None
                history.clear()
                shown = "얼굴 없음"
            else:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])  # 가장 큰 얼굴
                with_face += 1
                last_box = (x, y, w, h)
                crop = cv2.cvtColor(frame[y : y + h, x : x + w], cv2.COLOR_BGR2RGB)
                _, probs = rec.predict_emotions(crop, logits=False)
                pv = np.asarray(probs).reshape(-1)
                if args.debug:
                    order = np.argsort(-pv)[:3]
                    print("  box=(%d,%d,%d,%d) top3: %s" % (x, y, w, h, ", ".join(f"{names[i]} {pv[i]:.2f}" for i in order)), flush=True)
                result = decide(pv, names)
                raw_labels[result] += 1
                history.append(result)
                shown = stable_label(history)

        if not args.no_preview:
            view = frame.copy()
            if last_box is not None:
                x, y, w, h = last_box
                cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(view, f"Expression: {shown}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(view, "q: quit (not saved)", (10, view.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("companion face test", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

    rate = with_face / max(seen, 1)
    print(f"\n분석 시도: {seen}회 / 얼굴 검출: {with_face}회 ({rate:.0%})", flush=True)
    if raw_labels:
        print("개별 판단 (규칙 적용 후):", dict(raw_labels.most_common()), flush=True)
    else:
        print("얼굴이 검출되지 않았습니다. 카메라 정면, 밝은 조명을 확인하세요.", flush=True)


if __name__ == "__main__":
    main()
