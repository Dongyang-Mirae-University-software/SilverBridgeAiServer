"""표정 읽기가 밝기·기울기 변화에 얼마나 흔들리는지 본다 (계획서 4.10).

- 데이터: 공개 얼굴 사진 (Facial Emotion Recognition test 폴더, eval_expression_check.py 와 같은 60장, 시드 0)
- 변형: 원본, 밝게(1.5배), 어둡게(0.5배), 오른쪽 15도 기울임, 왼쪽 15도 기울임
- 판단: 사진 전체를 얼굴로 보고 읽음 (이 사진들은 이미 얼굴만 잘라 둔 것)
- 기록: 변형별 정확도 숫자만 logs/expression_robustness.json 에 쓴다.

실행 (저장소 루트에서):
    set FACE_TEST_DIR=<Facial Emotion Recognition.v2i.folder 의 test 폴더>
    venv\\Scripts\\python.exe -u eval\\eval_expression_robustness.py
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트
sys.path.insert(0, str(ROOT / "app" / "companion"))

import expression_check  # noqa: E402
import face_capture_test as fct  # noqa: E402

MAP = {"angry": "Anger", "fear": "Fear", "happy": "Happiness", "neutral": "Neutral", "sad": "Sadness", "surprise": "Surprise"}
PER_LABEL = 10
SEED = 0


def rotate(img: np.ndarray, deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REFLECT)


VARIANTS = {
    "원본": lambda im: im,
    "밝게 (1.5배)": lambda im: np.clip(im.astype(np.float32) * 1.5, 0, 255).astype(np.uint8),
    "어둡게 (0.5배)": lambda im: np.clip(im.astype(np.float32) * 0.5, 0, 255).astype(np.uint8),
    "오른쪽 15도": lambda im: rotate(im, -15),
    "왼쪽 15도": lambda im: rotate(im, 15),
}


def main() -> None:
    test_dir = Path(os.environ["FACE_TEST_DIR"])
    rng = random.Random(SEED)
    rows = []
    for label in MAP:
        files = sorted(p for p in (test_dir / label).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for p in rng.sample(files, min(PER_LABEL, len(files))):
            rows.append((label, p))

    rec, names = expression_check._recognizer()
    result = {}
    for name, fn in VARIANTS.items():
        correct = wrong = hold = 0
        for label, p in rows:
            img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(fn(img), cv2.COLOR_BGR2RGB)
            _, probs = rec.predict_emotions(img, logits=False)
            got = fct.decide(np.asarray(probs).reshape(-1), names)
            if got == fct.HOLD:
                hold += 1
            elif got == MAP[label]:
                correct += 1
            else:
                wrong += 1
        decided = correct + wrong
        result[name] = {"n": len(rows), "correct": correct, "wrong": wrong, "hold": hold,
                        "accuracy_on_decided": round(correct / max(decided, 1), 3)}
        print(name, result[name], flush=True)

    out = {"variants": result, "note": "공개 성인 얼굴 사진 기준. 사진 이름과 얼굴 이미지는 기록하지 않음."}
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "expression_robustness.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved logs/expression_robustness.json")


main()
