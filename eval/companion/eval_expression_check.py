"""표정 인식이 실제 얼굴 사진에서 되는지 확인한다 (공개 데이터, 라벨별 10장씩 무작위 60장).

- 데이터: Facial Emotion Recognition (Roboflow, CC BY 4.0) 의 test 폴더. 성인 얼굴이라 어르신 정확도를 대신하지 않는다.
- 판단: companion/expression_check.classify_jpeg (얼굴 검출 + HSEmotion + face_capture_test.decide 규칙)
- 기록: 숫자와 라벨만 logs/expression_check_sample.json 에 쓴다. 사진 이름은 쓰지 않는다.

실행 (저장소 루트에서):
    set FACE_TEST_DIR=<Facial Emotion Recognition.v2i.folder 의 test 폴더 경로>
    venv\\Scripts\\python.exe -u eval\\eval_expression_check.py
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트
sys.path.insert(0, str(ROOT / "app" / "companion"))

import expression_check  # noqa: E402
import face_capture_test as fct  # noqa: E402

# 데이터 라벨 → 모델 라벨
MAP = {"angry": "Anger", "fear": "Fear", "happy": "Happiness", "neutral": "Neutral", "sad": "Sadness", "surprise": "Surprise"}
PER_LABEL = 10
SEED = 0

test_dir = Path(os.environ["FACE_TEST_DIR"])
rng = random.Random(SEED)
rows = []
for label in MAP:
    files = sorted(p for p in (test_dir / label).glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for p in rng.sample(files, min(PER_LABEL, len(files))):
        rows.append((label, p))

counts = Counter()
per_label: dict[str, Counter] = {label: Counter() for label in MAP}
# 표정 모델만 따로 보기: 이 사진들은 이미 얼굴만 잘라 둔 것이라, 사진 전체를 얼굴로 보고 읽는다 (검출기 없이)
whole = Counter()
rec, names = expression_check._recognizer()
import cv2  # noqa: E402
import numpy as np  # noqa: E402
for label, p in rows:
    raw = p.read_bytes()
    img = cv2.cvtColor(cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    _, probs = rec.predict_emotions(img, logits=False)
    pv = np.asarray(probs).reshape(-1)
    whole_label = fct.decide(pv, names)
    whole['total'] += 1
    whole['hold' if whole_label == fct.HOLD else ('correct' if whole_label == MAP[label] else 'wrong')] += 1
    out = expression_check.classify_jpeg(raw)
    per_label[label]["total"] += 1
    if not out["face"]:
        counts["no_face"] += 1
        per_label[label]["no_face"] += 1
        continue
    counts["face"] += 1
    per_label[label]["face"] += 1
    got = out["expression"]
    if got == fct.HOLD:
        counts["hold"] += 1
        per_label[label]["hold"] += 1
    elif got == MAP[label]:
        counts["correct"] += 1
        per_label[label]["correct"] += 1
    else:
        counts["wrong"] += 1
        per_label[label]["wrong"] += 1

total = len(rows)
decided = counts["correct"] + counts["wrong"]
result = {
    "n": total,
    "face_detected": counts["face"],
    "face_detection_rate": round(counts["face"] / max(total, 1), 3),
    "decided": decided,
    "hold_판단보류": counts["hold"],
    "accuracy_on_decided": round(counts["correct"] / max(decided, 1), 3),
    "accuracy_all_faces": round(counts["correct"] / max(counts["face"], 1), 3),
    "whole_image_expression_only": {"n": whole["total"], "correct": whole["correct"], "wrong": whole["wrong"], "hold": whole["hold"],
        "accuracy_on_decided": round(whole["correct"] / max(whole["correct"] + whole["wrong"], 1), 3),
        "accuracy_all": round(whole["correct"] / max(whole["total"], 1), 3)},
    "per_label": {k: dict(v) for k, v in per_label.items()},
    "note": "사진 이름과 얼굴 이미지는 기록하지 않음. 성인 얼굴 공개 데이터 기준이며 어르신 정확도가 아님.",
}
out_dir = ROOT / "logs"
out_dir.mkdir(exist_ok=True)
(out_dir / "expression_check_sample.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=True, indent=2))
