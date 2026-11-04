"""표정 정확도 측정 도구 (S4-7).

입력 폴더 구조 (라벨 이름 폴더 아래에 사진을 둔다. 사진은 Git에 올리지 않는다):
    <root>/Happiness/*.jpg
    <root>/Sadness/*.jpg
    ...
라벨 이름은 HSEmotion 8개 중 하나여야 한다: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise.

실행: .venv/Scripts/python.exe -m scripts.companion.eval_face --root data/face_eval --out docs/eval-results
사진 한 장을 프레임 한 장으로 보고 표정 라벨을 맞춘다. 얼굴이 안 잡힌 사진은 따로 센다.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

from app.services.companion.face_service import FaceUnavailable, face_service

HSEMOTION_LABELS = ("Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def run(root: Path, out_dir: Path) -> dict:
    per_label = {}
    confusion: Counter = Counter()
    no_face = 0
    total = correct = 0
    for label_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        gold = label_dir.name
        if gold not in HSEMOTION_LABELS:
            continue
        hit = n = 0
        for photo in sorted(label_dir.iterdir()):
            if photo.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                result = face_service.analyze([photo.read_bytes()])
            except FaceUnavailable:
                no_face += 1
                continue
            if not result.usable or not result.label:
                no_face += 1
                continue
            n += 1
            total += 1
            if result.label == gold:
                hit += 1
                correct += 1
            else:
                confusion[f"{gold}->{result.label}"] += 1
        per_label[gold] = {"n": n, "recall": round(hit / n, 4) if n else None}

    summary = {
        "date": date.today().isoformat(),
        "dataset": "사용자 준비 표정 샘플 (폴더 라벨)",
        "scored": total,
        "accuracy": round(correct / total, 4) if total else None,
        "no_face_or_unusable": no_face,
        "per_label": per_label,
        "top_confusions": confusion.most_common(10),
        "model": "HSEmotion enet_b0_8_best_afew (EmotiEffLib 1.1.1) + YuNet",
        "limits": [
            "한 장을 한 프레임으로 본다. 실제 턴은 최대 5장을 본다",
            "얼굴이 안 잡힌 사진은 정확도에서 빠진다",
        ],
        "note": "사진 자체는 저장하지 않음. 점수와 혼동 집계만 기록.",
    }
    if total == 0:
        summary["status"] = "측정 불가: 라벨 폴더가 없거나 얼굴이 잡힌 사진이 없음"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{summary['date']}-face-emotion.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", default=Path("docs/eval-results"), type=Path)
    args = ap.parse_args()
    print(json.dumps(run(args.root, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
