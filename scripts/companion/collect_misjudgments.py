"""오판 샘플 수집 (S5-6). 공개 데이터셋(AI Hub 453 학습 CSV)에서 텍스트 감정 모델이 틀린 문장을 모은다.

실행: .venv/Scripts/python.exe -m scripts.companion.collect_misjudgments --csv data/aihub_453/train.csv --dialogues 800 --limit 200
결과: data/misjudgment_samples.json (data/ 아래라 Git에 올라가지 않는다). 공개 데이터의 문장이므로 원문을 담는다.
정답은 대화 단위 감정이라 문장 오판과 다를 수 있다 (결과 파일 limits 참고).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from app.core.config import get_settings
from app.services.companion.text_emotion_service import TextEmotionService
from scripts.companion.eval_text_emotion import SPEECH_COLUMNS


def collect(csv_path: Path, dialogues: int, limit: int, seed: int, out: Path) -> dict:
    model_labels = set(json.loads((Path(get_settings().model_base_path) / get_settings().companion_text_emotion_subdir
                                   / "kcelectra-korean-emotion" / "config.json").read_text(encoding="utf-8"))["id2label"].values())
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))[1:]
    random.Random(seed).shuffle(rows)

    svc = TextEmotionService()
    samples = []
    for dialogue_no, row in enumerate(rows[:dialogues]):
        gold = row[1]
        if gold not in model_labels:
            continue
        for col in SPEECH_COLUMNS:
            text = row[col].strip() if col < len(row) else ""
            if not text:
                continue
            pred = svc.classify(text).label
            if pred != gold:
                samples.append({"dialogueNo": dialogue_no, "text": text, "gold": gold, "pred": pred})

    # 한 대화에서 많이 나오지 않도록 섞은 뒤 상한까지만 담는다
    random.Random(seed).shuffle(samples)
    picked = samples[:limit]
    payload = {
        "source": "AI Hub 453 train.csv (공개 데이터)",
        "dialogues_scanned": min(dialogues, len(rows)),
        "misjudged_total": len(samples),
        "saved": len(picked),
        "note": "정답은 대화 단위 감정. 문장 자체가 틀린 것이 아닐 수 있다. 서비스 데이터가 아니므로 data/ 에만 둔다.",
        "samples": picked,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--dialogues", default=800, type=int)
    ap.add_argument("--limit", default=200, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--out", default=Path("data/misjudgment_samples.json"), type=Path)
    args = ap.parse_args()
    result = collect(args.csv, args.dialogues, args.limit, args.seed, args.out)
    print(f"오판 {result['misjudged_total']}건 중 {result['saved']}건 저장: {args.out}")


if __name__ == "__main__":
    main()
