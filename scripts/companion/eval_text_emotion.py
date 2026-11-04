"""텍스트 감정 정확도 측정 도구 (S11-2, 453 데이터).

입력: AI Hub 453 학습 CSV (열: 성별, 감정_대분류, 사람문장1, ...). 테스트 CSV에는 정답이 없어서 쓰지 않는다.
정답: 대화 한 건의 감정_대분류를 그 대화의 사람문장 각각에 붙인다 (문장 단위 근사치, 한계는 결과 파일에 적는다).
실행: .venv/Scripts/python.exe scripts/companion/eval_text_emotion.py --csv data/aihub_453/train.csv --dialogues 800
결과 파일에는 점수와 혼동 행렬만 남긴다. 발화 원문은 남기지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from datetime import date
from pathlib import Path

from app.core.config import get_settings
from app.services.companion.text_emotion_service import TextEmotionService

SPEECH_COLUMNS = (2, 4, 6, 8)  # 사람문장1~4 의 열 위치 (시스템응답 열은 건너뜀)


def macro_f1(pairs: list[tuple[str, str]]) -> float:
    labels = sorted({g for g, _ in pairs} | {p for _, p in pairs})
    scores = []
    for lab in labels:
        tp = sum(1 for g, p in pairs if g == lab and p == lab)
        fp = sum(1 for g, p in pairs if g != lab and p == lab)
        fn = sum(1 for g, p in pairs if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def run(csv_path: Path, dialogues: int, seed: int, out_dir: Path, mode: str = "sentence") -> dict:
    model_labels = set(json.loads((Path(get_settings().model_base_path) / get_settings().companion_text_emotion_subdir
                                   / "kcelectra-korean-emotion" / "config.json").read_text(encoding="utf-8"))["id2label"].values())
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))[1:]
    random.Random(seed).shuffle(rows)
    rows = rows[:dialogues]

    svc = TextEmotionService()
    pairs: list[tuple[str, str]] = []
    gold_unknown = Counter()
    for row in rows:
        gold = row[1]
        if gold not in model_labels:
            gold_unknown[gold] += 1
            continue
        preds = []
        for col in SPEECH_COLUMNS:
            text = row[col].strip() if col < len(row) else ""
            if not text:
                continue
            preds.append(svc.classify(text).label)
        if mode == "dialogue":
            # 대화 안 문장 예측의 최빈값을 대화 예측으로 쓴다 (동률이면 먼저 나온 라벨)
            if preds:
                pairs.append((gold, Counter(preds).most_common(1)[0][0]))
        else:
            pairs.extend((gold, p) for p in preds)

    correct = sum(1 for g, p in pairs if g == p)
    per_label = {}
    for lab in sorted(model_labels):
        total = sum(1 for g, _ in pairs if g == lab)
        hit = sum(1 for g, p in pairs if g == lab and p == lab)
        per_label[lab] = {"n": total, "recall": round(hit / total, 4) if total else None}
    confusion = Counter(f"{g}->{p}" for g, p in pairs if g != p)

    summary = {
        "date": date.today().isoformat(),
        "dataset": "AI Hub 453 train.csv (정답 있음)",
        "unit": "대화 (최빈 문장 라벨)" if mode == "dialogue" else "문장",
        "dialogues_sampled": len(rows),
        "scored_items": len(pairs),
        "accuracy": round(correct / len(pairs), 4) if pairs else None,
        "macro_f1": round(macro_f1(pairs), 4) if pairs else None,
        "per_label": per_label,
        "top_confusions": confusion.most_common(10),
        "gold_label_not_in_model": dict(gold_unknown),
        "model": "GGARA02/kcelectra-korean-emotion",
        "limits": [
            "정답은 대화 단위 감정이라 문장 단위와 다를 수 있다",
            "GGARA02는 AI Hub 86으로 학습되었다. 453은 같은 계열 데이터여서 점수가 높게 나올 수 있다",
            "어르신 전용 데이터가 아니다. 107 음성 대화와 다를 수 있다",
        ],
        "note": "발화 원문은 저장하지 않음. 점수와 혼동 집계만 기록.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "-dialogue" if mode == "dialogue" else ""
    (out_dir / f"{summary['date']}-text-emotion-453{suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--dialogues", default=800, type=int)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--mode", default="sentence", choices=["sentence", "dialogue"])
    ap.add_argument("--out", default=Path("docs/eval-results"), type=Path)
    args = ap.parse_args()
    print(json.dumps(run(args.csv, args.dialogues, args.seed, args.out, args.mode), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
