"""GGARA02 텍스트 감정 모델을 AI Hub 453(노년층 감성 분류) 데이터로 평가한다.

평가 원칙
- 453 은 GGARA02 학습 데이터(AI Hub 감성대화 말뭉치 86)와 출처가 다르므로 독립 평가셋이다.
- 453 train.csv 에는 정답이 있고 test.csv 에는 없으므로 train.csv 전체를 평가에 쓴다.
- 입력은 어르신 발화(사람문장1~4)를 이어 붙인 텍스트다. 시스템 응답은 넣지 않는다.

실행 (저장소 루트에서):
    venv\\Scripts\\python.exe eval\\eval_text_emotion_453.py
결과는 logs\\eval_gg_453.json 에 저장한다 (logs/ 는 git 제외).
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트
DATA = ROOT / "data" / "aihub_453" / "train.csv"
LOGS = ROOT / "logs"
CACHE = ROOT / "data" / "models_cache"
MODEL_ID = "GGARA02/kcelectra-korean-emotion"
UTTERANCE_COLS = ["사람문장1", "사람문장2", "사람문장3", "사람문장4"]
LABEL_COL = "감정_대분류"
MAX_LEN = 128
BATCH = 32


def read_rows(path: Path) -> list[dict]:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp949"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    import io

    return list(csv.DictReader(io.StringIO(text)))


def elder_text(row: dict) -> str:
    parts = [row[c].strip() for c in UTTERANCE_COLS if row.get(c, "").strip()]
    return " ".join(parts)


def main() -> None:
    torch.set_num_threads(max(1, torch.get_num_threads()))
    rows = read_rows(DATA)
    rows = [r for r in rows if r.get(LABEL_COL) and elder_text(r)]
    texts = [elder_text(r) for r in rows]
    gold = [r[LABEL_COL] for r in rows]
    print(f"평가 대상: {len(rows)}행, 정답 분포: {dict(Counter(gold))}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=str(CACHE))
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, cache_dir=str(CACHE))
    model.eval()
    id2label = model.config.id2label
    labels = [id2label[i] for i in range(len(id2label))]

    preds: list[str] = []
    probs_max: list[float] = []
    start = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            batch = tok(
                texts[i : i + BATCH],
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            logits = model(**batch).logits
            p = torch.softmax(logits, dim=-1)
            top = p.argmax(dim=-1)
            preds += [id2label[int(t)] for t in top]
            probs_max += [float(x) for x in p.max(dim=-1).values]
            if (i // BATCH) % 50 == 0:
                print(f"  {i}/{len(texts)} 진행 ({time.time() - start:.0f}s)")
    elapsed = time.time() - start

    acc = accuracy_score(gold, preds)
    macro_f1 = f1_score(gold, preds, labels=labels, average="macro")
    report = classification_report(gold, preds, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(gold, preds, labels=labels).tolist()

    result = {
        "model": MODEL_ID,
        "eval_set": "AI Hub 453 노년층 감성 분류 train.csv (GGARA02 학습 데이터와 다른 출처)",
        "n": len(rows),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "labels": labels,
        "per_label": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in report.items() if k in labels},
        "confusion_matrix_rows_gold_cols_pred": cm,
        "seconds": round(elapsed, 1),
        "max_len": MAX_LEN,
        "input": "사람문장1~4 연결",
    }
    LOGS.mkdir(exist_ok=True)
    out = LOGS / "eval_gg_453.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n정확도 {acc:.4f} / macro F1 {macro_f1:.4f} / {len(rows)}행 / {elapsed:.0f}초")
    print("라벨별 F1:", {k: round(report[k]["f1-score"], 3) for k in labels})
    print(f"결과 저장: {out}")


if __name__ == "__main__":
    main()
