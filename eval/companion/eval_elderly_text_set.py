"""어르신 말투 평가셋(50문장)으로 GGARA02 텍스트 감정을 측정한다. (계획서 Phase 4-A)

- 라벨이 있는 45문장: 정확도, macro F1, 혼동 행렬, 확신도 0.6 기준 적용 결과
- 애매 5문장(라벨 '애매'): 정확도에서 뺀다. 모델이 얼마나 확신하는지만 기록한다.
- 정답은 제작자(AI)가 붙인 라벨이다. 사용자님 검토 후 확정한다.

실행 (저장소 루트에서):
    venv\Scripts\python.exe -u eval\eval_elderly_text_set.py
결과: logs\eval_elderly_text_set.json
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트
SET = ROOT / "eval" / "elderly_text_set.csv"
LOGS = ROOT / "logs"
CACHE = ROOT / "data" / "models_cache"
MODEL = "GGARA02/kcelectra-korean-emotion"
THRESH = 0.6
LABELS = ["기쁨", "슬픔", "분노", "불안", "상처", "당황"]


def main() -> None:
    with SET.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=str(CACHE))
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, cache_dir=str(CACHE)).eval()
    id2 = model.config.id2label
    out = []
    with torch.no_grad():
        for r in rows:
            batch = tok([r["text"]], truncation=True, max_length=128, return_tensors="pt")
            p = torch.softmax(model(**batch).logits, dim=-1)[0]
            i = int(torch.argmax(p))
            out.append({"text": r["text"], "gold": r["label"], "pred": id2[i], "conf": round(float(p[i]), 3)})

    labeled = [o for o in out if o["gold"] != "애매"]
    gold = [o["gold"] for o in labeled]
    pred = [o["pred"] for o in labeled]
    acc = accuracy_score(gold, pred)
    f1 = f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0)
    cm = {g: Counter(p for gg, p in zip(gold, pred) if gg == g) for g in LABELS}
    kept = [o for o in labeled if o["conf"] >= THRESH]
    kept_acc = (sum(o["gold"] == o["pred"] for o in kept) / len(kept)) if kept else 0.0
    ambiguous = [o for o in out if o["gold"] == "애매"]
    result = {
        "labeled_n": len(labeled), "accuracy": round(acc, 4), "macro_f1": round(f1, 4),
        "confusion": {g: dict(c) for g, c in cm.items()},
        "threshold": THRESH, "kept_n": len(kept), "accuracy_on_kept": round(kept_acc, 4),
        "ambiguous_conf": [o["conf"] for o in ambiguous],
        "wrong": [o for o in labeled if o["gold"] != o["pred"]],
        "note": "정답 라벨은 제작자 AI가 붙인 것. 사용자 검토 전 임시 값.",
    }
    LOGS.mkdir(exist_ok=True)
    (LOGS / "eval_elderly_text_set.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"라벨 45문장 정확도 {acc:.4f} / macro F1 {f1:.4f}")
    print(f"확신도 {THRESH} 이상: {len(kept)}/{len(labeled)}문장, 그 중 정확도 {kept_acc:.4f}")
    print(f"애매 5문장 확신도: {result['ambiguous_conf']}")
    print("틀린 문장:")
    for w in result["wrong"]:
        print(f"  [{w['gold']} -> {w['pred']} {w['conf']}] {w['text']}")


if __name__ == "__main__":
    main()
