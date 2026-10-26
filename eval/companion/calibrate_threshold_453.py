"""확신도 기준 정하기: GGARA02 가 453 데이터에서 확신도별로 얼마나 맞히는지 본다.

결과: 확신도 구간(0.3, 0.4, ..., 0.9)마다 '이 구간에 들어온 비율'과 '그 구간의 정확도'를 기록한다.
기준(EMOTION_UNKNOWN_BELOW)을 고를 때 '정확도가 충분하면서 판단 불가가 너무 많지 않은' 값을 찾는다.

실행 (저장소 루트에서, 약 13분):
    venv\Scripts\python.exe -u eval\calibrate_threshold_453.py
결과: logs\calibrate_453.json
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트
DATA = ROOT / "data" / "aihub_453" / "train.csv"
LOGS = ROOT / "logs"
CACHE = ROOT / "data" / "models_cache"
MODEL = "GGARA02/kcelectra-korean-emotion"
COLS = ["사람문장1", "사람문장2", "사람문장3", "사람문장4"]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def read_rows(path: Path) -> list[dict]:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return list(csv.DictReader(io.StringIO(raw.decode(enc))))
        except UnicodeDecodeError:
            continue
    raise SystemExit("인코딩을 읽을 수 없습니다")


def main() -> None:
    rows = [r for r in read_rows(DATA) if r.get("감정_대분류") and any(r.get(c, "").strip() for c in COLS)]
    texts = [" ".join(r[c].strip() for c in COLS if r.get(c, "").strip()) for r in rows]
    gold = [r["감정_대분류"] for r in rows]
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=str(CACHE))
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, cache_dir=str(CACHE)).eval()
    id2 = model.config.id2label

    conf, correct = [], []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = tok(texts[i : i + 32], padding=True, truncation=True, max_length=128, return_tensors="pt")
            p = torch.softmax(model(**batch).logits, dim=-1)
            top_p, top_i = p.max(dim=-1)
            for j, (pp, ii) in enumerate(zip(top_p.tolist(), top_i.tolist())):
                conf.append(pp)
                correct.append(id2[ii] == gold[i + j])
            print(f"  {i + len(texts[i:i+32])}/{len(texts)}", flush=True)

    table = []
    n = len(conf)
    for t in THRESHOLDS:
        kept = [c for c, p in zip(correct, conf) if p >= t]
        table.append({
            "threshold": t,
            "coverage": round(len(kept) / n, 4),
            "accuracy_on_kept": round(sum(kept) / max(len(kept), 1), 4),
        })
    out = {"n": n, "overall_accuracy": round(sum(correct) / n, 4), "table": table}
    LOGS.mkdir(exist_ok=True)
    (LOGS / "calibrate_453.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in table:
        print(r, flush=True)


if __name__ == "__main__":
    main()
