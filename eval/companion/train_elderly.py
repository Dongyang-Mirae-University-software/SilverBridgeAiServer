"""노년층 데이터로 GGARA02 텍스트 감정 모델을 추가 학습한다 (계획서 Phase 5, 사용자 결정 B안).

데이터
- 학습: AI Hub 86 감성 대화 말뭉치 중 '연령=노년' 학습 데이터 (약 14,587행)
- 검증(에폭마다 점검, 학습 중단 시점 결정용): 같은 방식의 노년 검증 데이터 (약 2,053행)
- 최종 평가(학습에 쓰지 않음): AI Hub 453 train (10,150행), 우리 말투 평가셋 (50문장)

입력은 어르신 발화(사람문장1~4)를 이어 붙인 텍스트다. 라벨은 감정_대분류 6개다.
학습된 모델은 models/elderly_ft/ 에 저장한다 (models/ 는 git 제외).

실행 (저장소 루트에서, 장시간 작업이므로 백그라운드 권장):
    venv\\Scripts\\python.exe -u eval\\train_elderly.py
결과: logs\\train_elderly_result.json
"""

from __future__ import annotations

import csv
import io
import json
import random
import time
from collections import Counter
from pathlib import Path

import openpyxl
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트
DATA86 = ROOT / "data" / "aihub_86"
DATA453 = ROOT / "data" / "aihub_453" / "train.csv"
EVAL50 = ROOT / "eval" / "elderly_text_set.csv"
LOGS = ROOT / "logs"
CACHE = ROOT / "data" / "models_cache"
OUT = ROOT / "models" / "elderly_ft"
BASE = "GGARA02/kcelectra-korean-emotion"
UTT = ["사람문장1", "사람문장2", "사람문장3", "사람문장4"]
EPOCHS = 2
BATCH = 16
LR = 2e-5
MAX_LEN = 128
SEED = 0


def load86(path: Path) -> list[tuple[str, str]]:
    """86 엑셀에서 노년 행만 (텍스트, 대분류 라벨)으로 읽는다."""
    ws = openpyxl.load_workbook(path, read_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    head = rows[0]
    age_i, lab_i = head.index("연령"), head.index("감정_대분류")
    utt_i = [head.index(c) for c in UTT if c in head]  # 86 데이터는 사람문장1~3 만 있다
    out = []
    for r in rows[1:]:
        if not r or r[age_i] != "노년" or not r[lab_i]:
            continue
        text = " ".join(str(r[i]).strip() for i in utt_i if r[i] and str(r[i]).strip())
        if text:
            out.append((text, r[lab_i]))
    return out


def load453(path: Path) -> list[tuple[str, str]]:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp949"):
        try:
            rows = list(csv.DictReader(io.StringIO(raw.decode(enc))))
            break
        except UnicodeDecodeError:
            continue
    out = []
    for r in rows:
        if not r.get("감정_대분류"):
            continue
        text = " ".join(r[c].strip() for c in UTT if r.get(c, "").strip())
        if text:
            out.append((text, r["감정_대분류"]))
    return out


def load50(path: Path) -> list[tuple[str, str]]:
    with path.open(encoding="utf-8") as f:
        return [(r["text"], r["label"]) for r in csv.DictReader(f) if r["label"] != "애매"]


@torch.no_grad()
def evaluate(model, tok, data, id2label, label2id) -> dict:
    model.eval()
    preds, gold = [], []
    for i in range(0, len(data), 64):
        chunk = data[i : i + 64]
        batch = tok([t for t, _ in chunk], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
        idx = model(**batch).logits.argmax(dim=-1).tolist()
        preds += [id2label[j] for j in idx]
        gold += [y for _, y in chunk]
    labels = sorted(set(gold))
    return {"n": len(gold), "accuracy": round(accuracy_score(gold, preds), 4),
            "macro_f1": round(f1_score(gold, preds, labels=labels, average="macro", zero_division=0), 4)}


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(8)

    train = load86(DATA86 / "감성대화말뭉치(최종데이터)_Training.xlsx")
    valid = load86(DATA86 / "감성대화말뭉치(최종데이터)_Validation.xlsx")
    test453 = load453(DATA453)
    test50 = load50(EVAL50)
    print(f"학습 {len(train)} / 검증 {len(valid)} / 평가 453 {len(test453)} / 평가 50 {len(test50)}", flush=True)
    print("학습 라벨 분포:", dict(Counter(y for _, y in train)), flush=True)

    tok = AutoTokenizer.from_pretrained(BASE, cache_dir=str(CACHE))
    model = AutoModelForSequenceClassification.from_pretrained(BASE, cache_dir=str(CACHE))
    id2label = model.config.id2label
    label2id = {v: k for k, v in id2label.items()}
    base_453 = evaluate(model, tok, test453, id2label, label2id)
    base_50 = evaluate(model, tok, test50, id2label, label2id)
    print("학습 전 453:", base_453, "50문장:", base_50, flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * ((len(train) + BATCH - 1) // BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    loader = DataLoader(train, batch_size=BATCH, shuffle=True, collate_fn=lambda b: b)

    history = []
    best_val = -1.0
    start = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        for step, batch_items in enumerate(loader, 1):
            enc = tok([t for t, _ in batch_items], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
            y = torch.tensor([label2id[lab] for _, lab in batch_items])
            loss = model(**enc, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            if step % 100 == 0:
                print(f"  ep{ep} step {step}/{len(loader)} loss {loss.item():.3f} ({time.time() - start:.0f}s)", flush=True)
        val = evaluate(model, tok, valid, id2label, label2id)
        history.append({"epoch": ep, "valid": val})
        print(f"에폭 {ep} 검증: {val}", flush=True)
        if val["accuracy"] > best_val:
            best_val = val["accuracy"]
            OUT.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUT)
            tok.save_pretrained(OUT)
            print("  -> 최고 기록, 저장", flush=True)

    best = AutoModelForSequenceClassification.from_pretrained(OUT).eval()
    after_453 = evaluate(best, tok, test453, id2label, label2id)
    after_50 = evaluate(best, tok, test50, id2label, label2id)
    result = {
        "base": BASE, "train_n": len(train), "valid_n": len(valid),
        "epochs": EPOCHS, "batch": BATCH, "lr": LR, "seconds": round(time.time() - start, 1),
        "before": {"ai_hub_453": base_453, "elderly_50": base_50},
        "after": {"ai_hub_453": after_453, "elderly_50": after_50},
        "valid_history": history,
        "note": "453 과 50문장은 학습에 쓰지 않은 평가 데이터. 50문장 라벨은 임시 값.",
    }
    LOGS.mkdir(exist_ok=True)
    (LOGS / "train_elderly_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("학습 후 453:", after_453, "50문장:", after_50, flush=True)
    print("저장: logs\\train_elderly_result.json", flush=True)


if __name__ == "__main__":
    main()
