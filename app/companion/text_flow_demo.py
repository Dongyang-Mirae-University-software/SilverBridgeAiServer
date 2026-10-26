"""텍스트만으로 정서 라벨이 어떻게 나오는지 본다 (표정은 제외, 2026-10-25).

흐름: 음성 인식 글자(STT 결과) → GGARA02 텍스트 감정 → fusion.fuse(텍스트, 표정=None)

실행 (companion-emotion 폴더에서):
    venv\Scripts\python.exe companion\text_flow_demo.py --n 20
결과: logs\text_flow_demo.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fusion import Signal, fuse  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "logs"
CACHE = ROOT / "data" / "models_cache"
EMO_MODEL = "GGARA02/kcelectra-korean-emotion"
STT_FILE = LOGS / "stt_107_seastar105__whisper-medium-ko-zeroth.json"


def text_signal(model, tok, text: str) -> Signal:
    with torch.no_grad():
        batch = tok([text], truncation=True, max_length=128, return_tensors="pt")
        p = torch.softmax(model(**batch).logits, dim=-1)[0]
    i = int(torch.argmax(p))
    return Signal(model.config.id2label[i], float(p[i]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    rows = json.loads(STT_FILE.read_text(encoding="utf-8"))["rows"]
    rows = [r for r in rows if r["hyp"].strip()][: args.n]

    tok = AutoTokenizer.from_pretrained(EMO_MODEL, cache_dir=str(CACHE))
    model = AutoModelForSequenceClassification.from_pretrained(EMO_MODEL, cache_dir=str(CACHE))
    model.eval()

    out = []
    for r in rows:
        sig = text_signal(model, tok, r["hyp"])
        res = fuse(sig, None)
        out.append({"stt_text": r["hyp"], "text_label": sig.label, "text_score": round(sig.score, 3),
                    "final": res["label"], "reason": res["reason"]})
    LOGS.mkdir(exist_ok=True)
    (LOGS / "text_flow_demo.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("최종 라벨 분포:", dict(Counter(o["final"] for o in out)))
    print("텍스트 감정 분포:", dict(Counter(o["text_label"] for o in out)))
    for o in out[:8]:
        print(f"  [{o['final']}] {o['text_label']} {o['text_score']} | {o['stt_text'][:40]}")


if __name__ == "__main__":
    main()
