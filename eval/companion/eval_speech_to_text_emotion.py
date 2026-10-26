"""Phase 4-B: 음성 → 글자 → 텍스트 감정 파이프라인이 감정 판단을 얼마나 바꾸는지 본다.

107 샘플에는 감정 정답이 없다. 그래서 다음 두 가지를 비교한다.
- 정답 전사(stt_ref)를 GGARA02에 넣은 감정 라벨
- STT가 인식한 글자(hyp)를 GGARA02에 넣은 감정 라벨
두 라벨이 같은 비율(일치율)이 높을수록, 음성 인식 오류가 감정 판단을 덜 바꾼다는 뜻이다.

전제: eval_stt_107.py 를 먼저 실행해서 logs\\stt_107_<모델>.json 이 있어야 한다.
실행:
    venv\\Scripts\\python.exe -u eval\\eval_speech_to_text_emotion.py
결과: logs\\speech_to_text_emotion_107.json
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트
LOGS = ROOT / "logs"
CACHE = ROOT / "data" / "models_cache"
import sys
STT_FILE = LOGS / (sys.argv[1] if len(sys.argv) > 1 else "stt_107_seastar105__whisper-medium-ko-zeroth.json")
EMO_MODEL = "GGARA02/kcelectra-korean-emotion"


def predict(model, tok, texts: list[str]) -> list[str]:
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = tok(texts[i : i + 32], padding=True, truncation=True, max_length=128, return_tensors="pt")
            idx = model(**batch).logits.argmax(dim=-1).tolist()
            out += [model.config.id2label[j] for j in idx]
    return out


def main() -> None:
    rows = json.loads(STT_FILE.read_text(encoding="utf-8"))["rows"]
    rows = [r for r in rows if r["ref"].strip() and r["hyp"].strip()]
    tok = AutoTokenizer.from_pretrained(EMO_MODEL, cache_dir=str(CACHE))
    model = AutoModelForSequenceClassification.from_pretrained(EMO_MODEL, cache_dir=str(CACHE))
    model.eval()

    ref_labels = predict(model, tok, [r["ref"] for r in rows])
    hyp_labels = predict(model, tok, [r["hyp"] for r in rows])
    same = sum(a == b for a, b in zip(ref_labels, hyp_labels))
    result = {
        "n": len(rows),
        "agreement_ref_vs_stt": round(same / len(rows), 4),
        "ref_label_counts": dict(Counter(ref_labels)),
        "stt_label_counts": dict(Counter(hyp_labels)),
        "note": "107 샘플에는 감정 정답이 없어서 일치율만 본다 (정답 전사 기준 라벨 대비 STT 기준 라벨).",
        "examples_changed": [
            {"ref": r["ref"], "hyp": r["hyp"], "label_ref": a, "label_stt": b}
            for r, a, b in zip(rows, ref_labels, hyp_labels) if a != b
        ][:15],
    }
    out = LOGS / ("speech_to_text_emotion_107_" + STT_FILE.stem.split("_")[-1] + ".json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"일치율 {result['agreement_ref_vs_stt']} ({same}/{len(rows)})")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
