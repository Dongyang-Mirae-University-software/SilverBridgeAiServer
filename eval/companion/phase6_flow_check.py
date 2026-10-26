"""Phase 6 검증: 말투 평가셋 50문장을 실제 흐름과 같은 순서로 통과시킨다.

흐름: 텍스트 감정(GGARA02) -> 외로움 규칙(loneliness_rule) -> 결과 합치기(fusion, 표정 없음)
기대 결과(검증 기준)
- 라벨이 기쁨이고 외로움 신호 없음 -> 기쁨
- 부정 감정(분노·슬픔·불안·상처·당황)이고 확신 0.6 이상 -> 확인 필요
- 외로움 신호가 있고 모델이 부정이 아님 -> 확인 필요
- 확신 0.6 미만 -> 판단 불가
- 애매 문장 -> 판단 보류 또는 판단 불가 (확인 필요가 나오면 과잉 표시)
결과: logs/phase6_flow_check.json, 표는 화면에 출력
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트
sys.path.insert(0, str(ROOT / "app" / "companion"))
from fusion import Signal, fuse, is_ambiguous  # noqa: E402
from loneliness_rule import loneliness_signal  # noqa: E402

CACHE = ROOT / "data" / "models_cache"
MODEL = "GGARA02/kcelectra-korean-emotion"
SET = ROOT / "eval" / "elderly_text_set.csv"
THRESH = 0.6


def final_label(text: str, label: str, score: float) -> tuple[str, str]:
    sig = Signal(label, score)
    amb = is_ambiguous(text)
    res = fuse(sig, None, ambiguous=amb)
    if res["label"] in ("기쁨", "평온", "판단 보류") and not amb and loneliness_signal(text):
        return "확인 필요", "외로움·단절 표현"
    return res["label"], res["reason"]


def main() -> None:
    rows = list(csv.DictReader(SET.open(encoding="utf-8")))
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=str(CACHE))
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, cache_dir=str(CACHE)).eval()
    id2 = model.config.id2label
    out = []
    with torch.no_grad():
        for r in rows:
            batch = tok([r["text"]], truncation=True, max_length=128, return_tensors="pt")
            p = torch.softmax(model(**batch).logits, dim=-1)[0]
            i = int(torch.argmax(p))
            lab, score = id2[i], float(p[i])
            final, reason = final_label(r["text"], lab, score)
            out.append({"text": r["text"], "gold": r["label"], "model": lab, "conf": round(score, 3),
                        "final": final, "reason": reason})
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "phase6_flow_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("최종 라벨 분포:", dict(Counter(o["final"] for o in out)))
    labeled = [o for o in out if o["gold"] != "애매"]
    flagged_negative = [o for o in labeled if o["gold"] in ("슬픔", "분노", "불안", "상처", "당황")]
    caught = sum(o["final"] == "확인 필요" for o in flagged_negative)
    print(f"부정 감정 문장 {len(flagged_negative)}개 중 '확인 필요'로 잡힌 수: {caught}")
    amb = [o for o in out if o["gold"] == "애매"]
    print("애매 문장 결과:", [(o["final"], o["conf"]) for o in amb])
    missed = [o for o in flagged_negative if o["final"] not in ("확인 필요",)]
    print("확인 필요로 잡히지 않은 부정 문장:")
    for o in missed:
        print(f"  [{o['gold']} -> {o['final']} {o['conf']}] {o['text']}")


if __name__ == "__main__":
    main()
