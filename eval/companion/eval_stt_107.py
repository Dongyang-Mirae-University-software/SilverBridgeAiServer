"""어르신 음성(AI Hub 107 샘플 100개)으로 STT 모델의 오류율을 잰다.

지표
- CER(글자 오류율): 한국어에 적합한 기본 지표. 공백·문장부호는 빼고 잰다.
- WER(어절 오류율): 참고용.

실행 (저장소 루트에서):
    venv\\Scripts\\python.exe -u eval\\eval_stt_107.py seastar105/whisper-medium-ko-zeroth
결과: logs\\stt_107_<모델이름>.json (음성별 정답·인식 결과 포함)

주의: 정답은 AI Hub 107 샘플 JSON의 stt 값이다. 이 값이 사람이 교정한 전사인지
자동 인식 결과인지는 데이터 설명서로 확인해야 한다.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트
DATA = ROOT / "data" / "aihub_107_sample"
LOGS = ROOT / "logs"
CACHE = ROOT / "data" / "models_cache"
PUNCT = re.compile(r"[\s.,?!\"'~…·\-_()\[\]{}:;]+")


def norm(s: str) -> str:
    return PUNCT.sub("", s)


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def rate(refs: list[str], hyps: list[str], unit: str) -> float:
    errs = 0
    total = 0
    for r, h in zip(refs, hyps):
        rt = list(r) if unit == "char" else r.split()
        ht = list(h) if unit == "char" else h.split()
        errs += edit_distance(rt, ht)
        total += len(rt)
    return errs / max(total, 1)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    model_id = args[0] if args else "seastar105/whisper-medium-ko-zeroth"
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    asr = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=-1,  # CPU
        model_kwargs={"cache_dir": str(CACHE)},
    )  # 한국어 파인튜닝 모델이므로 언어 옵션은 주지 않는다 (생성 설정 호환 문제)
    rows = []
    start = time.time()
    if "--quick" in sys.argv:
        meta = meta[:3]
    for i, m in enumerate(meta, 1):
        audio, sr = sf.read(str(DATA / m["wav"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        out = asr({"array": audio, "sampling_rate": sr})
        hyp = out["text"].strip()
        rows.append({"wav": m["wav"], "ref": m["stt_ref"], "hyp": hyp, "age": m["age"], "gender": m["gender"]})
        if i % 10 == 0:
            print(f"  {i}/{len(meta)} ({time.time() - start:.0f}s)", flush=True)
    elapsed = time.time() - start

    refs = [norm(r["ref"]) for r in rows]
    hyps = [norm(r["hyp"]) for r in rows]
    result = {
        "model": model_id,
        "n": len(rows),
        "cer": round(rate(refs, hyps, "char"), 4),
        "wer_reference_only": round(rate([r["ref"] for r in rows], [r["hyp"] for r in rows], "word"), 4),
        "seconds_total": round(elapsed, 1),
        "seconds_per_clip": round(elapsed / max(len(rows), 1), 2),
        "device": "cpu",
        "rows": rows,
    }
    LOGS.mkdir(exist_ok=True)
    name = model_id.replace("/", "__")
    out = LOGS / f"stt_107_{name}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"CER {result['cer']} | 어절 WER {result['wer_reference_only']} | {elapsed:.0f}s ({result['seconds_per_clip']}s/건)")
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
