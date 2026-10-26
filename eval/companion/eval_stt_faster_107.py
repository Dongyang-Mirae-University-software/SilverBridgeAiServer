"""faster-whisper large-v3 (MIT)로 같은 107 샘플 100건을 측정한다 (STT 비교, 계획서 Phase 2).

eval_stt_107.py 와 같은 음성, 같은 정답, 같은 지표(CER)를 쓰므로 두 모델을 직접 비교할 수 있다.

실행 (저장소 루트에서, 백그라운드 권장):
    venv\\Scripts\\python.exe -u eval\\eval_stt_faster_107.py
결과: logs\\stt_107_faster-whisper-large-v3.json
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parents[2]  # 저장소 루트
DATA = ROOT / "data" / "aihub_107_sample"
LOGS = ROOT / "logs"
CACHE = ROOT / "data" / "models_cache"
MODEL = "large-v3"
PUNCT = re.compile(r"[\s.,?!\"'~…·\-_()\[\]{}:;]+")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_stt_107 import edit_distance, rate  # noqa: E402  (같은 지표 함수 재사용)


def norm(s: str) -> str:
    return PUNCT.sub("", s)


def main() -> None:
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    model = WhisperModel(MODEL, device="cpu", compute_type="int8", download_root=str(CACHE))
    rows = []
    start = time.time()
    for i, m in enumerate(meta, 1):
        audio, sr = sf.read(str(DATA / m["wav"]), dtype="float32")  # 16kHz 배열로 넘긴다 (PyAV 디코딩 오류 회피)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        segments, _ = model.transcribe(audio, language="ko", beam_size=5)
        hyp = " ".join(seg.text.strip() for seg in segments).strip()
        rows.append({"wav": m["wav"], "ref": m["stt_ref"], "hyp": hyp, "age": m["age"], "gender": m["gender"]})
        if i % 10 == 0:
            print(f"  {i}/{len(meta)} ({time.time() - start:.0f}s)", flush=True)
    elapsed = time.time() - start

    refs = [norm(r["ref"]) for r in rows]
    hyps = [norm(r["hyp"]) for r in rows]
    result = {
        "model": f"faster-whisper {MODEL} (int8, cpu)",
        "n": len(rows),
        "cer": round(rate(refs, hyps, "char"), 4),
        "wer_reference_only": round(rate([r["ref"] for r in rows], [r["hyp"] for r in rows], "word"), 4),
        "seconds_total": round(elapsed, 1),
        "seconds_per_clip": round(elapsed / max(len(rows), 1), 2),
        "rows": rows,
    }
    LOGS.mkdir(exist_ok=True)
    out = LOGS / "stt_107_faster-whisper-large-v3.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"CER {result['cer']} | 어절 WER {result['wer_reference_only']} | {elapsed:.0f}s ({result['seconds_per_clip']}s/건)", flush=True)
    print(f"저장: {out}", flush=True)


if __name__ == "__main__":
    main()
