"""음성 인식 정확도(CER) 재측정 도구 (S3-7).

입력: 매니페스트 CSV. 열: audio_path, reference  (audio_path 는 절대 경로 또는 manifest 기준 상대 경로)
실행: .venv/Scripts/python.exe scripts/companion/eval_stt.py --manifest <경로.csv> --out docs/eval-results/
데이터는 저장소 밖에 두고 경로만 넘긴다. 결과 파일에는 파일 이름과 점수만 남기고 발화 원문은 남기지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import wave
from datetime import date
from pathlib import Path


def _normalize(text: str) -> str:
    """비교 전 정규화: 공백과 문장 부호를 뺀다. 정답 전사 규칙(S3-2)에 맞춰 필요하면 여기서 바꾼다."""
    return re.sub(r"[\s\.,!?'\"~…\-]", "", text)


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def run(manifest: Path, out_dir: Path, engine: str = "sensevoice") -> dict:
    if engine == "whisper":
        from app.services.companion.stt_service import stt_service as svc
    else:
        from app.services.companion.sensevoice_service import sensevoice_service as svc

    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    scores: list[float] = []
    times: list[int] = []
    by_length: dict[str, list[float]] = {"짧은 말 (4초 미만)": [], "보통 (4~8초)": [], "긴 말 (8초 이상)": []}
    for row in rows:
        audio = Path(row["audio_path"])
        if not audio.is_absolute():
            audio = manifest.parent / audio
        result = svc.transcribe(audio.read_bytes())
        score = cer(row["reference"], result.text)
        scores.append(score)
        times.append(result.elapsed_ms)
        with wave.open(str(audio), "rb") as w:
            seconds = w.getnframes() / w.getframerate()
        bucket = "짧은 말 (4초 미만)" if seconds < 4 else "보통 (4~8초)" if seconds < 8 else "긴 말 (8초 이상)"
        by_length[bucket].append(score)
    summary = {
        "date": date.today().isoformat(),
        "model": svc.model_dir().name,
        "samples": len(scores),
        "mean_cer": round(sum(scores) / len(scores), 4) if scores else None,
        "mean_elapsed_ms": int(sum(times) / len(times)) if times else None,
        "by_length": {k: {"samples": len(v), "mean_cer": round(sum(v) / len(v), 4) if v else None} for k, v in by_length.items()},
        "note": "발화 원문은 저장하지 않음. 점수만 기록.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{summary['date']}-stt-cer-{engine}.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", default=Path("docs/eval-results"), type=Path)
    ap.add_argument("--engine", default="sensevoice", choices=["sensevoice", "whisper"])
    args = ap.parse_args()
    t0 = time.monotonic()
    summary = run(args.manifest, args.out, args.engine)
    print(json.dumps(summary, ensure_ascii=False, indent=2), f"\n({time.monotonic() - t0:.1f}s)")


if __name__ == "__main__":
    main()
