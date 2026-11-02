"""평가 입력 도구 (S8-7). 매니페스트 CSV를 읽어 평가 실행 목록(JSON)을 만든다.

입력 CSV 열: audio_path, reference (음성 파일 경로와 정답 글)
- 음성 파일이 없거나 정답이 비어 있는 줄은 빼고, 빠진 이유를 함께 적는다.
- 결과는 data/ 아래에 둔다 (음성 원본과 정답 글은 Git에 올리지 않는다).

실행: .venv/Scripts/python.exe -m scripts.companion.build_eval_set --manifest data/eval_manifest_107.csv --out data/eval_set_107.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build(manifest: Path, out: Path, base_dir: Path) -> dict:
    cases, skipped = [], []
    with manifest.open(encoding="utf-8-sig", newline="") as fh:
        for no, row in enumerate(csv.DictReader(fh), start=1):
            audio = (base_dir / row.get("audio_path", "")).resolve()
            reference = (row.get("reference") or "").strip()
            if not audio.is_file():
                skipped.append({"line": no, "reason": "음성 파일 없음"})
            elif not reference:
                skipped.append({"line": no, "reason": "정답 글 비어 있음"})
            else:
                cases.append({"id": f"case-{no:03d}", "audio": str(audio), "reference": reference})
    payload = {"cases": len(cases), "skipped": skipped, "items": cases}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--base-dir", default=Path("data"), type=Path, help="audio_path 의 기준 폴더")
    ap.add_argument("--out", default=Path("data/eval_set.json"), type=Path)
    args = ap.parse_args()
    result = build(args.manifest, args.out, args.base_dir)
    print(f"평가 입력 {result['cases']}건, 제외 {len(result['skipped'])}건: {args.out}")


if __name__ == "__main__":
    main()
