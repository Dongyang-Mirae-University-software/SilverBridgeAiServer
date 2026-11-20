#!/usr/bin/env python3
"""답 생성 입력 만들기(2026-11-20) - v8 문항에서 재생에 필요한 칸만 JSONL 로 뽑는다.

채점 기준·참조 요점·출처는 넣지 않는다(답을 만드는 쪽이 기준을 보지 않게). 한 줄 = 한 문항:
  {"id", "area", "speaker", "context", "turns": [{"message"} | {"ui": {"field", "value"|"pick"}}, "route": {...}|"fail"],
   "reservation_fixture"}
DGX 에서 `python tools/eval/replay_offline.py run --set <이 파일> --router stub --out <저장소 밖> --summary <저장소 밖>` 로 돈다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

GEN_KEYS = ("id", "area", "speaker", "context", "turns", "reservation_fixture")


def gen_line(item: dict) -> dict:
    return {k: item[k] for k in GEN_KEYS if k in item}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True)
    ap.add_argument("--areas", default="", help="쉼표로 구분한 영역만(비우면 전부)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    data = json.loads(pathlib.Path(args.set).read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    want = set(filter(None, args.areas.split(",")))
    lines = [gen_line(it) for it in items if not want or it.get("area") in want]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines), encoding="utf-8")
    print(f"답 생성 입력 {len(lines)}줄")
    return 0


if __name__ == "__main__":
    sys.exit(main())
