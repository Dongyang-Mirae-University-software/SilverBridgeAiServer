#!/usr/bin/env python3
"""판정 입력 만들기(2026-11-20) - 판정 모델 3개 다수결용. 한 줄 = (문항, 회차):
  {"id", "run", "area", "speaker", "user_turns": [...], "criteria": [{"cid", "text", "met_example", "unmet_example"}],
   "reference_points": [...], "answer_ref": {"file", "id", "run"}, "answer": "<본문 - --embed-answers 일 때만>"}

답 본문은 저장소에 두지 않는다. 기본은 answer_ref(답 파일 경로·id·회차)만 적고, --embed-answers 를 주면 본문을 넣되
--out 이 저장소 밖일 때만 허용한다(DGX 판정 실행 직전에 만든다). 판정 기준은 "요점이 응답에 있는가"만 묻는다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def outside_repo(path: str) -> pathlib.Path:
    p = pathlib.Path(path).expanduser().resolve()
    if p == REPO_ROOT or REPO_ROOT in p.parents:
        raise SystemExit(f"답 본문이 든 파일은 저장소 안에 쓸 수 없습니다: {p}")
    return p


def user_turns(item: dict) -> list[str]:
    return [t["message"] for t in item.get("turns") or [] if t.get("message")]


def judge_lines(items: list[dict], answers: list[dict] | None, answers_file: str, embed: bool) -> list[dict]:
    by_id: dict[str, list[dict]] = {}
    for a in answers or []:
        by_id.setdefault(a["id"], []).append(a)
    out = []
    for it in items:
        if not it.get("judge_criteria"):
            continue
        runs = by_id.get(it["id"]) or ([{"id": it["id"], "run": 1}] if answers is None else [])
        for a in runs:
            line = {
                "id": it["id"], "run": a.get("run", 1), "area": it.get("area"), "speaker": it.get("speaker"),
                "user_turns": user_turns(it), "criteria": it["judge_criteria"],
                "reference_points": it.get("reference_points") or [],
                "answer_ref": {"file": answers_file, "id": it["id"], "run": a.get("run", 1)},
            }
            if embed:
                line["answer"] = a.get("reply") if "reply" in a else (a.get("turns") or [{}])[-1].get("reply")
            out.append(line)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", required=True)
    ap.add_argument("--answers", default="", help="replay_offline --out(답 본문, 저장소 밖)")
    ap.add_argument("--embed-answers", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    data = json.loads(pathlib.Path(args.set).read_text(encoding="utf-8"))
    items = data["items"] if isinstance(data, dict) else data
    answers = None
    if args.answers:
        answers = json.loads(outside_repo(args.answers).read_text(encoding="utf-8"))
    if args.embed_answers and not args.answers:
        raise SystemExit("--embed-answers 에는 --answers 가 필요합니다")
    out = outside_repo(args.out) if args.embed_answers else pathlib.Path(args.out)
    lines = judge_lines(items, answers, args.answers or "<answers.json>", args.embed_answers)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines), encoding="utf-8")
    print(f"판정 입력 {len(lines)}줄")
    return 0


if __name__ == "__main__":
    sys.exit(main())
