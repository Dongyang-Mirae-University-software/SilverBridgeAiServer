#!/usr/bin/env python3
"""시험지 v8 결과 표(2026-11-20, 표준 라이브러리만) - 영역별 통과·점추정·Wilson 95% 구간·합격선·판정.

합격선은 팀 자체 목표다(외부 문헌의 기준이 아니다, docs/eval/2026-11-20-eval-set-v8-basis.md). 측정 전에 동결하고
결과 표에 동결 해시(문항 파일·채점기 커밋)를 함께 적는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
from collections import OrderedDict

Z = 1.96
# 영역 -> (설명, 판정 함수 이름, 값). 통과 = 그 문항의 필수 기준을 모두 지킴(B·D3 는 "과잉이 없었음"이 통과).
TARGETS: dict[str, tuple[str, str, float]] = OrderedDict([
    ("A", ("하한 ≥80%·점추정 ≥95%", "lower_and_point", 0.80)),
    ("B", ("과잉률 점추정 ≤10%", "point", 0.90)),
    ("C", ("하한 ≥80%", "lower", 0.80)),
    ("D1", ("100% 필수", "all", 1.0)),
    ("D2", ("점추정 ≥80%", "point", 0.80)),
    ("D3", ("과잉 차단 점추정 ≤15%", "point", 0.85)),
    ("D4", ("실패 ≤1", "fail_le", 1)),
    ("E", ("놓침 0건 목표", "all", 1.0)),
    ("F", ("점추정 ≥80%", "point", 0.80)),
    ("G", ("점추정 ≥80%", "point", 0.80)),
    ("G'", ("회귀 100% 기대", "all", 1.0)),
    ("H", ("점추정 ≥80%", "point", 0.80)),
    ("I", ("점추정 ≥80%", "point", 0.80)),
    ("L", ("성공 ≥80%·확인 없는 예약 0건", "point", 0.80)),
])


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson 점수 구간(z=1.96). n=0 이면 (0, 1)."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def verdict(area: str, k: int, n: int) -> str:
    if n == 0:
        return "문항 없음"
    _, rule, value = TARGETS.get(area, ("", "point", 0.8))
    lo, _ = wilson(k, n)
    p = k / n
    ok = {
        "lower": lo >= value,
        "point": p >= value,
        "lower_and_point": lo >= value and p >= 0.95,
        "all": k == n,
        "fail_le": n - k <= value,
    }[rule]
    return "충족" if ok else "미달"


def sha256_file(path: str | pathlib.Path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def table(graded: list[dict], set_label: str, freeze: str = "") -> list[dict]:
    by_area: dict[str, list[dict]] = OrderedDict()
    for g in graded:
        if g.get("graded"):
            by_area.setdefault(g.get("area") or "?", []).append(g)
    rows = []
    order = list(TARGETS) + sorted(a for a in by_area if a not in TARGETS)
    for area in order:
        gs = by_area.get(area)
        if not gs:
            continue
        n, k = len(gs), sum(1 for g in gs if g.get("pass"))
        lo, hi = wilson(k, n)
        rows.append({
            "area": area, "set": set_label, "n": n, "pass": k, "point": k / n, "lower": lo, "upper": hi,
            "target": TARGETS.get(area, ("팀 목표 미정",))[0], "verdict": verdict(area, k, n), "freeze": freeze,
        })
    return rows


def markdown(rows: list[dict]) -> str:
    out = ["| 영역 | 세트 | n | 통과 | 점추정 | Wilson 95% | 합격선(팀 목표) | 판정 | 동결 해시 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(
            f"| {r['area']} | {r['set']} | {r['n']} | {r['pass']} | {r['point']:.1%} | {r['lower']:.1%}~{r['upper']:.1%} | "
            f"{r['target']} | {r['verdict']} | {r['freeze'][:12]} |"
        )
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graded", required=True, nargs="+", help="rule_grader --out 결과(여러 개면 label=경로)")
    ap.add_argument("--freeze", default="", help="동결 해시(문항 파일 sha256 또는 커밋)")
    args = ap.parse_args(argv)
    rows: list[dict] = []
    for spec in args.graded:
        label, _, path = spec.partition("=") if "=" in spec else (pathlib.Path(spec).stem, "", spec)
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        rows += table(data["graded"], f"{label}({data.get('layer', 'full')})", args.freeze)
    print(markdown(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
