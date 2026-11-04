from __future__ import annotations

import csv
from pathlib import Path

from app.services.companion.fusion import fuse_turn

# S11-3: 턴 결합 기대 결과표 (scripts/companion/fusion_expected.csv) 를 실제 결합 함수와 대조한다
TABLE = Path(__file__).resolve().parents[1] / "scripts" / "companion" / "fusion_expected.csv"


def _flag(value: str) -> bool:
    return value == "True"


def test_fusion_matches_expected_table():
    with TABLE.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 9
    for row in rows:
        got = fuse_turn(
            no_response=row["no_response"] == "1",
            text_label=row["text_label"] or None,
            text_usable=_flag(row["text_usable"]),
            face_label=row["face_label"] or None,
            face_usable=_flag(row["face_usable"]),
        )
        assert got.source == row["expected_source"], row
        assert (got.label or "") == row["expected_label"], row
