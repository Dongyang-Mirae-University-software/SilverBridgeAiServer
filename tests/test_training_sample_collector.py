from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from app.core.config import get_settings
from app.services import stream_session_service as sss
from app.services import training_sample_collector as tsc
from app.services.fall_hold_tracker import fall_hold_tracker
from app.services.session_analysis_store import session_analysis_store
from app.services.training_sample_collector import TrainingSampleCollector

JPEG = b"\xff\xd8\xff\xe0fakejpeg\xff\xd9"
SID = "ward_test_cam"


class _Mono:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class _Wall:
    def __init__(self) -> None:
        self.t = datetime(2026, 10, 7, 3, 0, 0)  # UTC → KST 12:00

    def __call__(self) -> datetime:
        return self.t


def _settings(tmp_path: Path, **kw: Any):
    base = {
        "collect_enabled": True,
        "collect_session_ids": SID,
        "collect_dir": str(tmp_path / "collect"),
        "collect_min_confidence": 0.2,
        "collect_daily_max_per_session": 2000,
        "collect_max_total_gb": 20.0,
        "collect_min_free_gb": 0.0,
        "collect_retention_days": 14,
        "fire_smoke_danger_threshold": 0.6,
        "knife_danger_threshold": 0.5,
        "fall_danger_threshold": 0.4,
    }
    return get_settings().model_copy(update={**base, **kw})


def _res(dt: str, conf: float, danger: bool = False) -> dict[str, Any]:
    det = [] if dt in {"normal", "unknown"} else [{"detectedType": dt, "confidence": conf, "bbox": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}}]
    return {"detectedType": dt, "confidence": conf, "danger": danger, "detections": det, "detectedAt": "2026-10-07T03:00:00"}


def _files(tmp_path: Path, pattern: str = "*.jpg") -> list[Path]:
    return sorted((tmp_path / "collect").rglob(pattern)) if (tmp_path / "collect").exists() else []


@pytest.fixture
def clocks():
    return _Mono(), _Wall()


@pytest.fixture
def col(clocks):
    mono, wall = clocks
    return TrainingSampleCollector(inline=True, now_fn=wall, mono_fn=mono)


def _kinds(fire=("normal", 0.0, False), knife=("normal", 0.0, False), fall=("normal", 0.0, False)):
    return [("fire", _res(*fire)), ("knife", _res(*knife)), ("fall", _res(*fall))]


def test_disabled_saves_nothing(tmp_path, col):
    s = _settings(tmp_path, collect_enabled=False)
    col.offer(SID, JPEG, _kinds(fire=("fire", 0.9, True)), settings=s)
    assert _files(tmp_path) == []


def test_session_not_allowed_or_empty_list_saves_nothing(tmp_path, col):
    col.offer("other_session", JPEG, _kinds(fire=("fire", 0.9, True)), settings=_settings(tmp_path))
    col.offer(SID, JPEG, _kinds(fire=("fire", 0.9, True)), settings=_settings(tmp_path, collect_session_ids=""))
    assert _files(tmp_path) == []


@pytest.mark.parametrize(
    ("spec", "sid", "saved"),
    [
        ("ward_*", "ward_AbC123", True),
        ("ward_*", "other_AbC123", False),
        ("ward_exact, other_*", "ward_exact", True),
        ("ward_exact, other_*", "other_zzz", True),
        ("ward_exact", "ward_exact2", False),
        ("ward_*x", "ward_ax", False),  # 중간 * 는 무시
        ("a*b", "axb", False),
        ("*", "anything", True),
        (" , ", "ward_x", False),
        ("", "ward_x", False),
    ],
)
def test_session_allow_patterns(tmp_path, col, spec, sid, saved):
    s = _settings(tmp_path, collect_session_ids=spec)
    col.offer(sid, JPEG, _kinds(fire=("fire", 0.9, True)), settings=s)
    assert bool(_files(tmp_path)) is saved


def test_spec_change_is_reparsed(tmp_path, col):
    col.offer("ward_a", JPEG, _kinds(fire=("fire", 0.9, True)), settings=_settings(tmp_path, collect_session_ids="nope"))
    assert _files(tmp_path) == []
    col.offer("ward_a", JPEG, _kinds(fire=("fire", 0.9, True)), settings=_settings(tmp_path, collect_session_ids="ward_*"))
    assert len(_files(tmp_path)) == 1


def test_alert_and_near_classification(tmp_path, col):
    s = _settings(tmp_path)
    col.offer(SID, JPEG, _kinds(fire=("fire", 0.9, True), knife=("knife", 0.3, False)), settings=s)
    rel = {str(p.relative_to(tmp_path / "collect")).split("/", 1)[1].split("/")[0:2].__str__() for p in _files(tmp_path)}
    assert rel == {"['fire', 'alert']", "['knife', 'near']"}
    assert _files(tmp_path)[0].relative_to(tmp_path / "collect").parts[0] == "20261007"  # KST 날짜


def test_below_floor_normal_unknown_not_saved(tmp_path, col):
    s = _settings(tmp_path)
    col.offer(SID, JPEG, _kinds(fire=("fire", 0.19, False), knife=("normal", 0.9, False), fall=("unknown", 0.0, False)), settings=s)
    assert _files(tmp_path) == []


def test_fall_above_threshold_before_hold_is_near(tmp_path, col):
    col.offer(SID, JPEG, _kinds(fall=("fall", 0.8, False)), settings=_settings(tmp_path))
    files = _files(tmp_path)
    assert len(files) == 1 and files[0].parent.name == "near" and files[0].parent.parent.name == "fall"


def test_interval_10s_per_kind_and_reason(tmp_path, clocks):
    mono, wall = clocks
    col = TrainingSampleCollector(inline=True, now_fn=wall, mono_fn=mono)
    s = _settings(tmp_path)
    near = _kinds(fire=("fire", 0.3, False))
    alert = _kinds(fire=("fire", 0.9, True))
    col.offer(SID, JPEG, near, settings=s)
    mono.t += 5
    wall.t += timedelta(seconds=5)
    col.offer(SID, JPEG, near, settings=s)  # 10초 이내 → 건너뜀
    col.offer(SID, JPEG, alert, settings=s)  # 다른 이유 → near 가 alert 를 막지 않는다
    assert len(_files(tmp_path)) == 2
    mono.t += 6
    wall.t += timedelta(seconds=6)
    col.offer(SID, JPEG, near, settings=s)  # 첫 near 로부터 11초
    assert len(_files(tmp_path)) == 3


def test_daily_cap(tmp_path, clocks):
    mono, wall = clocks
    col = TrainingSampleCollector(inline=True, now_fn=wall, mono_fn=mono)
    s = _settings(tmp_path, collect_daily_max_per_session=2)
    for _ in range(4):
        mono.t += 11
        wall.t += timedelta(seconds=11)
        col.offer(SID, JPEG, _kinds(fire=("fire", 0.9, True)), settings=s)
    assert len(_files(tmp_path)) == 2


def test_total_size_cap_stops(tmp_path, col):
    s = _settings(tmp_path, collect_max_total_gb=0.0)
    col.offer(SID, JPEG, _kinds(fire=("fire", 0.9, True)), settings=s)
    assert _files(tmp_path) == []


def test_low_free_disk_stops(tmp_path, col, monkeypatch):
    s = _settings(tmp_path, collect_min_free_gb=5.0)
    monkeypatch.setattr(tsc.shutil, "disk_usage", lambda _p: type("U", (), {"free": 1024**3})())
    col.offer(SID, JPEG, _kinds(fire=("fire", 0.9, True)), settings=s)
    assert _files(tmp_path) == []


def test_old_date_folders_deleted(tmp_path, col):
    root = tmp_path / "collect"
    (root / "20260901" / "fire" / "alert").mkdir(parents=True)
    (root / "20260901" / "fire" / "alert" / "a.jpg").write_bytes(b"x" * 10)
    (root / "20261001").mkdir()  # 보관 기간 안(6일 전)
    col.offer(SID, JPEG, _kinds(fire=("fire", 0.9, True)), settings=_settings(tmp_path))
    assert not (root / "20260901").exists()
    assert (root / "20261001").exists()
    assert len(_files(tmp_path)) == 1


def test_json_content_and_atomic_files(tmp_path, col):
    s = _settings(tmp_path)
    col.offer(SID, JPEG, _kinds(fire=("fire", 0.9, True), knife=("knife", 0.3, False)), settings=s)
    jpg = next(p for p in _files(tmp_path) if p.parent.parent.name == "fire")
    assert jpg.read_bytes() == JPEG  # 원본 그대로
    meta = json.loads(jpg.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["sessionId"] == SID and meta["kind"] == "fire" and meta["reason"] == "alert"
    assert meta["confidence"] == 0.9 and meta["threshold"] == 0.6 and meta["bytes"] == len(JPEG)
    assert meta["detections"][0]["bbox"]["x2"] == 3
    assert {r["kind"] for r in meta["results"]} == {"fire", "knife", "fall"}
    assert meta["analyzedAt"] == "2026-10-07T03:00:00"
    assert not list((tmp_path / "collect").rglob(".*.tmp"))
    assert (os.stat(jpg).st_mode & 0o777) == 0o600
    assert jpg.name.endswith("_0.90.jpg") and "_" + SID + "_" in jpg.name


def test_unsafe_session_id_cannot_escape(tmp_path, col):
    bad = "../../evil"
    col.offer(bad, JPEG, _kinds(fire=("fire", 0.9, True)), settings=_settings(tmp_path, collect_session_ids=bad))
    files = _files(tmp_path)
    assert len(files) == 1 and (tmp_path / "collect") in files[0].parents
    assert "/" not in files[0].name and ".." not in files[0].name


def test_write_failure_never_raises_and_leaves_no_orphan(tmp_path, col, monkeypatch):
    real = TrainingSampleCollector._write_atomic
    calls = {"n": 0}

    def flaky(path, data):
        calls["n"] += 1
        if path.suffix == ".json":
            raise OSError("disk full")
        real(path, data)

    monkeypatch.setattr(TrainingSampleCollector, "_write_atomic", staticmethod(flaky))
    col.offer(SID, JPEG, _kinds(fire=("fire", 0.9, True)), settings=_settings(tmp_path))
    assert _files(tmp_path) == []


# --- _detect_and_store 통합 ---------------------------------------------------------------


class _Det:
    def __init__(self, dt: str, conf: float, danger: bool) -> None:
        self.r = _res(dt, conf, danger)

    def detect_from_jpeg(self, _b: bytes) -> dict[str, Any]:
        return dict(self.r)


def _service(monkeypatch, tmp_path, **kw):
    dets = {"fire": _Det("fire", 0.9, True), "knife": _Det("normal", 0.0, False), "fall": _Det("normal", 0.0, False)}
    monkeypatch.setattr(sss, "get_fire_smoke_detector", lambda: dets["fire"])
    monkeypatch.setattr(sss, "get_knife_detector", lambda: dets["knife"])
    monkeypatch.setattr(sss, "get_fall_detector", lambda: dets["fall"])
    s = _settings(tmp_path, fire_smoke_enabled=True, knife_enabled=True, fall_enabled=True, **kw)
    monkeypatch.setattr(sss, "get_settings", lambda: s)
    return sss.StreamSessionService(db=None, frame_store=sss.frame_store)  # type: ignore[arg-type]


def _run(service) -> dict[str, Any]:
    session_analysis_store.clear_session(SID)
    fall_hold_tracker.clear(SID)
    return service._detect_and_store(SID, JPEG)


def test_detect_and_store_result_same_when_collection_on_off(tmp_path, monkeypatch):
    inline = TrainingSampleCollector(inline=True)
    monkeypatch.setattr(sss, "training_sample_collector", inline)
    on = _run(_service(monkeypatch, tmp_path))
    assert len(_files(tmp_path)) == 1
    off = _run(_service(monkeypatch, tmp_path, collect_enabled=False))
    on.pop("analyzedAt"), off.pop("analyzedAt")
    assert on == off


def test_detect_and_store_survives_collector_exception(tmp_path, monkeypatch):
    class Boom:
        def offer(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr(sss, "training_sample_collector", Boom())
    got = _run(_service(monkeypatch, tmp_path))
    monkeypatch.setattr(sss, "training_sample_collector", TrainingSampleCollector(inline=True))
    ref = _run(_service(monkeypatch, tmp_path, collect_enabled=False))
    got.pop("analyzedAt"), ref.pop("analyzedAt")
    assert got == ref


def test_defaults_are_off():
    s = get_settings()
    assert s.collect_enabled is False and s.collect_session_ids == ""
