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


# --- 무작위 장면(reason=random, 모델 비교용) ----------------------------------------------------

RSID = "ward_random_cam"


def _rand_settings(tmp_path: Path, **kw: Any):
    base = {"collect_random_interval_sec": 100.0, "collect_random_session_ids": RSID, "collect_session_ids": ""}
    return _settings(tmp_path, **{**base, **kw})


def _rcol(clocks, fraction: float = 0.5):
    mono, wall = clocks
    return TrainingSampleCollector(inline=True, now_fn=wall, mono_fn=mono, random_fn=lambda: fraction)


def _randoms(tmp_path: Path) -> list[Path]:
    return [p for p in _files(tmp_path) if p.parent.name == "random"]


def _tick(clocks, sec: float) -> None:
    mono, wall = clocks
    mono.t += sec
    wall.t += timedelta(seconds=sec)


def _drive(col, s, clocks, sid: str = RSID, steps: int = 1, step_sec: float = 1.0, kinds=None) -> None:
    for _ in range(steps):
        _tick(clocks, step_sec)
        col.offer(sid, JPEG, kinds or _kinds(), settings=s)


def test_random_defaults_are_off():
    s = get_settings()
    assert s.collect_random_interval_sec == 0.0 and s.collect_random_session_ids == ""
    assert s.collect_random_daily_max == 300


def test_random_interval_zero_saves_nothing(tmp_path, clocks):
    col = _rcol(clocks)
    s = _rand_settings(tmp_path, collect_random_interval_sec=0.0)
    _drive(col, s, clocks, steps=500)
    assert _files(tmp_path) == []


def test_random_requires_collect_enabled(tmp_path, clocks):
    col = _rcol(clocks)
    s = _rand_settings(tmp_path, collect_enabled=False)
    _drive(col, s, clocks, steps=500)
    assert _files(tmp_path) == []


def test_random_only_listed_sessions(tmp_path, clocks):
    col = _rcol(clocks)
    s = _rand_settings(tmp_path)
    _drive(col, s, clocks, sid="ward_other_cam", steps=300)
    assert _files(tmp_path) == []
    _drive(col, s, clocks, steps=300)
    assert len(_randoms(tmp_path)) >= 1
    assert {p.stem.split("_", 2)[2].rsplit("_", 1)[0] for p in _randoms(tmp_path)} == {RSID}


def test_random_empty_list_saves_nothing(tmp_path, clocks):
    col = _rcol(clocks)
    s = _rand_settings(tmp_path, collect_random_session_ids="")
    _drive(col, s, clocks, steps=500)
    assert _files(tmp_path) == []


@pytest.mark.parametrize("spec", ["*", "ward_*", "ward_random_*", f"{RSID}*", "a*b"])
def test_random_wildcards_rejected_with_warning(tmp_path, clocks, caplog, spec):
    col = _rcol(clocks)
    s = _rand_settings(tmp_path, collect_random_session_ids=spec)
    with caplog.at_level("WARNING", logger=tsc._LOG.name):
        _drive(col, s, clocks, steps=500)
    assert _files(tmp_path) == []
    assert "wildcard entries rejected" in caplog.text


def test_random_mixed_list_keeps_exact_ids_only(tmp_path, clocks):
    col = _rcol(clocks)
    s = _rand_settings(tmp_path, collect_random_session_ids=f"*, {RSID}")
    _drive(col, s, clocks, sid="ward_other_cam", steps=300)
    assert _files(tmp_path) == []
    _drive(col, s, clocks, steps=300)
    assert len(_randoms(tmp_path)) >= 1


def test_random_first_frame_not_saved_and_jitter_lower_bound(tmp_path, clocks):
    col = _rcol(clocks, fraction=0.0)  # 간격 x 0.7 = 70초
    s = _rand_settings(tmp_path)
    _drive(col, s, clocks)  # 처음 본 세션: 저장하지 않고 70초 뒤로 예약
    assert _randoms(tmp_path) == []
    _tick(clocks, 69.0)
    col.offer(RSID, JPEG, _kinds(), settings=s)
    assert _randoms(tmp_path) == []
    _tick(clocks, 1.0)
    col.offer(RSID, JPEG, _kinds(), settings=s)
    assert len(_randoms(tmp_path)) == 1


def test_random_jitter_upper_bound(tmp_path, clocks):
    col = _rcol(clocks, fraction=1.0)  # 간격 x 1.3 = 130초
    s = _rand_settings(tmp_path)
    _drive(col, s, clocks)
    _tick(clocks, 129.0)
    col.offer(RSID, JPEG, _kinds(), settings=s)
    assert _randoms(tmp_path) == []
    _tick(clocks, 1.0)
    col.offer(RSID, JPEG, _kinds(), settings=s)
    assert len(_randoms(tmp_path)) == 1
    _tick(clocks, 129.0)  # 저장 뒤 다시 한 간격(최대 130초)을 기다린다
    col.offer(RSID, JPEG, _kinds(), settings=s)
    assert len(_randoms(tmp_path)) == 1
    _tick(clocks, 1.0)
    col.offer(RSID, JPEG, _kinds(), settings=s)
    assert len(_randoms(tmp_path)) == 2


def test_random_actual_gaps_stay_within_30_percent(tmp_path, clocks):
    mono, wall = clocks
    col = TrainingSampleCollector(inline=True, now_fn=wall, mono_fn=mono)  # 진짜 난수
    s = _rand_settings(tmp_path, collect_random_daily_max=10_000)
    times: list[float] = []
    last = len(_randoms(tmp_path))
    for _ in range(8_000):
        _tick(clocks, 1.0)
        col.offer(RSID, JPEG, _kinds(), settings=s)
        now_count = len(_randoms(tmp_path))
        if now_count != last:
            times.append(mono.t)
            last = now_count
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert len(gaps) > 40
    assert all(70.0 <= g <= 131.0 for g in gaps)  # 1초 틱 단위 반올림 여유 1초
    assert len({round(g) for g in gaps}) > 5  # 고정 주기가 아니다


def test_random_saved_regardless_of_score_and_meta_has_all_kinds(tmp_path, clocks):
    col = _rcol(clocks, fraction=0.0)
    s = _rand_settings(tmp_path)
    fq = {"ok": False, "reason": "blur"}
    _drive(col, s, clocks)
    _tick(clocks, 70.0)
    col.offer(RSID, JPEG, _kinds(knife=("knife", 0.31, False), fall=("normal", 0.0, False)), settings=s, frame_quality=fq)
    jpg = _randoms(tmp_path)[0]
    assert jpg.parent.parent.name == "20261007"  # <day>/random/ - 종류 폴더 밖
    assert jpg.name.startswith("r_") and jpg.read_bytes() == JPEG
    meta = json.loads(jpg.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["kind"] == "random" and meta["reason"] == "random" and meta["sessionId"] == RSID
    by_kind = {r["kind"]: r for r in meta["results"]}
    assert set(by_kind) == {"fire", "knife", "fall"}
    assert by_kind["knife"]["confidence"] == 0.31 and by_kind["knife"]["detectedType"] == "knife"
    assert all({"confidence", "danger", "detectedType"} <= set(r) for r in meta["results"])
    assert meta["frameQuality"] == fq
    assert not list((tmp_path / "collect").rglob(".*.tmp"))


def test_random_daily_cap_separate_from_near_alert(tmp_path, clocks):
    col = _rcol(clocks, fraction=0.0)
    s = _rand_settings(
        tmp_path, collect_random_daily_max=2, collect_session_ids=RSID, collect_daily_max_per_session=2000,
    )
    _drive(col, s, clocks)
    for _ in range(5):
        _tick(clocks, 70.0)
        col.offer(RSID, JPEG, _kinds(), settings=s)
    assert len(_randoms(tmp_path)) == 2  # 상한 2
    _tick(clocks, 11.0)
    col.offer(RSID, JPEG, _kinds(fire=("fire", 0.9, True)), settings=s)  # near/alert 는 별개 카운터
    assert len([p for p in _files(tmp_path) if p.parent.name == "alert"]) == 1


def test_random_does_not_change_near_alert(tmp_path, clocks):
    col = _rcol(clocks, fraction=0.0)
    s = _rand_settings(tmp_path, collect_session_ids=RSID)
    _drive(col, s, clocks)
    _tick(clocks, 70.0)
    col.offer(RSID, JPEG, _kinds(fire=("fire", 0.9, True)), settings=s)  # 같은 프레임: random + alert
    names = sorted((p.parent.name, p.name) for p in _files(tmp_path))
    assert [n[0] for n in names] == ["alert", "random"]
    assert len({n[1] for n in names}) == 2  # 파일 이름이 겹치지 않는다
    # random 대상이 아닌 세션의 near/alert 는 그대로 저장된다
    s2 = _rand_settings(tmp_path, collect_session_ids="ward_other_cam")
    col.offer("ward_other_cam", JPEG, _kinds(fire=("fire", 0.3, False)), settings=s2)
    assert any(p.parent.name == "near" for p in _files(tmp_path))


def test_random_session_not_in_near_alert_list_saves_only_random(tmp_path, clocks):
    col = _rcol(clocks, fraction=0.0)
    s = _rand_settings(tmp_path, collect_session_ids="ward_other_cam")
    _drive(col, s, clocks)
    _tick(clocks, 70.0)
    col.offer(RSID, JPEG, _kinds(fire=("fire", 0.9, True)), settings=s)
    assert [p.parent.name for p in _files(tmp_path)] == ["random"]


def test_random_write_failure_never_raises(tmp_path, clocks, monkeypatch):
    col = _rcol(clocks, fraction=0.0)
    s = _rand_settings(tmp_path)
    _drive(col, s, clocks)
    _tick(clocks, 70.0)
    monkeypatch.setattr(TrainingSampleCollector, "_write_atomic", staticmethod(lambda *a, **k: (_ for _ in ()).throw(OSError("x"))))
    col.offer(RSID, JPEG, _kinds(), settings=s)  # 예외가 올라오지 않는다
    assert _files(tmp_path) == []


def test_detect_and_store_passes_frame_quality_to_collector(tmp_path, monkeypatch):
    seen: dict[str, Any] = {}

    class Spy:
        def offer(self, *a, **k):
            seen.update(k)

    monkeypatch.setattr(sss, "training_sample_collector", Spy())
    _run(_service(monkeypatch, tmp_path))
    assert "frame_quality" in seen
