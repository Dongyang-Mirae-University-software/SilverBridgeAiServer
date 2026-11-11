"""2차 점검 묶음 D(A-1~A-8) - 실시간 감지 방송·세션·수집·클립."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from app.core.config import get_settings
from app.routers import health_router
from app.services import stream_session_service as sss
from app.services import training_sample_collector as tsc
from app.services.clip_buffer import ClipFrameBuffer
from app.schemas.clip_schema import ClipRequest
from app.services.clip_service import ClipService
from app.services.live_detection_merge import merge_kind_results
from app.services.live_ws_manager import LiveWebSocketManager
from app.services.session_analysis_store import SessionAnalysisStore, session_analysis_store
from app.services.training_sample_collector import TrainingSampleCollector
from tests.clip_helpers import make_jpeg

JPEG = b"\xff\xd8\xff\xe0fakejpeg\xff\xd9"


def _svc(**settings: Any) -> sss.StreamSessionService:
    svc = sss.StreamSessionService(db=None, frame_store=sss.StreamFrameStore())  # type: ignore[arg-type]
    svc.settings = get_settings().model_copy(update=settings)
    return svc


# --- A-2 ------------------------------------------------------------------------------------


def test_latest_analysis_fresh_is_returned_and_stale_is_dropped() -> None:
    svc = _svc(live_latest_analysis_max_age_sec=30.0)
    sid = "a2_stale"
    session_analysis_store.clear_session(sid)
    fresh = {"detectedType": "fire", "danger": True, "analyzedAt": datetime.utcnow().isoformat()}
    session_analysis_store.set_result(sid, fresh)
    assert svc.latest_analysis_for_session(sid, "cam")["danger"] is True

    old = dict(fresh, analyzedAt=(datetime.utcnow() - timedelta(seconds=120)).isoformat())
    session_analysis_store.set_result(sid, old)
    assert svc.latest_analysis_for_session(sid, "cam") is None
    session_analysis_store.set_result(sid, dict(fresh, analyzedAt="garbage"))
    assert svc.latest_analysis_for_session(sid, "cam") is None
    session_analysis_store.clear_session(sid)


def test_create_or_restart_clears_previous_result_and_counter() -> None:
    svc = _svc(stream_state_backend="memory")
    sid = "a2_restart"
    session_analysis_store.set_result(sid, {"detectedType": "fire", "danger": True})
    session_analysis_store.increment_frame(sid)
    svc.create_or_restart(sid, "cam", "web")
    assert session_analysis_store.get_result(sid) is None
    assert session_analysis_store.increment_frame(sid) == 1


# --- A-6 ------------------------------------------------------------------------------------


def test_store_drops_result_from_old_generation() -> None:
    store = SessionAnalysisStore()
    gen = store.generation("s")
    store.clear_session("s")
    assert store.set_result("s", {"x": 1}, generation=gen) is False
    assert store.get_result("s") is None
    assert store.set_result("s", {"x": 2}, generation=store.generation("s")) is True


def test_analysis_finishing_after_stop_does_not_refill(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _svc(stream_state_backend="memory", fire_smoke_enabled=True, knife_enabled=False, fall_enabled=False,
               frame_quality_gate_enabled=False, fire_hold_enabled=True)
    sid = "a6_stop"
    session_analysis_store.clear_session(sid)

    class _Det:
        def detect_from_jpeg(self, _: bytes) -> dict:
            svc.stop(svc.create_or_restart(sid, "cam", "web"))  # 추론 도중 stop
            return {"detectedType": "fire", "confidence": 0.9, "danger": True, "detections": [],
                    "detectedAt": datetime.utcnow().isoformat()}

    monkeypatch.setattr(sss, "get_fire_smoke_detector", lambda: _Det())
    svc._detect_and_store(sid, JPEG)
    assert session_analysis_store.get_result(sid) is None
    assert sss.fire_hold_tracker.update(sid, False, 0.0, "normal", 1.0, window_sec=5, ratio=0.5,
                                        min_scenes=1, max_gap_sec=5) is not None  # 추적기 접근 가능(비워진 상태에서 시작)
    sss.fire_hold_tracker.clear(sid)


# --- A-5 ------------------------------------------------------------------------------------


class _WS:
    def __init__(self, delay: float = 0.0, fail: bool = False) -> None:
        self.delay, self.fail, self.sent = delay, fail, []

    async def send_json(self, payload: dict) -> None:
        if self.fail:
            raise RuntimeError("closed")
        await asyncio.sleep(self.delay)
        self.sent.append(payload)


def test_broadcast_slow_subscriber_is_dropped_without_blocking_others(monkeypatch: pytest.MonkeyPatch) -> None:
    mgr = LiveWebSocketManager()
    monkeypatch.setattr(mgr, "_send_timeout", staticmethod(lambda: 0.2))

    async def run() -> tuple[float, _WS, _WS, _WS]:
        slow, fast, dead = _WS(delay=5), _WS(), _WS(fail=True)
        for w in (slow, fast, dead):
            await mgr.subscribe(w, "s1")  # type: ignore[arg-type]
        started = time.monotonic()
        await mgr.broadcast({"type": "t"}, session_id="s1")
        return time.monotonic() - started, slow, fast, dead

    elapsed, slow, fast, dead = asyncio.run(run())
    assert elapsed < 1.0
    assert fast.sent == [{"type": "t"}]
    assert mgr._subscriptions.get("s1") == {fast}  # 느린·실패 구독자는 빠졌다


def test_broadcast_task_references_are_held_and_released() -> None:
    mgr = LiveWebSocketManager()

    async def run() -> tuple[int, int]:
        w = _WS(delay=0.05)
        await mgr.subscribe(w, "s")  # type: ignore[arg-type]
        mgr.broadcast_nowait({"type": "t"}, session_id="s")
        during = len(mgr._tasks)
        await asyncio.sleep(0.3)
        return during, len(mgr._tasks)

    during, after = asyncio.run(run())
    assert during == 1 and after == 0


# --- A-3 ------------------------------------------------------------------------------------


def _res(dt: str, conf: float = 0.0, danger: bool = False) -> dict:
    return {"detectedType": dt, "confidence": conf, "danger": danger, "detections": []}


def test_merge_reports_unavailable_kinds_only_when_present() -> None:
    merged = merge_kind_results([("fire", _res("unknown")), ("knife", _res("normal")), ("fall", _res("normal"))])
    assert merged["detectedType"] == "normal" and merged["unavailable"] == ["fire"]
    ok = merge_kind_results([("fire", _res("normal")), ("knife", _res("normal"))])
    assert "unavailable" not in ok


def test_unavailable_error_log_once_per_ten_minutes(caplog: pytest.LogCaptureFixture) -> None:
    log = sss._UnavailableLog()
    with caplog.at_level(logging.ERROR, logger=sss._LOG.name):
        for _ in range(5):
            log.report("ward_abcdefghijklmn123456", "fire")
        log.report("ward_abcdefghijklmn123456", "knife")
    msgs = [r.getMessage() for r in caplog.records if "LIVE-DETECTOR-UNAVAILABLE" in r.getMessage()]
    assert len(msgs) == 2
    assert "kind=fire" in msgs[0] and msgs[0].endswith("...123456") and "ward_abc" not in msgs[0]


def test_health_degraded_when_enabled_detector_not_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health_router, "_detector_status", lambda: {
        "fire": {"enabled": True, "loaded": False}, "knife": {"enabled": False, "loaded": False},
        "fall": {"enabled": True, "loaded": True}})
    data = health_router.health()["data"]
    assert data["status"] == "ok" and data["degraded"] is True and data["unavailableDetectors"] == ["fire"]
    assert data["detectors"]["fire"] == {"enabled": True, "loaded": False}
    monkeypatch.setattr(health_router, "_detector_status", lambda: {
        "fire": {"enabled": True, "loaded": True}, "knife": {"enabled": False, "loaded": False}})
    assert "degraded" not in health_router.health()["data"]


# --- A-1 ------------------------------------------------------------------------------------


def _csettings(tmp_path: Path, **kw: Any):
    base = {"collect_enabled": True, "collect_session_ids": "ward_*", "collect_dir": str(tmp_path / "collect"),
            "collect_min_free_gb": 0.0, "collect_retention_days": 14, "collect_min_confidence": 0.2,
            "collect_daily_max_per_session": 2000, "collect_max_total_gb": 20.0,
            "fire_smoke_danger_threshold": 0.6, "knife_danger_threshold": 0.5, "fall_danger_threshold": 0.4}
    return get_settings().model_copy(update={**base, **kw})


def _col() -> TrainingSampleCollector:
    return TrainingSampleCollector(inline=True, now_fn=lambda: datetime(2026, 10, 7, 3, 0, 0), mono_fn=lambda: 1000.0)


def test_purge_expired_removes_old_day_folders_even_when_collection_disabled(tmp_path: Path) -> None:
    root = tmp_path / "collect"
    for day in ("20260901", "20260920", "20261007"):
        (root / day / "fire").mkdir(parents=True)
        (root / day / "fire" / "a.jpg").write_bytes(b"x")
    removed = _col().purge_expired(_csettings(tmp_path, collect_enabled=False))
    assert removed == 2
    assert sorted(p.name for p in root.iterdir()) == ["20261007"]
    assert _col().purge_expired(_csettings(tmp_path / "none")) == 0  # 폴더 없음


def test_purge_loop_starts_once_and_stops(tmp_path: Path) -> None:
    col = _col()
    col.purge_expired(_csettings(tmp_path), start_loop=True)
    first = col._purge_thread
    col.purge_expired(_csettings(tmp_path), start_loop=True)
    assert first is col._purge_thread and first.daemon and first.is_alive()
    col.stop_purge_loop()
    first.join(timeout=2)
    assert not first.is_alive()


def test_daily_counter_starts_from_existing_files_after_restart(tmp_path: Path) -> None:
    s = _csettings(tmp_path, collect_session_ids="ward_x", collect_daily_max_per_session=2)
    folder = tmp_path / "collect" / "20261007" / "fire" / "alert"
    folder.mkdir(parents=True)
    for i in range(2):  # 재시작 전에 이미 저장된 2장(다른 세션·무작위 파일은 세지 않는다)
        (folder / f"03000{i}000_ward_x_0.90.jpg").write_bytes(b"y")
    (folder / "030009000_ward_other_0.90.jpg").write_bytes(b"y")
    (folder.parent.parent / "random").mkdir()
    (folder.parent.parent / "random" / "r_030009000_ward_x_0.10.jpg").write_bytes(b"y")
    kinds = [("fire", {"detectedType": "fire", "confidence": 0.9, "danger": True, "detections": [],
                       "detectedAt": "2026-10-07T03:00:00"})]
    before = len(list((tmp_path / "collect").rglob("*.jpg")))
    _col().offer("ward_x", JPEG, kinds, settings=s)  # 새 인스턴스 = 재시작 직후, 상한 2 이미 도달
    assert len(list((tmp_path / "collect").rglob("*.jpg"))) == before


def test_broad_prefix_warns_once_without_value(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    col = _col()
    with caplog.at_level(logging.WARNING, logger=tsc._LOG.name):
        col.purge_expired(_csettings(tmp_path, collect_session_ids="ward_*"))
        col.purge_expired(_csettings(tmp_path, collect_session_ids="ward_*"))
        col.purge_expired(_csettings(tmp_path, collect_session_ids="ward_abc*"))
    warns = [r.getMessage() for r in caplog.records if "covers all user" in r.getMessage()]
    assert len(warns) == 1 and "ward_" not in warns[0]


# --- A-7 ------------------------------------------------------------------------------------


def test_stop_clears_frame_store_entries() -> None:
    svc = _svc(stream_state_backend="memory")
    svc.create_or_restart("a7_stop", "cam", "web")
    svc.frame_store.set_frame("a7_stop", JPEG)
    svc.frame_store.increment_viewer("a7_stop")
    svc.stop(svc.get_by_session_id("a7_stop"))  # type: ignore[arg-type]
    fs = svc.frame_store
    assert fs.get_frame("a7_stop") is None and fs.get_fps("a7_stop") == 0.0 and fs.get_viewer_count("a7_stop") == 0
    fs.decrement_viewer("a7_stop")  # MJPEG 시청 종료 시점 - 예외 없이 끝난다


def test_stale_session_memory_is_purged_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _svc(stream_state_backend="memory", live_stale_session_purge_sec=1800.0)
    monkeypatch.setattr(sss, "_last_stale_purge", None)
    svc.create_or_restart("a7_old", "cam", "web")
    svc.create_or_restart("a7_new", "cam", "web")
    svc.frame_store.set_frame("a7_old", JPEG)
    svc.frame_store.set_frame("a7_new", JPEG)
    svc.frame_store._frame_times["a7_old"] = datetime.utcnow() - timedelta(minutes=31)
    assert svc._purge_stale_sessions() == 1
    assert svc.frame_store.get_frame("a7_old") is None and svc.frame_store.get_frame("a7_new") == JPEG
    assert svc.get_by_session_id("a7_old") is not None  # 세션 기록은 남긴다
    assert svc._purge_stale_sessions() == 0  # 60초 안에는 다시 훑지 않는다


def test_stale_purge_skips_sessions_with_viewers(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = _svc(live_stale_session_purge_sec=1800.0)
    monkeypatch.setattr(sss, "_last_stale_purge", None)
    svc.frame_store.set_frame("a7_view", JPEG)
    svc.frame_store.increment_viewer("a7_view")
    svc.frame_store._frame_times["a7_view"] = datetime.utcnow() - timedelta(hours=2)
    assert svc._purge_stale_sessions() == 0


# --- A-8 ------------------------------------------------------------------------------------


def test_clip_fps_uses_actual_frame_time_range() -> None:
    buf = ClipFrameBuffer(buffer_seconds=30.0, max_frames=500)
    detected = datetime.utcnow() - timedelta(seconds=20)
    # 요청 구간은 5초(앞 3 + 뒤 2)인데 실제로는 마지막 2초(10프레임, 0.2초 간격)만 담겼다.
    for i in range(11):
        buf.add("a8", detected - timedelta(seconds=1) + timedelta(seconds=i * 0.2), make_jpeg(64, 48, i))
    seen: dict[str, float] = {}

    def enc(jpegs, fps, *_: object) -> bytes:
        seen["fps"], seen["n"] = fps, len(jpegs)
        return b"\x1a\x45\xdf\xa3x"

    service = ClipService(get_settings().model_copy(update={"clip_min_frames": 2}), buf, encoder=enc)
    result = asyncio.run(service.create_clip("a8", ClipRequest(detectedAt=detected)))
    assert seen["n"] == 11
    assert seen["fps"] == pytest.approx(11 / 2.0)  # 실제 범위 2초 → 5.5fps (예전: 11/5=2.2fps 로 2.5배 빠른 재생)
    assert result.duration_ms == 2000
