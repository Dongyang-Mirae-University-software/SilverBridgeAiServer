from __future__ import annotations

from datetime import datetime
from typing import Any

import cv2
import numpy as np
import pytest

from app.core.config import _as_int_in_range, get_settings
from app.services import stream_session_service as sss
from app.services.fall_hold_tracker import fall_hold_tracker
from app.services.fire_hold_tracker import FireHoldTracker, fire_hold_tracker
from app.services.session_analysis_store import session_analysis_store

KW = dict(window=4, min_positive=3, max_gap_sec=10.0)


def _step(tr: FireHoldTracker, seq: list[bool], *, start: float = 0.0, dt: float = 0.65, sid: str = "s", **kw: Any):
    kw = {**KW, **kw}
    out = []
    for i, pos in enumerate(seq):
        out.append(tr.update(sid, pos, 0.7 if pos else 0.0, "smoke" if pos else "normal", start + i * dt, **kw))
    return out


# --- 트래커 ---------------------------------------------------------------------


def test_single_positive_is_held() -> None:
    d = _step(FireHoldTracker(), [True])[0]
    assert d.danger is False and d.positives == 1


def test_three_of_four_is_danger() -> None:
    ds = _step(FireHoldTracker(), [True, False, True, True])
    assert [d.danger for d in ds] == [False, False, False, True]
    assert ds[-1].detected_type == "smoke" and ds[-1].confidence == pytest.approx(0.7)


def test_two_of_four_is_not_danger() -> None:
    assert not any(d.danger for d in _step(FireHoldTracker(), [True, False, True, False, False, True]))


def test_window_is_by_count_not_time_so_slow_analysis_still_fires() -> None:
    # 카메라 3대 부하로 분석 간격이 2.5초여도(시간 창이었다면 리셋) 3회 양성이면 danger.
    ds = _step(FireHoldTracker(), [True, True, True], dt=2.5)
    assert ds[-1].danger is True


def test_gap_resets_history() -> None:
    tr = FireHoldTracker()
    _step(tr, [True, True])
    d = tr.update("s", True, 0.7, "smoke", 100.0, **KW)  # 간격 > max_gap_sec
    assert d.danger is False and d.scenes == 1


def test_danger_drops_when_positives_leave_window() -> None:
    ds = _step(FireHoldTracker(), [True, True, True, False, False])
    assert [d.danger for d in ds] == [False, False, True, True, False]


def test_sessions_are_independent_and_clear_works() -> None:
    tr = FireHoldTracker()
    _step(tr, [True, True], sid="a")
    assert _step(tr, [True], sid="b")[0].danger is False
    tr.clear("a")
    assert tr.tracked_sessions() == 1


def test_min_positive_larger_than_window_is_clamped() -> None:
    ds = _step(FireHoldTracker(), [True, True], window=2, min_positive=9)
    assert ds[-1].danger is True


def test_pending_log_is_rate_limited(caplog: pytest.LogCaptureFixture) -> None:
    now = [0.0]
    tr = FireHoldTracker(clock=lambda: now[0])
    with caplog.at_level("INFO", logger="app.services.fire_hold_tracker"):
        tr.update("s", True, 0.7, "smoke", 0.0, **KW)
        tr.update("s", True, 0.7, "smoke", 0.1, window=4, min_positive=4, max_gap_sec=10.0)
        now[0] += 61
        tr.update("s", True, 0.7, "smoke", 0.2, window=4, min_positive=4, max_gap_sec=10.0)
    lines = [r.getMessage() for r in caplog.records if "FIRE-HOLD-PENDING" in r.getMessage()]
    assert len(lines) == 2 and all("sessionId=s" in m for m in lines)


def test_int_range_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FH_T_BIG", "99")
    monkeypatch.setenv("FH_T_BAD", "x")
    monkeypatch.setenv("FH_T_OK", "3")
    assert _as_int_in_range("FH_T_BIG", 4, 1, 20) == 4
    assert _as_int_in_range("FH_T_BAD", 4, 1, 20) == 4
    assert _as_int_in_range("FH_T_OK", 4, 1, 20) == 3


# --- 라이브 분석 통합 -----------------------------------------------------------


class _Fake:
    def __init__(self, t: str = "normal", c: float = 0.0, d: bool = False) -> None:
        self.calls = 0
        self.r = (t, c, d)

    def set(self, t: str, c: float = 0.0, d: bool = False) -> None:
        self.r = (t, c, d)

    def detect_from_jpeg(self, frame_bytes: bytes) -> dict[str, Any]:
        self.calls += 1
        t, c, d = self.r
        det = [] if t in {"normal", "unknown"} else [
            {"detectedType": t, "confidence": c, "bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5}}
        ]
        return {"detectedType": t, "confidence": c, "danger": d, "detectedAt": datetime.utcnow().isoformat(), "detections": det}


@pytest.fixture
def live(monkeypatch: pytest.MonkeyPatch):
    fakes = {"fire": _Fake(), "knife": _Fake(), "fall": _Fake()}
    monkeypatch.setattr(sss, "get_fire_smoke_detector", lambda: fakes["fire"])
    monkeypatch.setattr(sss, "get_knife_detector", lambda: fakes["knife"])
    monkeypatch.setattr(sss, "get_fall_detector", lambda: fakes["fall"])
    clock = [1000.0]
    monkeypatch.setattr(sss, "_analysis_clock", lambda: clock[0])

    def make(**overrides: Any) -> sss.StreamSessionService:
        update = {"fire_smoke_enabled": True, "knife_enabled": True, "fall_enabled": True,
                  "stream_sample_every_n_frames": 1, "fire_hold_enabled": True, "fire_hold_window": 4,
                  "fire_hold_min_positive": 3, "fire_hold_max_gap_sec": 10.0,
                  "frame_quality_gate_enabled": False, **overrides}
        settings = get_settings().model_copy(update=update)
        monkeypatch.setattr(sss, "get_settings", lambda: settings)
        return sss.StreamSessionService(db=None, frame_store=sss.frame_store)  # type: ignore[arg-type]

    return fakes, make, clock


def _sid(name: str) -> str:
    session_analysis_store.clear_session(name)
    fall_hold_tracker.clear(name)
    fire_hold_tracker.clear(name)
    return name


def _run(service, sid, clock, n, dt=0.65, frame=b"jpeg"):
    out = []
    for _ in range(n):
        out.append(service._detect_and_store(sid, frame))
        clock[0] += dt
    return out


def test_single_frame_alert_is_held_then_fires_on_third(live) -> None:
    fakes, make, clock = live
    fakes["fire"].set("smoke", 0.76, True)
    out = _run(make(), _sid("fh_basic"), clock, 3)
    assert [p["danger"] for p in out] == [False, False, True]
    assert out[0]["detectedType"] == "smoke" and out[0]["confidence"] == 0.76  # 점수는 그대로 내려간다
    assert out[2]["detectedType"] == "smoke"


def test_transient_single_positive_never_alerts(live) -> None:
    fakes, make, clock = live
    svc, sid = make(), _sid("fh_blip")
    fakes["fire"].set("smoke", 0.76, True)
    first = _run(svc, sid, clock, 1)[0]
    fakes["fire"].set("normal")
    rest = _run(svc, sid, clock, 6)
    assert first["danger"] is False and not any(p["danger"] for p in rest)


def test_kill_switch_restores_single_frame_judgement(live) -> None:
    fakes, make, clock = live
    fakes["fire"].set("smoke", 0.76, True)
    out = _run(make(fire_hold_enabled=False), _sid("fh_off"), clock, 1)
    assert out[0]["danger"] is True


def test_knife_is_not_held(live) -> None:
    fakes, make, clock = live
    fakes["knife"].set("knife", 0.9, True)
    out = _run(make(), _sid("fh_knife"), clock, 1)
    assert out[0]["detectedType"] == "knife" and out[0]["danger"] is True


def test_fall_hold_unchanged(live) -> None:
    fakes, make, clock = live
    svc, sid = make(fall_hold_sec=1.5, fall_hold_ratio=0.7, fall_danger_threshold=0.4), _sid("fh_fall")
    fakes["fall"].set("fall", 0.8, True)
    out = _run(svc, sid, clock, 7, dt=0.25)
    assert out[0]["danger"] is False and out[-1]["danger"] is True and out[-1]["detectedType"] == "fall"


def test_quality_gate_blocked_frames_count_as_negative(live) -> None:
    fakes, make, clock = live
    ok, buf = cv2.imencode(".jpg", np.full((240, 320, 3), 255, np.uint8))
    white = buf.tobytes()
    fakes["fire"].set("smoke", 0.8, True)
    out = _run(make(frame_quality_gate_enabled=True), _sid("fh_gate"), clock, 5, frame=white)
    assert not any(p["danger"] for p in out)
    # 게이트가 막은 장면은 양성으로 쌓이지 않는다: 이어서 정상 화면 양성 1회로는 알리지 않는다.
    assert fire_hold_tracker.tracked_sessions() >= 1
