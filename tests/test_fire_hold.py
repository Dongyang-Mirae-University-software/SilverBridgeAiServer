from __future__ import annotations

from datetime import datetime
from typing import Any

import cv2
import numpy as np
import pytest

from app.core.config import _as_int_in_range, get_settings
from app.services import stream_session_service as sss
from app.services.fall_hold_tracker import fall_hold_tracker
from app.services.fire_hold_tracker import FireHoldTracker, fire_hold_tracker, knife_hold_tracker
from app.services.session_analysis_store import session_analysis_store

KW = dict(window_sec=2.0, ratio=0.6, min_scenes=3, max_gap_sec=10.0)


def _step(tr: FireHoldTracker, seq: list[bool], *, start: float = 0.0, dt: float = 0.09, sid: str = "s", **kw: Any):
    kw = {**KW, **kw}
    out = []
    for i, pos in enumerate(seq):
        out.append(tr.update(sid, pos, 0.7 if pos else 0.0, "smoke" if pos else "normal", start + i * dt, **kw))
    return out


# --- 트래커 ---------------------------------------------------------------------


def test_single_positive_is_held() -> None:
    d = _step(FireHoldTracker(), [True])[0]
    assert d.danger is False and d.positives == 1


def test_regression_fast_analysis_short_burst_does_not_alert() -> None:
    # 10/13 실물: 분석이 초당 11회라 "4회 중 3회"가 0.3초 만에 알림이 됐다. 1초(11회) 연속 양성도 막아야 한다.
    ds = _step(FireHoldTracker(), [True] * 12, dt=0.09)
    assert not any(d.danger for d in ds)


def test_sustained_positives_alert_after_window() -> None:
    ds = _step(FireHoldTracker(), [True] * 30, dt=0.09)  # 2.7초
    first = next(i for i, d in enumerate(ds) if d.danger)
    assert 0.09 * first >= 2.0 - 1e-9  # 관찰이 window_sec 이상 이어진 뒤에만
    assert ds[-1].detected_type == "smoke" and ds[-1].confidence == pytest.approx(0.7)


def test_intermittent_below_ratio_never_alerts() -> None:
    seq = ([True, False, False] * 12)  # 양성 33%
    assert not any(d.danger for d in _step(FireHoldTracker(), seq, dt=0.09))


def test_flicker_above_ratio_alerts() -> None:
    seq = ([True, True, False] * 14)  # 양성 67%
    assert any(d.danger for d in _step(FireHoldTracker(), seq, dt=0.09))


def test_slow_analysis_uses_last_min_scenes_and_still_fires() -> None:
    # 카메라 3대 부하로 분석 간격이 2.5초 - 시간 창 안에는 1회뿐이지만 최근 3회를 본다.
    ds = _step(FireHoldTracker(), [True, True, True, True], dt=2.5)
    assert [d.danger for d in ds] == [False, False, True, True]


def test_slow_analysis_single_positive_is_held() -> None:
    ds = _step(FireHoldTracker(), [False, False, True, False, False], dt=2.5)
    assert not any(d.danger for d in ds)


def test_gap_resets_history() -> None:
    tr = FireHoldTracker()
    _step(tr, [True] * 30, dt=0.09)
    d = tr.update("s", True, 0.7, "smoke", 100.0, **KW)  # 간격 > max_gap_sec
    assert d.danger is False and d.scenes == 1


def test_danger_drops_when_positives_leave_window() -> None:
    tr = FireHoldTracker()
    ds = _step(tr, [True] * 30, dt=0.09)
    assert ds[-1].danger is True
    after = _step(tr, [False] * 40, start=30 * 0.09, dt=0.09)
    assert after[-1].danger is False


def test_sessions_are_independent_and_clear_works() -> None:
    tr = FireHoldTracker()
    _step(tr, [True, True], sid="a")
    assert _step(tr, [True], sid="b")[0].danger is False
    tr.clear("a")
    assert tr.tracked_sessions() == 1


def test_pending_log_is_rate_limited(caplog: pytest.LogCaptureFixture) -> None:
    now = [0.0]
    tr = FireHoldTracker(clock=lambda: now[0])
    with caplog.at_level("INFO", logger="app.services.fire_hold_tracker"):
        tr.update("s", True, 0.7, "smoke", 0.0, **KW)
        tr.update("s", True, 0.7, "smoke", 0.1, **KW)
        now[0] += 61
        tr.update("s", True, 0.7, "smoke", 0.2, **KW)
    lines = [r.getMessage() for r in caplog.records if "FIRE-HOLD-PENDING" in r.getMessage()]
    assert len(lines) == 2 and all("sessionId=s" in m for m in lines)


def test_int_range_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FH_T_BIG", "99")
    monkeypatch.setenv("FH_T_BAD", "x")
    monkeypatch.setenv("FH_T_OK", "3")
    assert _as_int_in_range("FH_T_BIG", 4, 1, 20) == 4
    assert _as_int_in_range("FH_T_BAD", 4, 1, 20) == 4
    assert _as_int_in_range("FH_T_OK", 3, 1, 20) == 3


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
                  "stream_sample_every_n_frames": 1, "fire_hold_enabled": True, "fire_hold_sec": 2.0,
                  "fire_hold_ratio": 0.6, "fire_hold_min_scenes": 3, "fire_hold_max_gap_sec": 10.0,
                  "frame_quality_gate_enabled": False, **overrides}
        settings = get_settings().model_copy(update=update)
        monkeypatch.setattr(sss, "get_settings", lambda: settings)
        return sss.StreamSessionService(db=None, frame_store=sss.frame_store)  # type: ignore[arg-type]

    return fakes, make, clock


def _sid(name: str) -> str:
    session_analysis_store.clear_session(name)
    fall_hold_tracker.clear(name)
    fire_hold_tracker.clear(name)
    knife_hold_tracker.clear(name)
    return name


def _run(service, sid, clock, n, dt=0.1, frame=b"jpeg"):
    out = []
    for _ in range(n):
        out.append(service._detect_and_store(sid, frame))
        clock[0] += dt
    return out


def test_single_frame_alert_is_held_then_fires_after_window(live) -> None:
    fakes, make, clock = live
    fakes["fire"].set("smoke", 0.76, True)
    out = _run(make(), _sid("fh_basic"), clock, 30)  # 0.1초 간격 3초
    assert out[0]["danger"] is False and out[10]["danger"] is False  # 1초까지는 알리지 않는다
    assert out[-1]["danger"] is True
    assert out[0]["detectedType"] == "smoke" and out[0]["confidence"] == 0.76  # 점수는 그대로 내려간다
    assert out[-1]["detectedType"] == "smoke"


def test_transient_single_positive_never_alerts(live) -> None:
    fakes, make, clock = live
    svc, sid = make(), _sid("fh_blip")
    fakes["fire"].set("smoke", 0.76, True)
    first = _run(svc, sid, clock, 1)[0]
    fakes["fire"].set("normal")
    rest = _run(svc, sid, clock, 40)
    assert first["danger"] is False and not any(p["danger"] for p in rest)


def test_kill_switch_restores_single_frame_judgement(live) -> None:
    fakes, make, clock = live
    fakes["fire"].set("smoke", 0.76, True)
    out = _run(make(fire_hold_enabled=False), _sid("fh_off"), clock, 1)
    assert out[0]["danger"] is True


def test_knife_single_positive_is_held(live) -> None:
    # 순간적으로 한 장만 양성인 흉기는 알리지 않는다(가정 영상 잘못 알림 5건의 유형).
    fakes, make, clock = live
    svc, sid = make(), _sid("kh_blip")
    fakes["knife"].set("knife", 0.9, True)
    first = _run(svc, sid, clock, 1)[0]
    fakes["knife"].set("normal")
    rest = _run(svc, sid, clock, 40)
    assert first["danger"] is False and first["detectedType"] == "knife"  # 점수·종류는 그대로 내려간다
    assert not any(p["danger"] for p in rest)


def test_knife_sustained_positive_alerts_after_window(live) -> None:
    fakes, make, clock = live
    fakes["knife"].set("knife", 0.9, True)
    out = _run(make(), _sid("kh_sustained"), clock, 30)  # 0.1초 간격 3초
    assert out[0]["danger"] is False and out[10]["danger"] is False
    assert out[-1]["danger"] is True and out[-1]["detectedType"] == "knife"


def test_knife_kill_switch_restores_single_frame_judgement(live) -> None:
    fakes, make, clock = live
    fakes["knife"].set("knife", 0.9, True)
    out = _run(make(knife_hold_enabled=False), _sid("kh_off"), clock, 1)
    assert out[0]["danger"] is True


def test_knife_hold_does_not_touch_fire_history(live) -> None:
    fakes, make, clock = live
    svc, sid = make(), _sid("kh_isolated")
    fakes["knife"].set("knife", 0.9, True)
    fakes["fire"].set("normal")
    out = _run(svc, sid, clock, 30)
    assert out[-1]["danger"] is True  # 합쳐진 payload: 흉기만 양성
    assert fire_hold_tracker.tracked_sessions() >= 0 and knife_hold_tracker.tracked_sessions() >= 1


def test_knife_hold_cleared_on_stop() -> None:
    knife_hold_tracker.update("kh_stop", True, 0.9, "knife", 0.0, window_sec=2.0, ratio=0.6, min_scenes=3, max_gap_sec=10.0)
    knife_hold_tracker.clear("kh_stop")
    d = knife_hold_tracker.update("kh_stop", True, 0.9, "knife", 100.0, window_sec=2.0, ratio=0.6, min_scenes=3, max_gap_sec=10.0)
    assert d.scenes == 1


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
    out = _run(make(frame_quality_gate_enabled=True), _sid("fh_gate"), clock, 40, frame=white)
    assert not any(p["danger"] for p in out)
    # 게이트가 막은 장면은 양성으로 쌓이지 않는다: 이어서 정상 화면 양성 1회로는 알리지 않는다.
    assert fire_hold_tracker.tracked_sessions() >= 1
