from __future__ import annotations

import time
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import pytest

from app.core.config import Settings, get_settings
from app.services import frame_quality as fq
from app.services import stream_session_service as sss
from app.services.fall_hold_tracker import fall_hold_tracker
from app.services.session_analysis_store import session_analysis_store

T = fq.QualityThresholds(white_ratio=0.5, black_ratio=0.95, std_min=12.0, lap_max=10.0, blur_std_max=30.0)


def _jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return buf.tobytes()


def _room(seed: int = 0) -> np.ndarray:
    """질감 있는 일반 실내 화면(블록 잡음 + 부드러운 그라데이션)."""
    rng = np.random.default_rng(seed)
    base = np.tile(np.linspace(60, 170, 960, dtype=np.float32), (720, 1))
    noise = cv2.resize(rng.normal(0, 28, (36, 48)).astype(np.float32), (960, 720), interpolation=cv2.INTER_NEAREST)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _fire_like() -> np.ndarray:
    img = cv2.cvtColor(_room(1), cv2.COLOR_GRAY2BGR)
    cv2.ellipse(img, (480, 420), (140, 220), 0, 0, 360, (30, 120, 255), -1)
    cv2.ellipse(img, (480, 460), (70, 120), 0, 0, 360, (200, 255, 255), -1)
    return img


# --- 순수 함수 ------------------------------------------------------------------


def test_white_screen_is_bad() -> None:
    v = fq.assess_gray(np.full((720, 960), 255, np.uint8), T)
    assert (v.ok, v.reason) == (False, "white")


def test_mostly_white_with_some_texture_is_bad() -> None:
    img = _room()
    img[:, :600] = 250  # 62% 흰색
    assert fq.assess_gray(img, T).reason == "white"


def test_black_screen_is_bad() -> None:
    v = fq.assess_gray(np.zeros((720, 960), np.uint8), T)
    assert (v.ok, v.reason) == (False, "black")


def test_flat_gray_screen_is_bad() -> None:
    v = fq.assess_gray(np.full((720, 960), 128, np.uint8), T)
    assert (v.ok, v.reason) == (False, "flat")


def test_blur_flat_is_bad() -> None:
    # std 가 12 이상 30 미만이고 질감이 거의 없는 화면(완만한 그라데이션 = 렌즈를 가린 손의 번짐).
    img = np.tile(np.linspace(80, 150, 960, dtype=np.float32), (720, 1)).astype(np.uint8)
    assert 12 <= img.std() < 30
    v = fq.assess_gray(img, T)
    assert (v.ok, v.reason) == (False, "blur_flat")


def test_normal_room_passes() -> None:
    assert fq.assess_gray(_room(), T).ok


def test_fire_photo_passes() -> None:
    assert fq.assess_jpeg(_jpeg(_fire_like()), T).ok  # type: ignore[union-attr]


def test_blurred_but_textured_scene_is_not_blocked() -> None:
    # 흐린 화면(연기 포함)은 막지 않는다: 질감이 줄어도 대비(std)가 30 이상이면 통과.
    img = cv2.GaussianBlur(_fire_like(), (0, 0), 6)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert gray.std() >= 30
    assert fq.assess_gray(gray, T).ok


def test_undecodable_bytes_return_none() -> None:
    assert fq.assess_jpeg(b"not-a-jpeg", T) is None
    assert fq.assess_jpeg(b"", T) is None


def test_processing_time_for_640_gray() -> None:
    img = _room()
    jpeg = _jpeg(img)
    fq.assess_jpeg(jpeg, T)
    started = time.perf_counter()
    for _ in range(50):
        fq.assess_gray(img, T)
    per_ms = (time.perf_counter() - started) / 50 * 1000
    print(f"[frame-quality] assess_gray 960x720 -> 640: {per_ms:.2f} ms/frame")
    assert per_ms < 10  # 목표 1ms(640 흑백)이나 CI 노이즈 여유. 실측값은 -s 로 출력한다.


def test_range_helper_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.config import _as_float_in_range

    monkeypatch.setenv("FQ_T_HIGH", "7")
    monkeypatch.setenv("FQ_T_BAD", "abc")
    monkeypatch.setenv("FQ_T_NEG", "-3")
    monkeypatch.setenv("FQ_T_OK", "0.9")
    assert _as_float_in_range("FQ_T_HIGH", 0.5, 0.05, 1.0) == 0.5
    assert _as_float_in_range("FQ_T_BAD", 12.0, 0.0, 60.0) == 12.0
    assert _as_float_in_range("FQ_T_NEG", 10.0, 0.0, 500.0) == 10.0
    assert _as_float_in_range("FQ_T_OK", 0.95, 0.05, 1.0) == 0.9
    assert _as_float_in_range("FQ_T_UNSET", 0.5, 0.05, 1.0) == 0.5
    assert Settings().frame_quality_gate_enabled is True  # 기본 켜짐


def test_block_log_is_limited_to_once_per_minute(caplog: pytest.LogCaptureFixture) -> None:
    now = [100.0]
    limiter = fq._BlockLogLimiter(60.0, clock=lambda: now[0])
    with caplog.at_level("INFO", logger=fq.__name__):
        limiter.log_blocked("s1", "white")
        limiter.log_blocked("s1", "white")
        limiter.log_blocked("s2", "black")
        now[0] += 61
        limiter.log_blocked("s1", "white")
    lines = [r.getMessage() for r in caplog.records if "FRAME-QUALITY-BLOCKED" in r.getMessage()]
    assert len(lines) == 3
    assert all("sessionId=" in m and "reason=" in m for m in lines)


# --- 라이브 분석 통합 -----------------------------------------------------------


class _Fake:
    def __init__(self, detected_type: str, confidence: float, danger: bool) -> None:
        self.calls = 0
        self.r = (detected_type, confidence, danger)

    def detect_from_jpeg(self, frame_bytes: bytes) -> dict[str, Any]:
        self.calls += 1
        t, c, d = self.r
        det = [] if t in {"normal", "unknown"} else [
            {"detectedType": t, "confidence": c, "bbox": {"x1": 0, "y1": 0, "x2": 5, "y2": 5}}
        ]
        return {"detectedType": t, "confidence": c, "danger": d, "detectedAt": datetime.utcnow().isoformat(), "detections": det}


@pytest.fixture
def live(monkeypatch: pytest.MonkeyPatch):
    fakes = {"fire": _Fake("fire", 0.9, True), "knife": _Fake("knife", 0.8, True), "fall": _Fake("fall", 0.7, False)}
    monkeypatch.setattr(sss, "get_fire_smoke_detector", lambda: fakes["fire"])
    monkeypatch.setattr(sss, "get_knife_detector", lambda: fakes["knife"])
    monkeypatch.setattr(sss, "get_fall_detector", lambda: fakes["fall"])

    def make(**overrides: Any) -> sss.StreamSessionService:
        update = {"fire_smoke_enabled": True, "knife_enabled": True, "fall_enabled": True,
                  "stream_sample_every_n_frames": 1, "fire_hold_enabled": False, "knife_hold_enabled": False,
                  **overrides}
        settings = get_settings().model_copy(update=update)
        monkeypatch.setattr(sss, "get_settings", lambda: settings)
        return sss.StreamSessionService(db=None, frame_store=sss.frame_store)  # type: ignore[arg-type]

    return fakes, make


def _sid(name: str) -> str:
    session_analysis_store.clear_session(name)
    fall_hold_tracker.clear(name)
    return name


WHITE = _jpeg(np.full((480, 640, 3), 255, np.uint8))
GOOD = _jpeg(_fire_like())


def test_bad_frame_clears_fire_danger_and_boxes(live) -> None:
    fakes, make = live
    fakes["knife"].r = ("normal", 0.0, False)
    fakes["fall"].r = ("normal", 0.0, False)
    payload = make()._detect_and_store(_sid("fq_white"), WHITE)
    assert payload["danger"] is False
    assert payload["detections"] == []
    assert payload["frameQuality"] == {"ok": False, "reason": "white"}
    fire = next(r for r in payload["results"] if "frameQuality" in r)
    assert fire["detectedType"] == "fire"  # 비감지 항목은 종류 이름으로 표기된다(기존 규칙)
    assert fire["danger"] is False


def test_good_frame_keeps_fire_danger(live) -> None:
    fakes, make = live
    payload = make()._detect_and_store(_sid("fq_good"), GOOD)
    assert payload["detectedType"] == "fire" and payload["danger"] is True
    assert payload["frameQuality"] == {"ok": True, "reason": None}


def test_knife_danger_unchanged_on_bad_frame(live) -> None:
    fakes, make = live
    payload = make()._detect_and_store(_sid("fq_knife"), WHITE)
    assert payload["detectedType"] == "knife" and payload["danger"] is True  # 흉기는 게이트 대상이 아니다
    assert fakes["knife"].calls == 1


def test_kill_switch_makes_gate_pass_through(live) -> None:
    _, make = live
    payload = make(frame_quality_gate_enabled=False)._detect_and_store(_sid("fq_off"), WHITE)
    assert payload["detectedType"] == "fire" and payload["danger"] is True
    assert "frameQuality" not in payload


def test_undecodable_frame_is_not_blocked(live) -> None:
    _, make = live
    payload = make()._detect_and_store(_sid("fq_raw"), b"jpeg")
    assert payload["danger"] is True and "frameQuality" not in payload


def test_unknown_fire_result_is_kept_unknown(live) -> None:
    fakes, make = live
    fakes["fire"].r = ("unknown", 0.0, False)
    fakes["knife"].r = ("normal", 0.0, False)
    fakes["fall"].r = ("normal", 0.0, False)
    payload = make()._detect_and_store(_sid("fq_unknown"), WHITE)
    fire = next(r for r in payload["results"] if "frameQuality" in r)
    assert fire["available"] is False


def test_laplacian_f32_verdict_matches_f64_reference() -> None:
    # 속도 개선(CV_64F -> CV_32F)이 판정을 바꾸지 않는다: lap_max 근처 화면 포함 400장에서 기존 계산과 비교.
    rng = np.random.default_rng(3)

    def reference(gray: np.ndarray) -> bool:
        h, w = gray.shape
        g = cv2.resize(gray, (640, max(1, int(h * 640 / w))), interpolation=cv2.INTER_AREA) if w > 640 else gray
        if g.std() < T.blur_std_max and float(cv2.Laplacian(g, cv2.CV_64F).var()) < T.lap_max and g.std() >= T.std_min:
            return False
        return True

    mismatches = 0
    for i in range(400):
        base, amp, noise = rng.uniform(40, 200), rng.uniform(0, 14), rng.uniform(0.2, 3.0)
        y, x = np.mgrid[0:360, 0:640]
        g = np.clip(base + amp * np.sin(x / 40.0) * np.cos(y / 30.0) + rng.normal(0, noise, (360, 640)), 0, 255).astype(np.uint8)
        v = fq.assess_gray(g, T)
        # reference 는 white/black/flat 단계를 생략하므로 blur_flat 여부만 비교한다.
        ref_blur = not reference(g)
        if (v.reason == fq.REASON_BLUR_FLAT) != ref_blur and v.reason in (None, fq.REASON_BLUR_FLAT):
            mismatches += 1
    assert mismatches == 0


def test_quality_time_is_recorded_in_live_infer_timings(live, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import stream_session_service as sss_mod

    recorded: list[dict[str, float]] = []
    monkeypatch.setattr(sss_mod._inference_timing, "record", lambda t: recorded.append(dict(t)))
    fakes, make = live
    make(frame_quality_gate_enabled=True)._detect_and_store(_sid("fq_timing"), GOOD)
    assert "quality" in recorded[-1] and recorded[-1]["quality"] >= 0.0
    assert {"fire", "knife", "fall"} <= set(recorded[-1])


def test_quality_time_is_absent_when_gate_is_off(live, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.services import stream_session_service as sss_mod

    recorded: list[dict[str, float]] = []
    monkeypatch.setattr(sss_mod._inference_timing, "record", lambda t: recorded.append(dict(t)))
    fakes, make = live
    make(frame_quality_gate_enabled=False)._detect_and_store(_sid("fq_timing_off"), GOOD)
    assert "quality" not in recorded[-1]
