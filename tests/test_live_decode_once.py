"""라이브 분석 디코드 1회화 — 프레임당 한 번 디코드한 배열을 감지기 3개·품질 검사가 함께 쓴다.

결과(종류·신뢰도·danger·품질 판정)는 감지기마다 따로 디코드하던 때와 같아야 한다.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest

from app.core.config import get_settings
from app.services import fire_smoke_detection_service as fsd
from app.services import frame_quality as fq
from app.services import stream_session_service as sss
from app.services import yolo_class_detector as ycd
from app.services.fall_detection_service import FallDetectionService
from app.services.fall_hold_tracker import fall_hold_tracker
from app.services.fire_hold_tracker import fire_hold_tracker
from app.services.fire_smoke_detection_service import FireSmokeDetectionService
from app.services.frame_decode import decode_bgr
from app.services.knife_detection_service import KnifeDetectionService
from app.services.session_analysis_store import session_analysis_store

T = fq.QualityThresholds(white_ratio=0.5, black_ratio=0.95, std_min=12.0, lap_max=10.0, blur_std_max=30.0)


def _jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()


def _scene(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((480, 640, 3), rng.integers(50, 180, 3), np.uint8)
    for _ in range(10):
        x, y = (int(v) for v in rng.integers(0, 540, 2))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(img, (x, y % 380), (x + int(rng.integers(30, 200)), y % 380 + int(rng.integers(30, 200))), color, -1)
    return img


SCENE = _jpeg(_scene())


# --- 감지기: detect_from_jpeg == detect_from_frame(decode_bgr) ---------------------------------


class _Box:
    def __init__(self, cls_id: int, score: float) -> None:
        self.xyxy = [_T([1.0, 2.0, 30.0, 40.0])]
        self.conf = [_T(score)]
        self.cls = [_T(float(cls_id))]

    def __len__(self) -> int:
        return 1


class _T:
    def __init__(self, v: Any) -> None:
        self.v = v

    def item(self) -> Any:
        return self.v

    def tolist(self) -> Any:
        return self.v


class _Result:
    def __init__(self, cls_id: int, score: float) -> None:
        self.boxes = _Box(cls_id, score)


class _Model:
    """predict 입력을 기록하는 가짜 YOLO. 입력 배열을 바꾸지 않는지도 본다."""

    def __init__(self, cls_id: int, score: float) -> None:
        self.inputs: list[np.ndarray] = []
        self._r = (cls_id, score)

    def predict(self, source: np.ndarray, **_: Any) -> list[_Result]:
        self.inputs.append(source)
        return [_Result(*self._r)]


class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _Torch:
    cuda = _Cuda()


def _strip(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k != "detectedAt"}


def _loaded(detector: Any, cls_name: str, score: float) -> _Model:
    model = _Model(0, score)
    detector._model = model
    detector._class_names = {0: cls_name}
    detector._loaded = True
    return model


@pytest.fixture
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ycd, "torch", _Torch())
    monkeypatch.setattr(fsd, "torch", _Torch())


def _detectors() -> list[tuple[Any, str, float]]:
    s = get_settings().model_copy(update={"fire_smoke_enabled": True, "knife_enabled": True, "fall_enabled": True})
    return [
        (FireSmokeDetectionService(s), "fire", 0.91),
        (KnifeDetectionService(s), "knife", 0.77),
        (FallDetectionService(s), "fallen", 0.66),
    ]


@pytest.mark.parametrize("frame_bytes", [SCENE, b"not-a-jpeg"], ids=["scene", "garbage"])
def test_frame_input_matches_jpeg_input(fake_torch: None, frame_bytes: bytes) -> None:
    for detector, cls_name, score in _detectors():
        _loaded(detector, cls_name, score)
        by_jpeg = detector.detect_from_jpeg(frame_bytes)
        by_frame = detector.detect_from_frame(decode_bgr(frame_bytes))
        assert _strip(by_jpeg) == _strip(by_frame), detector


def test_empty_bytes_still_unknown_even_when_disabled() -> None:
    s = get_settings().model_copy(update={"fire_smoke_enabled": False, "knife_enabled": False, "fall_enabled": False})
    for detector in (FireSmokeDetectionService(s), KnifeDetectionService(s), FallDetectionService(s)):
        assert detector.detect_from_jpeg(b"")["detectedType"] == "unknown"


def test_unloaded_model_reports_load_error_before_decode_failure(tmp_path) -> None:  # noqa: ANN001
    s = get_settings().model_copy(update={"knife_enabled": True, "knife_model_path": str(tmp_path / "nope.pt")})
    detector = KnifeDetectionService(s)
    detector.try_load()
    by_jpeg = detector.detect_from_jpeg(b"not-a-jpeg")
    by_frame = detector.detect_from_frame(None)
    assert _strip(by_jpeg) == _strip(by_frame)
    assert by_frame["detectedType"] == "unknown" and by_frame["loadError"]


def test_detectors_do_not_modify_shared_frame(fake_torch: None) -> None:
    frame = decode_bgr(SCENE)
    assert frame is not None
    before = frame.copy()
    for detector, cls_name, score in _detectors():
        model = _loaded(detector, cls_name, score)
        detector.detect_from_frame(frame)
        assert model.inputs[0] is frame  # 복사하지 않고 같은 배열을 넘긴다
    assert np.array_equal(frame, before)


# --- 품질 검사: 색 변환 흑백 == 흑백 디코드 --------------------------------------------------


def _quality_frames() -> list[bytes]:
    rng = np.random.default_rng(7)
    frames = [SCENE, _jpeg(np.full((480, 640, 3), 255, np.uint8)), _jpeg(np.zeros((480, 640, 3), np.uint8))]
    for v in (238, 239, 240, 241):  # 흰색 기준 240 근처, 비율 0.5 근처, 색 섞음
        for frac in (0.49, 0.5, 0.51):
            img = np.full((480, 640, 3), 100, np.int16)
            img[:, : int(640 * frac)] = v
            frames.append(_jpeg(np.clip(img + np.array((8, -4, -4)), 0, 255).astype(np.uint8)))
    for v in (24, 25, 26):  # 검은색 기준 25 근처, 비율 0.95 근처
        for frac in (0.94, 0.95, 0.96):
            img = np.full((480, 640, 3), 120, np.uint8)
            img[:, : int(640 * frac)] = v
            frames.append(_jpeg(img))
    for std in np.linspace(10, 14, 9):  # 밝기 표준편차 12 근처
        frames.append(_jpeg(np.clip(128 + rng.normal(0, std, (480, 640, 3)), 0, 255).astype(np.uint8)))
    for blur in (2, 4, 8):  # 흐린 단색(선명도·표준편차 30 근처)
        small = np.clip(128 + rng.normal(0, 90, (24, 32, 3)), 0, 255).astype(np.uint8)
        frames.append(_jpeg(cv2.GaussianBlur(cv2.resize(small, (640, 480), interpolation=cv2.INTER_CUBIC), (0, 0), blur)))
    return frames


def test_quality_verdict_same_as_grayscale_decode() -> None:
    for frame_bytes in _quality_frames():
        assert fq.assess_bgr(decode_bgr(frame_bytes), T) == fq.assess_jpeg(frame_bytes, T)


def test_quality_undecodable_frame_is_none() -> None:
    assert fq.assess_bgr(None, T) is None
    assert fq.assess_bgr(decode_bgr(b"not-a-jpeg"), T) is None


# --- _detect_and_store: 프레임당 디코드 1회 ----------------------------------------------------


class _Recorder:
    def __init__(self, detected_type: str = "normal", confidence: float = 0.0, danger: bool = False) -> None:
        self.frames: list[Any] = []
        self.r = {"detectedType": detected_type, "confidence": confidence, "danger": danger, "detections": []}

    def detect_from_frame(self, frame: Any) -> dict[str, Any]:
        self.frames.append(frame)
        return dict(self.r)

    def detect_from_jpeg(self, frame_bytes: bytes) -> dict[str, Any]:  # pragma: no cover - 불리면 안 된다
        raise AssertionError("라이브 경로는 감지기에 JPEG 를 다시 넘기지 않는다")


@pytest.fixture
def live(monkeypatch: pytest.MonkeyPatch):
    dets = {"fire": _Recorder(), "knife": _Recorder(), "fall": _Recorder()}
    monkeypatch.setattr(sss, "get_fire_smoke_detector", lambda: dets["fire"])
    monkeypatch.setattr(sss, "get_knife_detector", lambda: dets["knife"])
    monkeypatch.setattr(sss, "get_fall_detector", lambda: dets["fall"])
    decodes: list[bytes] = []

    def counting_decode(frame_bytes: bytes) -> Any:
        decodes.append(frame_bytes)
        return decode_bgr(frame_bytes)

    monkeypatch.setattr(sss, "decode_bgr", counting_decode)
    timings: list[dict[str, float]] = []
    monkeypatch.setattr(sss._inference_timing, "record", lambda t: timings.append(dict(t)))

    def make(**overrides: Any) -> sss.StreamSessionService:
        update = {"fire_smoke_enabled": True, "knife_enabled": True, "fall_enabled": True,
                  "stream_sample_every_n_frames": 1, "fire_hold_enabled": False, **overrides}
        settings = get_settings().model_copy(update=update)
        monkeypatch.setattr(sss, "get_settings", lambda: settings)
        return sss.StreamSessionService(db=None, frame_store=sss.frame_store)  # type: ignore[arg-type]

    return dets, decodes, timings, make


def _sid(name: str) -> str:
    session_analysis_store.clear_session(name)
    fall_hold_tracker.clear(name)
    fire_hold_tracker.clear(name)
    return name


def test_one_decode_per_frame_shared_by_all_detectors(live) -> None:
    dets, decodes, timings, make = live
    service = make()

    service._detect_and_store(_sid("decode_once"), SCENE)

    assert len(decodes) == 1
    shared = dets["fire"].frames[0]
    assert isinstance(shared, np.ndarray)
    assert dets["knife"].frames[0] is shared and dets["fall"].frames[0] is shared
    # 디코드·품질 검사 시간도 합계에 들어간다(감지기 밖으로 옮겨 숫자만 줄어 보이지 않게)
    assert {"decode", "quality", "fire", "knife", "fall"} <= set(timings[0])


def test_undecodable_frame_passes_none_and_skips_quality(live) -> None:
    dets, decodes, timings, make = live
    service = make()

    payload = service._detect_and_store(_sid("decode_fail"), b"not-a-jpeg")

    assert len(decodes) == 1
    assert all(d.frames == [None] for d in dets.values())
    assert "frameQuality" not in payload  # 판정 불가는 막지 않는다(기존과 같음)
    assert "quality" not in timings[0]


def test_quality_gate_uses_shared_frame(live) -> None:
    dets, decodes, _, make = live
    dets["fire"].r.update({"detectedType": "fire", "confidence": 0.9, "danger": True})
    service = make()

    payload = service._detect_and_store(_sid("decode_white"), _jpeg(np.full((480, 640, 3), 255, np.uint8)))

    assert len(decodes) == 1
    assert payload["frameQuality"] == {"ok": False, "reason": "white"}
    assert payload["danger"] is False


def test_decode_exception_is_treated_as_undecodable(live, monkeypatch: pytest.MonkeyPatch) -> None:
    dets, _, _, make = live

    def boom(_: bytes) -> Any:
        raise RuntimeError("decoder crashed")

    monkeypatch.setattr(sss, "decode_bgr", boom)
    payload = make()._detect_and_store(_sid("decode_boom"), SCENE)

    assert all(d.frames == [None] for d in dets.values())
    assert payload["danger"] is False
