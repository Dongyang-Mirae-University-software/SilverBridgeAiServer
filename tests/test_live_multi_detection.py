from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import pytest

from app.core.config import get_settings
from app.services import stream_session_service as sss
from app.services.fall_detection_service import FallDetectionService
from app.services.fall_hold_tracker import fall_hold_tracker
from app.services.knife_detection_service import KnifeDetectionService
from app.services.session_analysis_store import session_analysis_store


class _FakeDetector:
    """detect_from_jpeg 형태만 흉내 낸다. result 를 테스트가 프레임마다 바꾼다."""

    def __init__(self, detected_type: str = "normal", confidence: float = 0.0, danger: bool = False) -> None:
        self.calls = 0
        self.set(detected_type, confidence, danger)

    def set(self, detected_type: str, confidence: float = 0.0, danger: bool = False) -> None:
        self.result = {"detectedType": detected_type, "confidence": confidence, "danger": danger}

    def detect_from_jpeg(self, frame_bytes: bytes) -> dict[str, Any]:
        self.calls += 1
        detections = []
        if self.result["detectedType"] not in {"normal", "unknown"}:
            detections.append(
                {
                    "detectedType": self.result["detectedType"],
                    "confidence": self.result["confidence"],
                    "bbox": {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
                },
            )
        return {**self.result, "detectedAt": datetime.utcnow().isoformat(), "detections": detections}


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    detectors = {"fire": _FakeDetector(), "knife": _FakeDetector(), "fall": _FakeDetector()}
    monkeypatch.setattr(sss, "get_fire_smoke_detector", lambda: detectors["fire"])
    monkeypatch.setattr(sss, "get_knife_detector", lambda: detectors["knife"])
    monkeypatch.setattr(sss, "get_fall_detector", lambda: detectors["fall"])
    clock = _Clock()
    monkeypatch.setattr(sss, "_analysis_clock", clock)
    return {"detectors": detectors, "clock": clock, "monkeypatch": monkeypatch}


def _service(fakes: dict[str, Any], **overrides: Any) -> sss.StreamSessionService:
    update = {
        "fire_smoke_enabled": True,
        "knife_enabled": True,
        "fall_enabled": True,
        "stream_sample_every_n_frames": 1,
        "fall_danger_threshold": 0.4,
        "fall_hold_sec": 1.5,
        "fall_hold_ratio": 0.7,
        **overrides,
    }
    settings = get_settings().model_copy(update=update)
    fakes["monkeypatch"].setattr(sss, "get_settings", lambda: settings)
    return sss.StreamSessionService(db=None, frame_store=sss.frame_store)  # type: ignore[arg-type]


def _fresh(session_id: str) -> str:
    session_analysis_store.clear_session(session_id)
    fall_hold_tracker.clear(session_id)
    return session_id


def _hold_fall(service: sss.StreamSessionService, fakes: dict[str, Any], sid: str, score: float = 0.8) -> dict:
    """낙상 점수를 0.25초 간격으로 1.5초 동안 유지시켜 낙상 danger 를 만든다. 마지막 결과를 돌려준다."""
    payload: dict = {}
    for _ in range(7):
        fakes["detectors"]["fall"].set("fall", score, True)
        payload = service._detect_and_store(sid, b"jpeg")
        fakes["clock"].now += 0.25
    fakes["clock"].now -= 0.25
    return payload


# (a) 대표 선택 우선순위 -------------------------------------------------------------


def test_fire_danger_wins_over_knife_and_fall(fakes: dict[str, Any]) -> None:
    service = _service(fakes)
    sid = _fresh("multi_priority_fire")
    fakes["detectors"]["fire"].set("smoke", 0.62, True)
    fakes["detectors"]["knife"].set("knife", 0.95, True)
    payload = _hold_fall(service, fakes, sid, score=0.99)

    assert payload["detectedType"] == "smoke"  # 점수가 낮아도 화재가 앞선다
    assert payload["confidence"] == 0.62
    assert payload["danger"] is True
    kinds = {r["detectedType"]: r for r in payload["results"]}
    assert kinds["knife"]["danger"] is True and kinds["fall"]["danger"] is True


def test_knife_danger_wins_over_fall_danger(fakes: dict[str, Any]) -> None:
    service = _service(fakes)
    sid = _fresh("multi_priority_knife")
    fakes["detectors"]["fire"].set("fire", 0.5, False)
    fakes["detectors"]["knife"].set("knife", 0.55, True)
    payload = _hold_fall(service, fakes, sid, score=0.9)

    assert (payload["detectedType"], payload["danger"]) == ("knife", True)


def test_without_danger_highest_score_is_reported(fakes: dict[str, Any]) -> None:
    service = _service(fakes)
    sid = _fresh("multi_no_danger")
    fakes["detectors"]["fire"].set("smoke", 0.3, False)
    fakes["detectors"]["knife"].set("knife", 0.45, False)
    fakes["detectors"]["fall"].set("fall", 0.35, False)

    payload = service._detect_and_store(sid, b"jpeg")

    assert (payload["detectedType"], payload["confidence"], payload["danger"]) == ("knife", 0.45, False)
    assert len(payload["detections"]) == 3  # 모든 종류의 박스를 싣는다


def test_nothing_detected_is_normal_and_all_unavailable_is_unknown(fakes: dict[str, Any]) -> None:
    service = _service(fakes)
    sid = _fresh("multi_normal")
    fakes["detectors"]["fire"].set("unknown")
    payload = service._detect_and_store(sid, b"jpeg")
    assert (payload["detectedType"], payload["confidence"], payload["danger"]) == ("normal", 0.0, False)
    assert [r["available"] for r in payload["results"]] == [False, True, True]

    for detector in fakes["detectors"].values():
        detector.set("unknown")
    assert service._detect_and_store(sid, b"jpeg")["detectedType"] == "unknown"


def test_one_detector_crash_does_not_block_fire(fakes: dict[str, Any]) -> None:
    service = _service(fakes)
    sid = _fresh("multi_crash")
    fakes["detectors"]["fire"].set("fire", 0.9, True)

    def boom(_: bytes) -> dict:
        raise RuntimeError("knife model exploded")

    fakes["detectors"]["knife"].detect_from_jpeg = boom  # type: ignore[method-assign]
    payload = service._detect_and_store(sid, b"jpeg")

    assert (payload["detectedType"], payload["danger"]) == ("fire", True)
    assert payload["results"][1] == {"detectedType": "knife", "confidence": 0.0, "danger": False, "available": False}


# (b) 낙상 유지 조건 ------------------------------------------------------------------


def _fall_only(fakes: dict[str, Any]) -> sss.StreamSessionService:
    return _service(fakes, fire_smoke_enabled=False, knife_enabled=False)


def test_fall_single_spike_is_not_danger(fakes: dict[str, Any]) -> None:
    service = _fall_only(fakes)
    sid = _fresh("fall_spike")
    fall = fakes["detectors"]["fall"]
    dangers = []
    for i in range(12):  # 3초 관찰, 0.75초 지점에 한 장만 튄다
        if i == 3:
            fall.set("fall", 0.95, True)
        else:
            fall.set("normal")
        dangers.append(service._detect_and_store(sid, b"jpeg")["danger"])
        fakes["clock"].now += 0.25

    assert not any(dangers)


def test_fall_first_frame_alone_is_not_danger(fakes: dict[str, Any]) -> None:
    service = _fall_only(fakes)
    sid = _fresh("fall_first")
    fakes["detectors"]["fall"].set("fall", 0.99, True)

    payload = service._detect_and_store(sid, b"jpeg")

    assert payload["detectedType"] == "fall" and payload["danger"] is False  # 한 장 판정 True 를 뒤집는다


def test_fall_held_for_hold_sec_with_ratio_is_danger(fakes: dict[str, Any]) -> None:
    service = _fall_only(fakes)
    sid = _fresh("fall_hold")
    fall = fakes["detectors"]["fall"]
    dangers = []
    for _ in range(7):  # t = 0.00 ~ 1.50
        fall.set("fall", 0.6, True)
        dangers.append(service._detect_and_store(sid, b"jpeg")["danger"])
        fakes["clock"].now += 0.25

    assert dangers == [False] * 6 + [True]  # 1.5초를 채운 순간에만

    # 유지 구간 안에서 한 장을 놓쳐도(6/7 = 86% >= 70%) 낙상은 유지된다
    fall.set("normal")
    payload = service._detect_and_store(sid, b"jpeg")
    assert (payload["detectedType"], payload["danger"]) == ("fall", True)
    assert payload["confidence"] == 0.6  # 임계를 넘은 장면들의 평균


def test_fall_below_ratio_is_not_danger(fakes: dict[str, Any]) -> None:
    service = _fall_only(fakes)
    sid = _fresh("fall_ratio")
    fall = fakes["detectors"]["fall"]
    dangers = []
    for i in range(14):  # 절반만 넘는다(50% < 70%)
        if i % 2 == 0:
            fall.set("fall", 0.8, True)
        else:
            fall.set("fall", 0.2, False)
        dangers.append(service._detect_and_store(sid, b"jpeg")["danger"])
        fakes["clock"].now += 0.25

    assert not any(dangers)


def test_fall_threshold_is_configurable(fakes: dict[str, Any]) -> None:
    service = _service(fakes, fire_smoke_enabled=False, knife_enabled=False, fall_danger_threshold=0.7)
    sid = _fresh("fall_threshold")
    payload = _hold_fall(service, fakes, sid, score=0.6)
    assert payload["danger"] is False


def test_fall_gap_restarts_observation(fakes: dict[str, Any]) -> None:
    service = _fall_only(fakes)
    sid = _fresh("fall_gap")
    fall = fakes["detectors"]["fall"]
    fall.set("fall", 0.9, True)
    for _ in range(4):  # 0.75초 관찰
        service._detect_and_store(sid, b"jpeg")
        fakes["clock"].now += 0.25
    fakes["clock"].now += 2.0  # 분석이 2초 끊김(> FALL_HOLD_SEC)

    dangers = []
    for _ in range(7):
        dangers.append(service._detect_and_store(sid, b"jpeg")["danger"])
        fakes["clock"].now += 0.25

    assert dangers == [False] * 6 + [True]  # 끊긴 뒤 다시 1.5초를 채워야 한다


def test_stop_clears_fall_state(fakes: dict[str, Any]) -> None:
    service = _fall_only(fakes)
    sid = _fresh("fall_stop")
    service.create_or_restart(sid, "cam", "ipad")
    assert _hold_fall(service, fakes, sid)["danger"] is True

    service.stop(service.require_session(sid))
    assert sid not in fall_hold_tracker._sessions

    fakes["clock"].now += 0.25
    payload = service._detect_and_store(sid, b"jpeg")
    assert payload["danger"] is False  # 종료 전 이력을 이어 쓰지 않는다


# (c) 꺼진 감지기는 호출하지 않는다 -----------------------------------------------------


def test_disabled_detectors_are_not_called(fakes: dict[str, Any]) -> None:
    service = _service(fakes, knife_enabled=False, fall_enabled=False)
    sid = _fresh("disabled_calls")

    service.analyze_stream_frame(sid, b"jpeg")

    calls = {k: d.calls for k, d in fakes["detectors"].items()}
    assert calls == {"fire": 1, "knife": 0, "fall": 0}


def test_all_disabled_skips_analysis(fakes: dict[str, Any]) -> None:
    service = _service(fakes, fire_smoke_enabled=False, knife_enabled=False, fall_enabled=False)
    sid = _fresh("disabled_all")
    session_analysis_store.set_result(sid, {"detectedType": "normal", "danger": False})

    sync_result = service.analyze_stream_frame(sid, b"jpeg")
    async_result = asyncio.run(service.analyze_stream_frame_async(sid, b"jpeg"))

    assert sync_result == async_result == {"detectedType": "normal", "danger": False}
    assert all(d.calls == 0 for d in fakes["detectors"].values())


def test_fire_disabled_but_knife_enabled_still_analyzes(fakes: dict[str, Any]) -> None:
    service = _service(fakes, fire_smoke_enabled=False, fall_enabled=False)
    sid = _fresh("knife_only")
    fakes["detectors"]["knife"].set("knife", 0.7, True)

    result = asyncio.run(service.analyze_stream_frame_async(sid, b"jpeg"))

    assert (result["detectedType"], result["danger"]) == ("knife", True)
    assert fakes["detectors"]["fire"].calls == 0


# (d) 화재만 켜져 있으면 기존 응답 그대로 -----------------------------------------------


@pytest.mark.parametrize(
    ("detected_type", "confidence", "danger"),
    [("fire", 0.91, True), ("smoke", 0.4, False), ("normal", 0.0, False), ("unknown", 0.0, False)],
)
def test_fire_only_payload_matches_previous_contract(
    fakes: dict[str, Any],
    detected_type: str,
    confidence: float,
    danger: bool,
) -> None:
    service = _service(fakes, knife_enabled=False, fall_enabled=False)
    sid = _fresh(f"fire_contract_{detected_type}")
    fire = fakes["detectors"]["fire"]
    fire.set(detected_type, confidence, danger)
    raw = fire.detect_from_jpeg(b"jpeg")

    payload = service._detect_and_store(sid, b"jpeg")

    # 이전 _detect_and_store 가 만들던 값과 같다(results 는 추가 필드)
    expected = {
        "detectedType": raw["detectedType"],
        "confidence": raw["confidence"],
        "danger": raw["danger"],
        "detections": raw["detections"],
    }
    assert {k: payload[k] for k in expected} == expected
    assert isinstance(payload["analyzedAt"], str)
    assert set(payload) == {"detectedType", "confidence", "danger", "detections", "analyzedAt", "results"}
    assert session_analysis_store.get_result(sid) == payload


# 감지기 클래스명 처리 ----------------------------------------------------------------


class _Scalar:
    def __init__(self, value: float) -> None:
        self.value = value

    def item(self) -> float:
        return self.value


class _Row:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


class _Boxes:
    def __init__(self, items: list[tuple[int, float]]) -> None:
        self.cls = [_Scalar(c) for c, _ in items]
        self.conf = [_Scalar(s) for _, s in items]
        self.xyxy = [_Row([1.0, 2.0, 3.0, 4.0]) for _ in items]

    def __len__(self) -> int:
        return len(self.cls)


class _Result:
    def __init__(self, items: list[tuple[int, float]]) -> None:
        self.boxes = _Boxes(items)


def test_knife_accepts_only_knife_class_case_insensitively() -> None:
    detector = KnifeDetectionService(get_settings())
    detector._class_names = {0: "knife", 1: "knife_handle"}  # try_load 가 소문자로 만든 상태(6월 모델: Knife·Knife_Handle)

    detections, best_type, best_conf = detector._parse_boxes([_Result([(1, 0.9), (0, 0.6)])])

    assert (best_type, best_conf) == ("knife", 0.6)  # 손잡이 0.9 는 버린다
    assert [d["detectedType"] for d in detections] == ["knife"]


def test_fall_maps_fallen_class_to_fall_type() -> None:
    detector = FallDetectionService(get_settings())
    detector._class_names = {0: "fallen"}

    detections, best_type, _ = detector._parse_boxes([_Result([(0, 0.5)])])

    assert best_type == "fall"  # 백엔드는 fallen 을 모른다
    assert detections[0]["detectedType"] == "fall"


def test_missing_model_does_not_raise_and_reports_unknown(tmp_path) -> None:  # noqa: ANN001
    settings = get_settings().model_copy(
        update={"knife_enabled": True, "knife_model_path": str(tmp_path / "nope.pt")},
    )
    detector = KnifeDetectionService(settings)

    detector.try_load()
    result = detector.detect_from_jpeg(b"jpeg")

    assert detector.loaded is False and detector.load_error
    assert (result["detectedType"], result["danger"]) == ("unknown", False)


def test_disabled_detector_reports_env_name() -> None:
    detector = FallDetectionService(get_settings().model_copy(update={"fall_enabled": False}))
    detector.try_load()
    assert detector.load_error == "FALL_ENABLED=false"
