from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import app.models.ai_model  # noqa: F401  (라우터가 끌고 오는 FK 대상 — test_clip_endpoint 와 같은 이유)
import app.models.camera  # noqa: F401
from app.core.config import get_settings
from app.core.response import error_response
from app.core.security import require_api_key
from app.routers import live_stream_router as router_module
from app.services import bbox_overlay as bo
from app.services.bbox_overlay import BboxOverlay, Box, active_boxes, draw_boxes, select_boxes
from app.services.clip_buffer import clip_frame_buffer
from app.services.clip_service import ffmpeg_available
from app.services.session_analysis_store import session_analysis_store
from app.services.stream_session_service import frame_store

API_KEY = "test-bbox-key"
HEADERS = {"X-API-Key": API_KEY}
NOW = datetime(2026, 10, 6, 12, 0, 0)


def _settings(**overrides):
    base = {
        "api_key": API_KEY,
        "fire_smoke_danger_threshold": 0.45,
        "knife_danger_threshold": 0.5,
        "fall_danger_threshold": 0.35,
        "live_draw_bbox": True,
        "live_bbox_hold_seconds": 1.0,
        "live_bbox_min_confidence": "danger",
        "live_bbox_draw_timeout_ms": 150,
    }
    return get_settings().model_copy(update={**base, **overrides})


@pytest.fixture
def use_settings(monkeypatch: pytest.MonkeyPatch):
    def apply(**overrides):
        s = _settings(**overrides)
        monkeypatch.setattr(bo, "get_settings", lambda: s)
        return s

    apply()
    return apply


def _det(kind: str, score: float, box=(10, 10, 60, 50)) -> dict:
    x1, y1, x2, y2 = box
    return {"detectedType": kind, "confidence": score, "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


def _fire_jpeg(w: int = 160, h: int = 120) -> bytes:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (30, 120, 255)  # 주황(BGR) - 파란 픽셀이 원래 없는 불 사진 대용
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    return buf.tobytes()


def _blue_mask(frame: np.ndarray) -> np.ndarray:
    b, g, r = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
    return (b > 180) & (g < 100) & (r < 100)


def _decode(jpeg: bytes) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert frame is not None
    return frame


def _store_result(session_id: str, detections: list[dict], analyzed_at: datetime | None = None) -> None:
    at = analyzed_at or NOW
    session_analysis_store.set_result(
        session_id,
        {"detectedType": "fire", "confidence": 0.9, "danger": True, "detections": detections,
         "analyzedAt": at.isoformat()},
    )


@pytest.fixture(autouse=True)
def _frozen_clock(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bo, "_utcnow", lambda: NOW)


# ---------------------------------------------------------------- 그리기

def test_draw_puts_blue_border_at_box_position() -> None:
    out = draw_boxes(_fire_jpeg(), (Box("fire", 0.73, 20, 30, 100, 90),))
    frame = _decode(out)
    mask = _blue_mask(frame)
    assert mask[60, 19:23].any() and mask[60, 99:102].any()  # 왼쪽·오른쪽 변
    assert mask[29:32, 60].any() or mask[88:92, 60].any()  # 위·아래 변
    assert not mask[50:70, 40:80].any()  # 박스 안쪽은 원본 그대로(불 색)


def test_label_stays_inside_frame_when_box_touches_top_edge() -> None:
    out = draw_boxes(_fire_jpeg(), (Box("smoke", 0.88, 5, 0, 100, 60),))
    mask = _blue_mask(_decode(out))
    # 라벨 배경이 박스 안쪽 위(y 0~)에 그려져 위쪽 가장자리 근처에 파란 면이 있다
    assert mask[1:10, 6:40].sum() > 100


def test_label_goes_above_box_when_there_is_room() -> None:
    out = draw_boxes(_fire_jpeg(), (Box("fire", 0.73, 20, 60, 100, 100),))
    mask = _blue_mask(_decode(out))
    assert mask[45:58, 22:50].sum() > 80  # 박스 위쪽에 라벨 배경


def test_out_of_range_coordinates_are_clipped_to_frame() -> None:
    original = _fire_jpeg()
    out = draw_boxes(original, (Box("fire", 0.9, -50, -20, 500, 400),))
    mask = _blue_mask(_decode(out))
    assert out != original and mask.any()


def test_degenerate_boxes_leave_bytes_untouched() -> None:
    original = _fire_jpeg()
    assert draw_boxes(original, ()) == original
    assert draw_boxes(original, (Box("fire", 0.9, 300, 10, 400, 50),)) == original  # 프레임 밖으로만 걸침
    assert draw_boxes(b"not-a-jpeg", (Box("fire", 0.9, 1, 1, 5, 5),)) == b"not-a-jpeg"


# ---------------------------------------------------------------- 표시 기준(위험 기준과 동일)

def test_fire_044_not_drawn_and_045_drawn(use_settings) -> None:
    s = use_settings()
    assert select_boxes([_det("fire", 0.44)], s) == ()
    assert len(select_boxes([_det("fire", 0.45)], s)) == 1
    assert len(select_boxes([_det("smoke", 0.45)], s)) == 1  # 연기도 화재 기준


def test_boundary_follows_danger_threshold_setting(use_settings) -> None:
    s = use_settings(fire_smoke_danger_threshold=0.6)
    assert select_boxes([_det("fire", 0.5)], s) == ()
    s = use_settings(fire_smoke_danger_threshold=0.3)
    assert len(select_boxes([_det("fire", 0.5)], s)) == 1


def test_fall_034_not_drawn_and_035_drawn_even_when_model_is_off(use_settings) -> None:
    s = use_settings(fall_enabled=False)
    assert select_boxes([_det("fall", 0.34)], s) == ()
    boxes = select_boxes([_det("fall", 0.35)], s)
    assert len(boxes) == 1 and boxes[0].label == "fall"


def test_knife_uses_its_own_threshold(use_settings) -> None:
    s = use_settings()
    assert select_boxes([_det("knife", 0.49)], s) == ()
    assert len(select_boxes([_det("knife", 0.5)], s)) == 1


def test_only_boxes_over_threshold_are_drawn_from_one_frame(use_settings) -> None:
    s = use_settings()
    boxes = select_boxes([_det("fire", 0.50, (0, 0, 20, 20)), _det("fire", 0.40, (30, 30, 60, 60))], s)
    assert [b.confidence for b in boxes] == [0.50]


def test_numeric_override_applies_to_every_kind(use_settings) -> None:
    s = use_settings(live_bbox_min_confidence="0.3")
    dets = [_det("fire", 0.31), _det("smoke", 0.31), _det("knife", 0.31), _det("fall", 0.31)]
    assert len(select_boxes(dets, s)) == 4
    assert select_boxes([_det("fire", 0.29), _det("knife", 0.29), _det("fall", 0.29)], s) == ()


def test_invalid_override_falls_back_to_danger_mode(use_settings) -> None:
    for raw in ("abc", "1.5", "-1", "nan"):
        s = use_settings(live_bbox_min_confidence=raw)
        assert select_boxes([_det("fire", 0.44)], s) == ()
        assert len(select_boxes([_det("fire", 0.45)], s)) == 1


def test_malformed_and_unknown_detections_are_skipped(use_settings) -> None:
    s = use_settings()
    dets = [
        "x", {"detectedType": "fire"}, _det("person", 0.99), _det("fire", 0.9, (50, 10, 10, 50)),
        {"detectedType": "fire", "confidence": "bad", "bbox": {"x1": 1, "y1": 1, "x2": 9, "y2": 9}},
        {"detectedType": "fire", "confidence": 0.9, "bbox": {"x1": float("nan"), "y1": 1, "x2": 9, "y2": 9}},
    ]
    assert select_boxes(dets, s) == ()
    assert select_boxes(None, s) == ()


# ---------------------------------------------------------------- 유지 시간

def test_hold_keeps_box_until_expiry(use_settings) -> None:
    s = use_settings(live_bbox_hold_seconds=1.0)
    _store_result("hold1", [_det("fire", 0.8)], NOW - timedelta(seconds=0.6))
    assert len(active_boxes("hold1", s)) == 1  # 유지 중
    _store_result("hold1", [_det("fire", 0.8)], NOW - timedelta(seconds=1.5))
    assert active_boxes("hold1", s) == ()  # 만료


def test_zero_detection_result_clears_immediately(use_settings) -> None:
    s = use_settings()
    _store_result("hold2", [_det("fire", 0.8)], NOW)
    assert len(active_boxes("hold2", s)) == 1
    _store_result("hold2", [], NOW)  # 감지 0건 결과가 오면 유지 시간이 남아 있어도 즉시 지운다
    assert active_boxes("hold2", s) == ()


def test_unusable_analyzed_at_draws_nothing(use_settings) -> None:
    s = use_settings()
    session_analysis_store.set_result("hold3", {"detections": [_det("fire", 0.8)]})
    assert active_boxes("hold3", s) == ()
    session_analysis_store.set_result("hold3", {"detections": [_det("fire", 0.8)], "analyzedAt": "garbage"})
    assert active_boxes("hold3", s) == ()
    assert active_boxes("no-result", s) == ()


def test_timezone_aware_analyzed_at_is_converted_to_utc(use_settings) -> None:
    s = use_settings()
    session_analysis_store.set_result(
        "hold4", {"detections": [_det("fire", 0.8)], "analyzedAt": (NOW - timedelta(seconds=0.2)).isoformat() + "+00:00"},
    )
    assert len(active_boxes("hold4", s)) == 1


# ---------------------------------------------------------------- 원본 보존·성능(6-1)

class _NoCv2:
    def __getattr__(self, name: str):
        raise AssertionError(f"cv2.{name} 호출 - 감지 없는 프레임은 디코딩·인코딩하면 안 된다")


def test_frame_without_detections_is_returned_as_is_without_cv2(use_settings, monkeypatch) -> None:
    overlay = BboxOverlay()
    original = _fire_jpeg()
    monkeypatch.setattr(bo, "cv2", _NoCv2())
    session_analysis_store.clear_session("calm")
    assert overlay.render("calm", original, 1) is original  # 결과 없음
    _store_result("calm", [])
    assert overlay.render("calm", original, 2) is original  # 감지 0건
    _store_result("calm", [_det("fire", 0.8)], NOW - timedelta(seconds=5))
    assert overlay.render("calm", original, 3) is original  # 유지 시간 밖
    _store_result("calm", [_det("fire", 0.2)])
    assert overlay.render("calm", original, 4) is original  # 기준 미달
    assert asyncio.run(overlay.render_async("calm", original, 5)) is original


def test_draw_flag_off_returns_original_bytes(use_settings, monkeypatch) -> None:
    use_settings(live_draw_bbox=False)
    overlay = BboxOverlay()
    original = _fire_jpeg()
    _store_result("off", [_det("fire", 0.9)])
    monkeypatch.setattr(bo, "cv2", _NoCv2())
    assert overlay.render("off", original, 1) is original
    assert asyncio.run(overlay.render_async("off", original, 2)) is original


def test_original_bytes_are_not_modified_by_drawing(use_settings) -> None:
    overlay = BboxOverlay()
    original = _fire_jpeg()
    before = bytes(original)
    _store_result("pure", [_det("fire", 0.9)])
    drawn = overlay.render("pure", original, 1)
    assert drawn != before and original == before


def test_same_frame_is_drawn_once_for_many_viewers(use_settings, monkeypatch) -> None:
    overlay = BboxOverlay()
    calls: list[int] = []
    real = bo.draw_boxes
    monkeypatch.setattr(bo, "draw_boxes", lambda *a: (calls.append(1), real(*a))[1])
    original = _fire_jpeg()
    _store_result("shared", [_det("fire", 0.9)])

    async def viewers() -> list[bytes]:
        return await asyncio.gather(*(overlay.render_async("shared", original, 7) for _ in range(5)))

    outs = asyncio.run(viewers())
    outs += [overlay.render("shared", original, 7) for _ in range(3)]
    assert len(calls) == 1 and len(set(outs)) == 1 and outs[0] != original
    overlay.render("shared", original, 8)  # 다음 프레임은 새로 그린다
    assert len(calls) == 2


def test_draw_over_time_limit_falls_back_to_original(use_settings, monkeypatch) -> None:
    use_settings(live_bbox_draw_timeout_ms=50)
    overlay = BboxOverlay()
    release = threading.Event()
    monkeypatch.setattr(bo, "draw_boxes", lambda frame, boxes: (release.wait(2), frame)[1])
    original = _fire_jpeg()
    _store_result("slow", [_det("fire", 0.9)])
    started = time.monotonic()
    assert overlay.render("slow", original, 1) is original
    assert asyncio.run(overlay.render_async("slow", original, 1)) is original
    assert time.monotonic() - started < 1.0  # 그리기가 끝나길 기다리지 않는다
    assert overlay._timeouts["slow"] == 2
    release.set()


def test_draw_exception_returns_original(use_settings, monkeypatch) -> None:
    overlay = BboxOverlay()

    def boom(frame, boxes):
        raise RuntimeError("secret-detail")

    monkeypatch.setattr(bo, "draw_boxes", boom)
    original = _fire_jpeg()
    _store_result("boom", [_det("fire", 0.9)])
    assert overlay.render("boom", original, 1) is original
    assert asyncio.run(overlay.render_async("boom", original, 1)) is original


# ---------------------------------------------------------------- 통합(라우터·원본 보존·클립 회귀)

def _make_app() -> FastAPI:
    app = FastAPI()

    @app.exception_handler(HTTPException)
    async def _handler(_: Request, exc: HTTPException) -> JSONResponse:
        if isinstance(exc.detail, dict) and "success" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(status_code=exc.status_code, content=error_response(str(exc.detail), "HTTP_ERROR", None))

    app.include_router(router_module.router, dependencies=[Depends(require_api_key)])
    app.dependency_overrides[get_settings] = lambda: _settings()
    return app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    from app.database.base import Base
    from app.database.session import engine

    Base.metadata.create_all(bind=engine)
    monkeypatch.setattr(bo, "bbox_overlay", BboxOverlay())
    monkeypatch.setattr(router_module, "bbox_overlay", bo.bbox_overlay)
    with TestClient(_make_app()) as c:
        yield c


def _start_session(client: TestClient, session_id: str) -> None:
    res = client.post(
        "/api/v1/stream-sessions", json={"sessionId": session_id, "cameraIdentifier": "cam-bbox"}, headers=HEADERS,
    )
    assert res.status_code == 200, res.text


def _fresh_result(session_id: str, detections: list[dict]) -> None:
    session_analysis_store.set_result(
        session_id,
        {"detectedType": "fire", "confidence": 0.9, "danger": True, "detections": detections,
         "analyzedAt": datetime.utcnow().isoformat()},
    )


def test_latest_frame_and_mjpeg_carry_blue_box_but_originals_stay_clean(client: TestClient, monkeypatch) -> None:
    # 실시간 시계로 되돌린다(_frozen_clock 은 순수 단위 테스트용).
    monkeypatch.undo()
    monkeypatch.setattr(bo, "bbox_overlay", BboxOverlay())
    monkeypatch.setattr(router_module, "bbox_overlay", bo.bbox_overlay)
    sid = "bbox_it"
    _start_session(client, sid)
    original = _fire_jpeg(320, 240)
    for _ in range(3):
        frame_store.set_frame(sid, original)  # 링버퍼에도 같은 원본이 들어간다
    _fresh_result(sid, [_det("fire", 0.8, (60, 50, 220, 190))])

    res = client.get(f"/api/v1/live-streams/{sid}/latest-frame", headers=HEADERS)
    assert res.status_code == 200 and res.headers["content-type"] == "image/jpeg"
    mask = _blue_mask(_decode(res.content))
    assert mask[120, 59:63].any() and mask[120, 219:223].any()  # 박스 좌표 위치에 파란 변
    assert not mask[100:140, 100:180].any()

    async def first_mjpeg_chunk() -> bytes:
        resp = await router_module.stream_mjpeg(sid, db=None)  # type: ignore[arg-type]
        try:
            return await resp.body_iterator.__anext__()
        finally:
            await resp.body_iterator.aclose()

    chunk = asyncio.run(first_mjpeg_chunk())
    jpeg = chunk.split(b"\r\n\r\n", 1)[1].rsplit(b"\r\n", 1)[0]
    assert _blue_mask(_decode(jpeg))[120, 59:63].any()

    # 원본(frame_store)·클립 링버퍼는 박스가 없다
    assert frame_store.get_frame(sid) == original
    buffered = clip_frame_buffer.snapshot(sid, datetime.utcnow() - timedelta(seconds=30), datetime.utcnow() + timedelta(seconds=5))
    assert buffered and all(b == original for _, b in buffered)

    client.post(f"/api/v1/stream-sessions/{sid}/stop", headers=HEADERS)


@pytest.mark.skipif(not ffmpeg_available(), reason="imageio-ffmpeg 미설치")
def test_clip_has_no_box_even_while_box_is_shown(client: TestClient, monkeypatch, tmp_path) -> None:
    monkeypatch.undo()
    monkeypatch.setattr(bo, "bbox_overlay", BboxOverlay())
    monkeypatch.setattr(router_module, "bbox_overlay", bo.bbox_overlay)
    sid = "bbox_clip"
    _start_session(client, sid)
    original = _fire_jpeg(320, 240)
    for _ in range(6):
        frame_store.set_frame(sid, original)
        time.sleep(0.05)
    _fresh_result(sid, [_det("fire", 0.8, (60, 50, 220, 190))])
    shown = client.get(f"/api/v1/live-streams/{sid}/latest-frame", headers=HEADERS)
    assert _blue_mask(_decode(shown.content)).any()  # 화면에는 박스가 있다

    clip = client.post(
        f"/api/v1/live-streams/{sid}/clips",
        json={"detectedAt": datetime.utcnow().isoformat() + "Z", "preSeconds": 3, "postSeconds": 0.1},
        headers=HEADERS,
    )
    assert clip.status_code == 200, clip.text
    path = tmp_path / "clip.webm"
    path.write_bytes(clip.content)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        pytest.skip("이 OpenCV 빌드는 WebM 을 열지 못한다")
    ok, frame = cap.read()
    cap.release()
    assert ok and not _blue_mask(frame).any()  # 클립 프레임에는 파란 박스가 없다

    client.post(f"/api/v1/stream-sessions/{sid}/stop", headers=HEADERS)
