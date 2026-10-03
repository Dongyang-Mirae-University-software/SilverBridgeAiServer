from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from tests.clip_helpers import make_jpeg
from app.schemas.clip_schema import ClipRequest
from app.services.clip_buffer import ClipFrameBuffer
from app.services.clip_service import _to_naive_utc, even_size, parse_jpeg_size, select_frames

T0 = datetime(2026, 10, 4, 1, 0, 0)


def _at(sec: float) -> datetime:
    return T0 + timedelta(seconds=sec)


def test_buffer_drops_frames_older_than_buffer_seconds() -> None:
    buf = ClipFrameBuffer(buffer_seconds=10, max_frames=1000)
    for i in range(30):  # 0~14.5초, 0.5초 간격
        buf.add("s1", _at(i * 0.5), b"f%d" % i)

    frames = buf.snapshot("s1", _at(-100), _at(100))
    assert frames[0][0] == _at(4.5)  # 14.5 - 10
    assert frames[-1][0] == _at(14.5)


def test_buffer_caps_frame_count() -> None:
    buf = ClipFrameBuffer(buffer_seconds=10, max_frames=5)
    for i in range(20):
        buf.add("s1", _at(i * 0.01), b"x")
    assert len(buf.snapshot("s1", _at(-1), _at(1))) == 5


def test_snapshot_selects_inclusive_window_and_shares_bytes() -> None:
    buf = ClipFrameBuffer(buffer_seconds=10, max_frames=100)
    payloads = [bytes([i]) * 10 for i in range(10)]
    for i, payload in enumerate(payloads):
        buf.add("s1", _at(i), payload)

    frames = buf.snapshot("s1", _at(2), _at(5))
    assert [t for t, _ in frames] == [_at(2), _at(3), _at(4), _at(5)]
    assert frames[0][1] is payloads[2]  # 복사가 아니라 참조


def test_snapshot_survives_later_eviction() -> None:
    buf = ClipFrameBuffer(buffer_seconds=2, max_frames=100)
    for i in range(3):
        buf.add("s1", _at(i), b"old")
    frames = buf.snapshot("s1", _at(0), _at(2))
    for i in range(3, 10):
        buf.add("s1", _at(i), b"new")
    assert len(frames) == 3  # 스냅샷은 링버퍼에서 밀려나도 남는다


def test_clear_and_latest_time() -> None:
    buf = ClipFrameBuffer(buffer_seconds=10, max_frames=100)
    assert buf.latest_time("s1") is None
    buf.add("s1", _at(1), b"a")
    buf.add("s1", _at(2), b"b")
    assert buf.latest_time("s1") == _at(2)
    buf.clear("s1")
    assert buf.latest_time("s1") is None
    assert buf.snapshot("s1", _at(0), _at(5)) == []


def test_sweep_removes_sessions_that_stopped_sending(monkeypatch: pytest.MonkeyPatch) -> None:
    buf = ClipFrameBuffer(buffer_seconds=10, max_frames=100)
    monkeypatch.setattr(ClipFrameBuffer, "_SWEEP_INTERVAL_SEC", 0.0)
    buf.add("gone", _at(0), b"a")
    buf.add("alive", _at(5), b"b")
    assert buf.session_count() == 2
    buf.add("alive", _at(11), b"c")  # gone 의 마지막 프레임이 11-10=1초보다 오래됨
    assert buf.session_count() == 1
    assert buf.latest_time("gone") is None


def test_parse_jpeg_size_reads_odd_dimensions() -> None:
    assert parse_jpeg_size(make_jpeg(641, 361)) == (641, 361)
    assert parse_jpeg_size(make_jpeg(1920, 1080)) == (1920, 1080)
    assert parse_jpeg_size(b"not a jpeg") is None
    assert parse_jpeg_size(b"\xff\xd8\xff\xd9") is None


def test_even_size() -> None:
    assert even_size(641, 361) == (640, 360)
    assert even_size(1920, 1080) == (1920, 1080)
    assert even_size(1, 1) == (2, 2)


def test_select_frames_fps_and_subsampling() -> None:
    frames = [bytes([i]) for i in range(25)]
    out, fps = select_frames(frames, 5.0)
    assert len(out) == 25 and fps == 5.0

    many = [bytes([i % 256]) for i in range(100)]
    out, fps = select_frames(many, 5.0)
    assert len(out) == 75 and fps == 15.0  # 길이 5초 유지
    assert out[0] == many[0]

    out, fps = select_frames(frames[:2], 5.0)
    assert len(out) == 2 and fps == 1.0  # 하한


def test_clip_request_defaults_and_time_parsing() -> None:
    req = ClipRequest.model_validate({})
    assert (req.detectedAt, req.preSeconds, req.postSeconds) == (None, 3, 2)

    z = ClipRequest.model_validate({"detectedAt": "2026-10-04T01:00:00.500Z"})
    assert _to_naive_utc(z.detectedAt) == datetime(2026, 10, 4, 1, 0, 0, 500000)

    kst = ClipRequest.model_validate({"detectedAt": "2026-10-04T10:00:00+09:00"})
    assert _to_naive_utc(kst.detectedAt) == datetime(2026, 10, 4, 1, 0, 0)

    naive = ClipRequest.model_validate({"detectedAt": "2026-10-04T01:00:00"})  # analyzedAt 원본 형태
    assert _to_naive_utc(naive.detectedAt) == datetime(2026, 10, 4, 1, 0, 0)

    before = datetime.now(timezone.utc).replace(tzinfo=None)
    assert _to_naive_utc(None) >= before


@pytest.mark.parametrize(
    "payload",
    [
        {"preSeconds": 5.1},
        {"preSeconds": -1},
        {"postSeconds": 3.5},
        {"preSeconds": 0, "postSeconds": 0},
        {"preSeconds": math.nan},
        {"detectedAt": "어제"},
    ],
)
def test_clip_request_rejects_invalid(payload: dict) -> None:
    with pytest.raises(ValidationError):
        ClipRequest.model_validate(payload)
