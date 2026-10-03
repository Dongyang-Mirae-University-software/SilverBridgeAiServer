from __future__ import annotations

import re
import subprocess
from pathlib import Path

import cv2
import pytest

from tests.clip_helpers import WEBM_MAGIC, make_jpeg
from app.services.clip_service import ClipEncodeError, encode_webm, even_size, ffmpeg_available

pytestmark = pytest.mark.skipif(not ffmpeg_available(), reason="imageio-ffmpeg 미설치")


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _duration_sec(path: Path) -> float:
    proc = subprocess.run([_ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", proc.stderr)
    assert match, proc.stderr
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


@pytest.mark.parametrize("size", [(640, 360), (641, 361)])
def test_encode_webm_produces_playable_5s_clip(tmp_path: Path, size: tuple[int, int]) -> None:
    frames = [make_jpeg(size[0], size[1], i) for i in range(25)]
    width, height = even_size(*size)

    data = encode_webm(frames, 5.0, width, height, "3M", 20.0, str(tmp_path))

    assert data[:4] == WEBM_MAGIC
    assert list(tmp_path.iterdir()) == []  # 임시 파일 삭제됨

    out = tmp_path.parent / f"check_{size[0]}.webm"
    out.write_bytes(data)
    assert abs(_duration_sec(out) - 5.0) < 0.25  # duration 정보가 들어 있다

    cap = cv2.VideoCapture(str(out))
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape[1] == width and frame.shape[0] == height
        count += 1
    cap.release()
    assert count == 25


def test_encode_webm_handles_resolution_change_mid_clip(tmp_path: Path) -> None:
    frames = [make_jpeg(640, 360, i) for i in range(5)] + [make_jpeg(360, 640, i) for i in range(5)]
    data = encode_webm(frames, 5.0, 640, 360, "3M", 20.0, str(tmp_path))
    assert data[:4] == WEBM_MAGIC


def test_encode_webm_failure_cleans_up(tmp_path: Path) -> None:
    with pytest.raises(ClipEncodeError):
        encode_webm([b"garbage"] * 5, 5.0, 640, 360, "3M", 20.0, str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_encode_webm_timeout_kills_process_and_cleans_up(tmp_path: Path) -> None:
    frames = [make_jpeg(1920, 1080, i) for i in range(40)]
    with pytest.raises(ClipEncodeError) as exc_info:
        encode_webm(frames, 15.0, 1920, 1080, "3M", 0.05, str(tmp_path))
    assert exc_info.value.reason == "timeout"
    assert list(tmp_path.iterdir()) == []
    leftover = subprocess.run(["pgrep", "-f", str(tmp_path)], capture_output=True, text=True)
    assert leftover.stdout.strip() == ""
