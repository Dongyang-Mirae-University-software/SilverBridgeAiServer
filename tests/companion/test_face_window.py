"""얼굴 확인 모듈(companion/face_window.py) 테스트. AI 서버 없이 돌아간다.

실행 (저장소 루트에서):
    venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app" / "companion"))
import face_window as fw  # noqa: E402
import stream_client as sc  # noqa: E402


def blank_jpeg() -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((240, 320, 3), 120, np.uint8))
    assert ok
    return buf.tobytes()


class AlwaysFace:
    def detectMultiScale(self, *args, **kwargs):
        return [(10, 10, 80, 80)]


class NeverFace:
    def detectMultiScale(self, *args, **kwargs):
        return ()


def fake_clock():
    now = [0.0]
    return (lambda: now[0]), (lambda s: now.__setitem__(0, now[0] + s))


def set_stream_env(monkeypatch):
    monkeypatch.setenv("COMPANION_STREAM_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("COMPANION_STREAM_API_KEY", "test-key")


def test_summarize_seen_when_most_frames_have_face():
    out = fw.summarize([True, True, True, False, False])
    assert out == {"faceStatus": fw.STATUS_SEEN, "frames": 5, "withFace": 3}


def test_summarize_unknown_when_face_rarely_seen():
    out = fw.summarize([True, False, False, False, False])
    assert out["faceStatus"].startswith(fw.STATUS_UNKNOWN)
    assert out["withFace"] == 1


def test_summarize_unknown_when_no_frames():
    out = fw.summarize([])
    assert out["faceStatus"] == f"{fw.STATUS_UNKNOWN} (프레임 없음)"
    assert out["frames"] == 0


def test_has_face_uses_detector_result():
    jpeg = blank_jpeg()
    assert fw.has_face(jpeg, AlwaysFace()) is True
    assert fw.has_face(jpeg, NeverFace()) is False


def test_has_face_treats_undecodable_bytes_as_no_face():
    assert fw.has_face(b"not a jpeg", AlwaysFace()) is False


def test_has_face_with_real_detector_on_blank_image():
    assert fw.has_face(blank_jpeg()) is False


def test_collect_counts_frames_and_keeps_going_on_missing_frames(monkeypatch, tmp_path):
    set_stream_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    clock, sleep = fake_clock()
    jpeg = blank_jpeg()
    responses = [jpeg, jpeg, jpeg]

    def fetch(session_id):
        if responses:
            return responses.pop(0)
        raise sc.StreamError("STREAM_FRAME_NOT_FOUND", "최신 프레임이 없습니다.")

    out = fw.collect("sess-1", seconds=5, every_sec=1, fetch=fetch, detector=AlwaysFace(), clock=clock, sleep=sleep)
    assert out["frames"] == 5
    assert out["withFace"] == 3
    assert out["faceStatus"] == fw.STATUS_SEEN
    # 프레임은 파일로 남지 않는다 (작업 폴더에 아무것도 생기지 않음)
    assert list(tmp_path.rglob("*")) == []


def test_collect_reports_unreachable_server(monkeypatch):
    set_stream_env(monkeypatch)
    clock, sleep = fake_clock()

    def fetch(session_id):
        raise sc.StreamError("UNREACHABLE", "연결 실패")

    out = fw.collect("sess-1", seconds=5, every_sec=1, fetch=fetch, detector=AlwaysFace(), clock=clock, sleep=sleep)
    assert out["faceStatus"] == fw.STATUS_UNREACHABLE


def test_collect_without_stream_settings_says_not_checked(monkeypatch):
    monkeypatch.setenv("COMPANION_STREAM_BASE_URL", "")
    monkeypatch.setenv("COMPANION_STREAM_API_KEY", "")
    out = fw.collect("sess-1", seconds=5)
    assert out == {"faceStatus": fw.STATUS_NO_STREAM, "frames": 0, "withFace": 0}


@pytest.mark.parametrize("seconds, expected_frames", [(0.5, 1), (10, 10)])
def test_collect_frame_count_follows_seconds(monkeypatch, seconds, expected_frames):
    set_stream_env(monkeypatch)
    clock, sleep = fake_clock()
    out = fw.collect("sess-1", seconds=seconds, every_sec=1, fetch=lambda s: blank_jpeg(),
                     detector=NeverFace(), clock=clock, sleep=sleep)
    assert out["frames"] == expected_frames
    assert out["withFace"] == 0
    assert out["faceStatus"].startswith(fw.STATUS_UNKNOWN)
