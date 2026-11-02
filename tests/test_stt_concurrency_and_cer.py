from __future__ import annotations

import threading
from pathlib import Path

import pytest

from app.services.companion import stt_service as stt_mod
from app.services.companion.stt_service import SttService
from scripts.companion.eval_stt import cer, edit_distance


def test_cer_basic():
    assert edit_distance("abc", "abd") == 1
    assert cer("잘 잤어요", "잘잤어요") == 0.0  # 공백은 비교에서 뺀다
    assert cer("안녕하세요", "안녕하셔요") == pytest.approx(1 / 5)


def test_concurrent_requests_load_model_once(monkeypatch, tmp_path):
    # 모델 폴더가 있는 것처럼 꾸미고, 모델 생성 호출 횟수를 센다 (S3-8)
    model_dir = tmp_path / "companion" / "stt" / "large-v3"
    model_dir.mkdir(parents=True)
    monkeypatch.setattr(stt_mod.get_settings(), "model_base_path", str(tmp_path), raising=False)

    calls = []

    class FakeWhisper:
        def __init__(self, *a, **k):
            calls.append(1)

    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisper)
    svc = SttService()
    monkeypatch.setattr(svc, "model_dir", lambda: model_dir)

    barrier = threading.Barrier(10)
    results = []

    def worker():
        barrier.wait()
        results.append(svc._load())

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1
    assert len(results) == 10
    assert all(r is results[0] for r in results)
