"""MedGemma 적재 실패가 기동·응급 턴을 막지 않는다(2026-11-10 2차 점검 R-1·I-1·R-3)."""
from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

import app.main as main
from app.models.medgemma_loader import MedGemmaLoader


class _Detector:
    def __init__(self, order: list[str], name: str) -> None:
        self.order, self.name = order, name

    def try_load(self) -> None:
        self.order.append(self.name)


class _Loader:
    def __init__(self, order: list[str], *, loaded: bool, warmup_error: bool = False) -> None:
        self.order, self.state, self.warmup_error = order, SimpleNamespace(loaded=loaded, load_error="x"), warmup_error

    def load(self) -> None:
        self.order.append("medgemma.load")

    def warmup(self) -> None:
        self.order.append("medgemma.warmup")
        if self.warmup_error:
            raise RuntimeError("secret path /models/x")


def _run_lifespan(monkeypatch: pytest.MonkeyPatch, loader: _Loader, order: list[str]) -> None:
    monkeypatch.setattr(main, "get_medgemma_loader", lambda: loader)
    for name in ("get_fire_smoke_detector", "get_knife_detector", "get_fall_detector"):
        monkeypatch.setattr(main, name, lambda n=name: _Detector(order, n))
    monkeypatch.setattr(main, "log_gpu_status", lambda: None, raising=False)
    monkeypatch.setattr(main, "_start_companion_warmup", lambda: None)
    monkeypatch.setattr(main, "_startup_housekeeping", lambda: None)
    monkeypatch.setattr(main, "get_medication_ocr_service", lambda: None)
    monkeypatch.setattr(main, "get_drug_name_list", lambda: None)
    monkeypatch.setattr(main.settings, "medgemma_enabled", True, raising=False)

    async def go() -> None:
        async with main.lifespan(main.app):
            order.append("serving")

    asyncio.run(go())


def test_startup_continues_when_medgemma_not_loaded(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    order: list[str] = []
    with caplog.at_level("ERROR"):
        _run_lifespan(monkeypatch, _Loader(order, loaded=False), order)
    assert "serving" in order and "medgemma.warmup" not in order
    assert order.index("get_fire_smoke_detector") < order.index("medgemma.load")  # 탐지기가 먼저 올라간다
    assert "[MEDGEMMA-STARTUP-FAILED]" in caplog.text


def test_startup_continues_when_warmup_raises_and_log_has_no_message(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    order: list[str] = []
    with caplog.at_level("ERROR"):
        _run_lifespan(monkeypatch, _Loader(order, loaded=True, warmup_error=True), order)
    assert "serving" in order
    assert "RuntimeError" in caplog.text and "secret path" not in caplog.text


def _loader() -> MedGemmaLoader:
    settings = SimpleNamespace(medgemma_enabled=True, medgemma_max_concurrent_generate=1)
    try:
        return MedGemmaLoader(settings)  # type: ignore[arg-type]
    except Exception:  # 설정 칸이 더 필요하면 실제 설정으로
        from app.core.config import get_settings

        return MedGemmaLoader(get_settings())


def test_ensure_loaded_does_not_wait_for_a_load_in_progress() -> None:
    loader = _loader()
    calls: list[int] = []
    loader.load = lambda: calls.append(1)  # type: ignore[method-assign]
    loader._lock.acquire()  # 다른 스레드가 적재 중
    try:
        started = time.monotonic()
        loader.ensure_loaded()
        assert time.monotonic() - started < 0.5 and calls == []
    finally:
        loader._lock.release()
    loader.ensure_loaded()
    assert calls == [1]  # 잠금이 풀리면 적재를 시도한다


def test_ensure_loaded_skips_retry_right_after_failure() -> None:
    loader = _loader()
    calls: list[int] = []
    loader.load = lambda: calls.append(1)  # type: ignore[method-assign]
    loader._load_failed_at = time.monotonic()
    loader.ensure_loaded()
    assert calls == []
    loader._load_failed_at = time.monotonic() - loader.LOAD_RETRY_SEC - 1
    loader.ensure_loaded()
    assert calls == [1]


def test_set_enabled_does_not_take_the_load_lock() -> None:
    loader = _loader()
    loader._lock.acquire()
    try:
        done = threading.Event()

        def flip() -> None:
            loader.set_enabled(False)
            done.set()

        threading.Thread(target=flip, daemon=True).start()
        assert done.wait(1.0) and loader.enabled is False
    finally:
        loader._lock.release()
    loader._load_failed_at = 1.0
    loader.set_enabled(True)
    assert loader._load_failed_at is None  # 다시 켜면 바로 적재를 시도한다


def test_generate_max_time_default_and_bad_value() -> None:
    loader = _loader()
    assert loader._generate_max_time() == 90.0 or loader._generate_max_time() > 0
    loader._settings = SimpleNamespace(medgemma_generate_max_time_sec=0)
    assert loader._generate_max_time() == 90.0
    loader._settings = SimpleNamespace(medgemma_generate_max_time_sec="abc")
    assert loader._generate_max_time() == 90.0


def test_startup_housekeeping_calls_purge_and_survives_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.services.training_sample_collector as collector

    calls: list[dict] = []
    monkeypatch.setattr(collector, "purge_expired", lambda *a, **k: calls.append(k))
    main._startup_housekeeping()
    assert calls == [{"start_loop": True}]

    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(collector, "purge_expired", boom)
    main._startup_housekeeping()  # 예외를 올리지 않는다
