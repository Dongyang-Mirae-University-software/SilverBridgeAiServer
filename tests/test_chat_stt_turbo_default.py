"""2026-11-11: 챗봇 받아쓰기(/companion/stt) 기본 엔진 turbo 전환 · SenseVoice 이모지 섞임 제거 · turbo 미리 올리기.

모델은 가짜로 바꿔 끼운다 (실제 모델·API 키를 쓰지 않는다).
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from app.core.config import get_settings
from app.services.companion import stt_engine
from app.services.companion import sensevoice_service as sv
from app.services.companion.sensevoice_service import sensevoice_service, strip_marks
from app.services.companion.turbo_service import turbo_service


@pytest.fixture
def settings(monkeypatch):
    s = get_settings()

    def set_(**kw):
        for k, v in kw.items():
            monkeypatch.setattr(s, k, v)
    return set_


# ---------- 엔진 선택 ----------

def test_chat_stt_default_engine_is_turbo():
    from app.core.config import Settings
    assert Settings.model_fields["companion_stt_api_engine"].default == "turbo"
    assert Settings.model_fields["companion_stt_warmup"].default is True


def test_chat_stt_turbo_is_not_sensevoice(settings, monkeypatch):
    settings(companion_stt_api_engine="turbo")
    picked = []
    monkeypatch.setattr(stt_engine, "_engine", lambda n: picked.append(n) or turbo_service)
    run = stt_engine.get_stt_api_transcriber()
    assert picked == ["turbo"] and run != sensevoice_service.transcribe_pcm


def test_chat_stt_sensevoice_rollback_still_works(settings):
    settings(companion_stt_api_engine="sensevoice")
    assert stt_engine.get_stt_api_transcriber() == sensevoice_service.transcribe_pcm


# ---------- SenseVoice 이모지·표시 제거 ----------

@pytest.mark.parametrize("raw, clean", [
    ("괜찮아요.😔", "괜찮아요."),            # <|SAD|>
    ("😀좋아요😊", "좋아요"),                # <|Laughter|> + <|HAPPY|>
    ("🎼테스트 테스트", "테스트 테스트"),     # <|BGM|>
    ("👏네 알겠어요😡", "네 알겠어요"),
    ("❓약 먹었어요", "약 먹었어요"),
    ("<|ko|><|NEUTRAL|>밥 먹었어", "밥 먹었어"),  # 표시가 남은 경우
    ("내일 병원 3시, OK?", "내일 병원 3시, OK?"),  # 한글·영문·숫자·문장부호는 그대로
    ("😔", ""),
])
def test_strip_marks(raw, clean):
    assert strip_marks(raw) == clean


class _FakeModel:
    def __init__(self, raw: str) -> None:
        self.raw = raw

    def generate(self, **_kw):
        return [{"text": self.raw}]


@pytest.mark.parametrize("raw, clean", [
    ("<|ko|><|SAD|><|Speech|><|withitn|>괜찮아요.", "괜찮아요."),
    ("<|ko|><|HAPPY|><|Laughter|><|withitn|>좋아요", "좋아요"),
    ("<|ko|><|NEUTRAL|><|BGM|><|withitn|>테스트", "테스트"),
    ("<|ko|><|SAD|><|Speech|><|withitn|>그.", ""),  # 이모지를 떼면 군말만 남는다 → 대답 없음
])
def test_sensevoice_transcribe_has_no_emoji(monkeypatch, raw, clean):
    """실제 funasr 후처리를 거쳐도 이모지·표시가 글자에 섞이지 않는다."""
    pytest.importorskip("funasr")
    monkeypatch.setattr(sensevoice_service, "_load", lambda: _FakeModel(raw))
    pcm = (np.random.default_rng(0).standard_normal(16000) * 0.1).astype(np.float32)
    assert sensevoice_service.transcribe_pcm(pcm).text == clean


def test_filler_check_runs_after_strip():
    assert "그" in sv.FILLER_ONLY  # 위 마지막 사례의 전제


# ---------- turbo 미리 올리기 ----------

def _warm(monkeypatch, llm=False, **kw):
    """llm=False 면 EXAONE 은 건너뛰고 turbo 만 본다."""
    import app.main as main
    from app.services.companion import llm_client as lc
    called = threading.Event()
    monkeypatch.setattr(turbo_service, "warm_up", called.set)
    monkeypatch.setattr(lc.llm_client, "warm_up", lambda: None)
    base = {"companion_api_key": "k", "companion_stt_warmup": True, "companion_llm_warmup": llm,
            "companion_stt_engine": "turbo", "companion_stt_api_engine": "turbo"}
    base.update(kw)
    for k, v in base.items():
        monkeypatch.setattr(main.settings, k, v)
    main._start_companion_warmup()
    return called


def test_stt_warmup_runs_when_chat_engine_is_turbo(monkeypatch):
    assert _warm(monkeypatch, companion_stt_engine="sensevoice").wait(2)


def test_stt_warmup_runs_when_companion_engine_is_turbo(monkeypatch):
    assert _warm(monkeypatch, companion_stt_api_engine="sensevoice").wait(2)


@pytest.mark.parametrize("kw", [
    {"companion_api_key": ""},
    {"companion_stt_warmup": False},
    {"companion_stt_engine": "sensevoice", "companion_stt_api_engine": "sensevoice"},
])
def test_stt_warmup_skipped(monkeypatch, kw):
    assert not _warm(monkeypatch, **kw).wait(0.3)


def test_turbo_warms_after_llm_in_one_thread(monkeypatch):
    """11/11 gosky: 두 모델을 동시에 올리면 turbo 가 실패했다 - EXAONE 이 끝난 뒤 같은 스레드에서 올린다."""
    import app.main as main
    from app.services.companion import llm_client as lc
    order: list[tuple[str, str]] = []
    done = threading.Event()
    llm_release = threading.Event()

    def llm_warm():
        order.append(("llm-start", threading.current_thread().name))
        llm_release.wait(2)
        order.append(("llm-end", threading.current_thread().name))

    def stt_warm():
        order.append(("stt", threading.current_thread().name))
        done.set()

    monkeypatch.setattr(lc.llm_client, "warm_up", llm_warm)
    monkeypatch.setattr(turbo_service, "warm_up", stt_warm)
    for k, v in {"companion_api_key": "k", "companion_llm_warmup": True, "companion_stt_warmup": True,
                 "companion_stt_engine": "turbo", "companion_stt_api_engine": "turbo"}.items():
        monkeypatch.setattr(main.settings, k, v)
    main._start_companion_warmup()
    assert not done.wait(0.3)  # EXAONE 이 끝나기 전에는 turbo 를 올리지 않는다
    llm_release.set()
    assert done.wait(2)
    assert [o[0] for o in order] == ["llm-start", "llm-end", "stt"]
    assert len({o[1] for o in order}) == 1


def test_stt_warmup_runs_even_if_llm_warmup_fails(monkeypatch):
    import app.main as main
    from app.services.companion import llm_client as lc
    called = threading.Event()

    def boom():
        raise RuntimeError("x")

    monkeypatch.setattr(lc.llm_client, "warm_up", boom)
    monkeypatch.setattr(turbo_service, "warm_up", called.set)
    for k, v in {"companion_api_key": "k", "companion_llm_warmup": True, "companion_stt_warmup": True,
                 "companion_stt_api_engine": "turbo"}.items():
        monkeypatch.setattr(main.settings, k, v)
    main._start_companion_warmup()
    assert called.wait(2)


def test_stt_warmup_failure_is_swallowed(monkeypatch, caplog):
    import app.main as main
    done = threading.Event()

    def boom():
        done.set()
        raise RuntimeError("secret path /models/x")

    monkeypatch.setattr(turbo_service, "warm_up", boom)
    for k, v in {"companion_api_key": "k", "companion_stt_warmup": True, "companion_llm_warmup": False,
                 "companion_stt_api_engine": "turbo"}.items():
        monkeypatch.setattr(main.settings, k, v)
    with caplog.at_level("WARNING"):
        main._start_companion_warmup()
        assert done.wait(2)
        for _ in range(50):
            if "turbo warm-up failed" in caplog.text:
                break
            threading.Event().wait(0.02)
    assert "turbo warm-up failed (RuntimeError)" in caplog.text and "secret" not in caplog.text
