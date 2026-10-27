"""MedGemma 동시 생성 1개 제한(2026-10-27).

상담이 겹쳐 생성 2건이 같은 GPU에서 돌면 감지(화재·흉기·낙상) 처리 시간이 2배가 됐다(10/27 측정).
생성 슬롯으로 한 줄로 세우고, 대기 상한을 넘기면 생성하지 않고 고정 문구로 답한다. 응급 턴은 상한이 짧다.
모델은 가짜다 - 실제 생성(_generate_structured_text)만 바꿔 끼운다.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import Settings
from app.models.chat_log import ChatLog
from app.models.medgemma_loader import GenerationSlot, MedGemmaBusyError, MedGemmaLoader
from app.schemas.chat_schema import ChatContext, ChatRequest
from app.services import chat_service as cs
from app.services.medical_llm_service import MedicalLlmService

JSON_REPLY = '{"empathy": "걱정되셨겠어요.", "symptoms": "기침", "features": "", "advice": "", "reply": ""}'


def _loader(settings: Settings, fake_generate) -> MedGemmaLoader:
    loader = MedGemmaLoader(settings)
    loader._state.loaded = True
    loader._model = object()
    loader._torch = object()
    loader._generate_structured_text = fake_generate  # type: ignore[method-assign]
    return loader


class _Tracker:
    """동시에 생성 중인 수의 최댓값을 센다."""

    def __init__(self, hold_sec: float = 0.2) -> None:
        self.hold_sec = hold_sec
        self.active = 0
        self.peak = 0
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, *_a: Any) -> str:
        with self._lock:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)
        time.sleep(self.hold_sec)
        with self._lock:
            self.active -= 1
        return JSON_REPLY


def _run_parallel(loader: MedGemmaLoader, count: int, timeout: float = 10) -> list[BaseException | None]:
    errors: list[BaseException | None] = [None] * count

    def work(i: int) -> None:
        try:
            loader.generate_structured_text("s", "u", wait_timeout_sec=timeout)
        except BaseException as exc:  # noqa: BLE001
            errors[i] = exc

    threads = [threading.Thread(target=work, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(15)
    return errors


def _hold_slot(loader: MedGemmaLoader) -> tuple[threading.Event, threading.Thread]:
    """앞 상담이 생성 중인 상태를 만든다. release.set() 하면 풀린다."""
    held, release = threading.Event(), threading.Event()

    def holder() -> None:
        with loader._gen_slot.acquire(10):
            held.set()
            release.wait(10)

    thread = threading.Thread(target=holder)
    thread.start()
    assert held.wait(5)
    return release, thread


# ---- 슬롯 -----------------------------------------------------------------------------------------------


def test_default_limit_is_one() -> None:
    assert Settings().medgemma_max_concurrent_generate == 1
    assert Settings().medgemma_gen_wait_timeout_sec == 60
    assert Settings().medgemma_gen_wait_timeout_emergency_sec == 3


def test_concurrent_generation_runs_one_at_a_time() -> None:
    tracker = _Tracker()
    loader = _loader(Settings(medgemma_max_concurrent_generate=1), tracker)
    errors = _run_parallel(loader, 3)
    assert errors == [None, None, None]
    assert tracker.calls == 3 and tracker.peak == 1


def test_limit_zero_disables_slot() -> None:
    tracker = _Tracker()
    loader = _loader(Settings(medgemma_max_concurrent_generate=0), tracker)
    errors = _run_parallel(loader, 3)
    assert errors == [None, None, None]
    assert tracker.peak == 3


def test_slot_released_after_generation_error() -> None:
    def boom(*_a: Any) -> str:
        raise RuntimeError("empty generation")

    loader = _loader(Settings(), boom)
    with pytest.raises(RuntimeError):
        loader.generate_structured_text("s", "u", wait_timeout_sec=1)
    loader._generate_structured_text = _Tracker(hold_sec=0)  # type: ignore[method-assign]
    started = time.perf_counter()
    assert loader.generate_structured_text("s", "u", wait_timeout_sec=1) == JSON_REPLY
    assert time.perf_counter() - started < 0.5  # 예외 뒤 슬롯이 풀려 기다리지 않는다


def test_wait_timeout_raises_busy_without_generating(caplog) -> None:
    tracker = _Tracker(hold_sec=0)
    loader = _loader(Settings(), tracker)
    release, thread = _hold_slot(loader)
    try:
        with caplog.at_level(logging.INFO, logger="app.models.medgemma_loader"):
            started = time.perf_counter()
            with pytest.raises(MedGemmaBusyError):
                loader.generate_structured_text("s", "민감한 상담 문장", wait_timeout_sec=0.3)
            elapsed = time.perf_counter() - started
    finally:
        release.set()
        thread.join(5)
    assert 0.25 <= elapsed < 1.5
    assert tracker.calls == 0  # 상한을 넘기면 생성하지 않는다
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "[MEDGEMMA-GEN-BUSY] waitMs=" in joined and "waiting=1" in joined
    assert "민감한" not in joined
    # 상한 초과 뒤에도 대기 수가 원래대로 돌아오고 슬롯을 다시 잡을 수 있다
    assert loader._gen_slot._waiting == 0
    assert loader.generate_structured_text("s", "u", wait_timeout_sec=1) == JSON_REPLY


def test_waiting_generation_logs_wait(caplog) -> None:
    loader = _loader(Settings(), _Tracker(hold_sec=0))
    release, thread = _hold_slot(loader)
    threading.Timer(0.2, release.set).start()
    with caplog.at_level(logging.INFO, logger="app.models.medgemma_loader"):
        assert loader.generate_structured_text("s", "u", wait_timeout_sec=5) == JSON_REPLY
    thread.join(5)
    assert any("[MEDGEMMA-GEN-WAIT] waitMs=" in r.getMessage() for r in caplog.records)


def test_no_wait_log_when_slot_free(caplog) -> None:
    loader = _loader(Settings(), _Tracker(hold_sec=0))
    with caplog.at_level(logging.INFO, logger="app.models.medgemma_loader"):
        loader.generate_structured_text("s", "u", wait_timeout_sec=1)
    assert not any("MEDGEMMA-GEN-" in r.getMessage() for r in caplog.records)


def test_generation_slot_unbounded_when_limit_not_positive() -> None:
    slot = GenerationSlot(-1)
    with slot.acquire(0.01), slot.acquire(0.01):
        pass


# ---- 응급 턴은 짧게 기다린다 -----------------------------------------------------------------------------


class _SpyLoader:
    def __init__(self) -> None:
        self.state = SimpleNamespace(model_name="fake")
        self.timeouts: list[float | None] = []

    def ensure_loaded(self) -> None:
        return None

    def generate_structured_text(self, system_prompt: str, user_prompt: str, prefill: str = "", *, wait_timeout_sec=None) -> str:  # noqa: ARG002
        self.timeouts.append(wait_timeout_sec)
        return JSON_REPLY


@pytest.mark.parametrize(
    ("message", "intent", "expected"),
    [
        ("어제부터 기침이 나요", "medical_advice", 60),
        ("가슴이 너무 아파서 숨을 못 쉬겠어요", "medical_advice", 3),  # 키워드 응급
        ("엄마가 좀 이상해요", "emergency_guidance", 3),  # 라우터 응급
    ],
)
def test_emergency_turn_uses_short_wait(message: str, intent: str, expected: int) -> None:
    spy = _SpyLoader()
    MedicalLlmService(Settings(), spy).generate(message, {}, [], intent=intent)  # type: ignore[arg-type]
    assert spy.timeouts == [expected]


class _Router:
    def __init__(self, emergency: bool) -> None:
        self.emergency = emergency

    def route(self, message: str, history, ctx, prev_reservation=None) -> dict[str, Any]:  # noqa: ARG002
        return {"route": "symptom", "normalized_message": message, "symptom_summary": message,
                "emergency": self.emergency, "reason": "t", "engine": "fake"}


@pytest.fixture()
def chat_env(monkeypatch):
    engine = create_engine(f"sqlite:///{tempfile.mkdtemp(prefix='sb_gen_slot_')}/t.db")
    ChatLog.__table__.create(bind=engine)
    db = sessionmaker(bind=engine)()
    tracker = _Tracker(hold_sec=0)
    loader = _loader(Settings(), tracker)
    holder = SimpleNamespace(router=_Router(emergency=False), loader=loader, tracker=tracker)
    monkeypatch.setattr(cs, "get_chat_router_service", lambda: holder.router)
    monkeypatch.setattr(cs, "get_reservation_credential_service", lambda: object())
    monkeypatch.setattr(cs, "get_reservation_api_client", lambda: object())
    monkeypatch.setattr(cs, "get_medical_llm_service", lambda: MedicalLlmService(Settings(), loader))
    svc = cs.ChatService()
    svc._settings = SimpleNamespace(has_reservation_account_secret=True, claude_model_name="fake")
    holder.svc, holder.db = svc, db
    yield holder
    db.close()


def _send(env, message: str) -> dict[str, Any]:
    payload = ChatRequest(userId="user-1", message=message, sessionId="s1", context=ChatContext(name="홍길동"), history=[])
    return env.svc.process_message(env.db, payload)


@pytest.mark.parametrize(
    ("message", "router_emergency"),
    [("가슴이 너무 아파서 숨을 못 쉬겠어요", False), ("엄마가 아까부터 좀 이상해요 대답을 잘 못하세요", True)],
)
def test_emergency_turn_gets_119_fallback_within_3s_while_slot_held(chat_env, caplog, message: str, router_emergency: bool) -> None:
    chat_env.router = _Router(emergency=router_emergency)
    release, thread = _hold_slot(chat_env.loader)
    try:
        with caplog.at_level(logging.INFO):
            started = time.perf_counter()
            out = _send(chat_env, message)
            elapsed = time.perf_counter() - started
    finally:
        release.set()
        thread.join(5)
    assert elapsed < 3.5  # 상한 3초 + 여유
    assert chat_env.tracker.calls == 0
    assert out["engine"] == "fallback"
    assert out["riskLevel"] == "high"
    assert out["emergencyWarning"] == [MedicalLlmService.EMERGENCY_WARNING]
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "[CHAT-LLM-FAILED] MedGemmaBusyError" in joined
    assert "숨을" not in joined and "엄마가" not in joined


def test_normal_turn_waits_for_slot_and_gets_model_reply(chat_env) -> None:
    release, thread = _hold_slot(chat_env.loader)
    threading.Timer(0.5, release.set).start()
    out = _send(chat_env, "어제부터 기침이 나요")
    thread.join(5)
    assert out["engine"] == "medgemma"
    assert chat_env.tracker.calls == 1
