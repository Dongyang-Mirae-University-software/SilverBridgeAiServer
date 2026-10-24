"""라우터·예약 추출 Claude 모델 분리(2026-10-24 O-2).

- 설정 우선순위: 용도별(CHAT_ROUTER_MODEL·RESERVATION_EXTRACT_MODEL) > CLAUDE_MODEL_NAME > claude-opus-5-5
- 인자 없이 만든 클라이언트 = 예약 추출 모델, 라우터 = 라우터 모델
- 오류 로그·예외 문구에 API 오류 본문을 싣지 않는다
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from app.core import config
from app.core.config import get_settings
from app.services import chat_router_service
from app.services.claude_structured_client import ClaudeStructuredClient


def _settings(**overrides):
    return get_settings().model_copy(update={"anthropic_api_key": "sk-test", **overrides})


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"CHAT_ROUTER_MODEL": "claude-haiku-5-5", "CLAUDE_MODEL_NAME": "claude-sonnet-5-5"}, "claude-haiku-5-5"),
        ({"CHAT_ROUTER_MODEL": "  ", "CLAUDE_MODEL_NAME": "claude-sonnet-5-5"}, "claude-sonnet-5-5"),
        ({"CLAUDE_MODEL_NAME": ""}, "claude-opus-5-5"),
        ({}, "claude-opus-5-5"),
    ],
)
def test_model_setting_priority(monkeypatch, env, expected) -> None:
    for name in ("CHAT_ROUTER_MODEL", "CLAUDE_MODEL_NAME"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert config._first_env(("CHAT_ROUTER_MODEL", "CLAUDE_MODEL_NAME"), "claude-opus-5-5") == expected


def test_client_without_model_uses_reservation_extract_model() -> None:
    client = ClaudeStructuredClient(_settings(reservation_extract_model="claude-sonnet-5-5", chat_router_model="claude-haiku-5-5"))
    assert client.state.model_name == "claude-sonnet-5-5"


def test_blank_model_falls_back_to_default() -> None:
    assert ClaudeStructuredClient(_settings(), "  ").state.model_name == "claude-opus-5-5"


def test_router_uses_router_model(monkeypatch) -> None:
    monkeypatch.setattr(
        chat_router_service,
        "get_settings",
        lambda: _settings(chat_router_model="claude-haiku-5-5", reservation_extract_model="claude-sonnet-5-5"),
    )
    chat_router_service.get_chat_router_service.cache_clear()
    try:
        assert chat_router_service.get_chat_router_service()._loader.state.model_name == "claude-haiku-5-5"
    finally:
        chat_router_service.get_chat_router_service.cache_clear()


class _Recorder:
    def __init__(self, exc: Exception | None = None) -> None:
        self.kwargs: dict | None = None
        self._exc = exc

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self._exc:
            raise self._exc
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text='{"route":"general"}')])


def _client_with(model: str, recorder: _Recorder) -> ClaudeStructuredClient:
    client = ClaudeStructuredClient(_settings(), model)
    client._client = SimpleNamespace(beta=SimpleNamespace(messages=recorder))
    return client


def test_status_error_does_not_leak_body(caplog) -> None:
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    secret = "사용자 문장 원문 SECRET"
    exc = anthropic.BadRequestError(secret, response=httpx.Response(400, request=req), body={"error": {"message": secret}})
    client = _client_with("claude-sonnet-5-5", _Recorder(exc))
    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError) as err:
        client.generate_structured_text("sys", "user")
    assert secret not in str(err.value)
    assert secret not in caplog.text
    assert "status=400" in caplog.text and "claude-sonnet-5-5" in caplog.text
