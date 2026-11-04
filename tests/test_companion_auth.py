from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from scripts.companion.dev_app import app


def _client_with_settings(**overrides):
    # Settings 의 os.getenv 값은 클래스 정의 시점에 굳으므로 환경 변수 대신 의존성을 재정의한다.
    settings = get_settings().model_copy(update=overrides)
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


@pytest.fixture
def client_with_key():
    client = _client_with_settings(companion_api_key="test-companion-key")
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def client_without_key():
    client = _client_with_settings(companion_api_key="")
    yield client
    app.dependency_overrides.clear()


def test_no_key_is_rejected(client_with_key):
    r = client_with_key.get("/companion/status")
    assert r.status_code == 401
    body = r.json()
    assert body["success"] is False
    assert body["code"] == "AUTH_INVALID_KEY"  # 명세 오류 코드
    assert body["errorCode"] == "AUTH_INVALID_KEY"  # S2 객관식 답: code 와 errorCode 둘 다


def test_wrong_key_is_rejected(client_with_key):
    r = client_with_key.get("/companion/status", headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_correct_key_is_accepted(client_with_key):
    r = client_with_key.get("/companion/status", headers={"X-API-Key": "test-companion-key"})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["data"]["enabled"] is True
    assert body["data"]["maxTurns"] == 8
    assert body["data"]["sessionLimitSec"] == 0  # D1: 시간 제한 없음 (기본값)
    assert "test-companion-key" not in r.text  # 키 값은 응답에 없어야 한다


def test_empty_server_key_rejects_everything(client_without_key):
    # 설정 키가 비어 있으면 헤더가 있어도 거부 (실패 시 닫힘)
    r = client_without_key.get("/companion/status", headers={"X-API-Key": ""})
    assert r.status_code == 401
    r = client_without_key.get("/companion/status", headers={"X-API-Key": "anything"})
    assert r.status_code == 401
