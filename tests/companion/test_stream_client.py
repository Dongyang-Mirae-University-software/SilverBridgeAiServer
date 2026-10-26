"""말벗 프레임 조회 모듈(companion/stream_client.py) 테스트.

AI 서버 없이 돌아간다. 테스트 안에서 가짜 AI 서버를 띄우고, 키도 실행할 때마다 임시로 만든다.
.env 값은 읽지 않는다(환경 변수로 직접 지정).

실행 (저장소 루트에서):
    venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import json
import os
import secrets
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app" / "companion"))
import stream_client as sc  # noqa: E402

JPEG = b"\xff\xd8\xff\xd9"


class FakeAiServer(BaseHTTPRequestHandler):
    """AI 서버의 말벗 관련 경로만 흉내 낸다 (live_stream_router.py 와 같은 응답 모양)."""

    main_key = ""
    companion_key = ""
    seen: list[tuple[str, str | None]] = []

    def log_message(self, *args):  # 테스트 출력 조용히
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, error_code, message):
        body = json.dumps({"detail": {"success": False, "message": message, "errorCode": error_code, "data": None}})
        self._send(code, "application/json", body.encode())

    def do_GET(self):
        key = self.headers.get("X-API-Key")
        FakeAiServer.seen.append((self.path, key))
        if "apiKey=" in self.path:  # 쿼리스트링 키는 받지 않는다
            return self._error(401, "AUTH_INVALID_KEY", "유효하지 않은 API Key 입니다.")
        if key not in (FakeAiServer.main_key, FakeAiServer.companion_key):
            return self._error(401, "AUTH_INVALID_KEY", "유효하지 않은 API Key 입니다.")
        if self.path.endswith("/latest-frame"):
            if "empty" in self.path:
                return self._error(404, "STREAM_FRAME_NOT_FOUND", "최신 프레임이 없습니다.")
            return self._send(200, "image/jpeg", JPEG)
        if self.path.endswith("/status"):
            body = {"success": True, "message": "세션 상태 조회 완료", "data": {"status": "LIVE", "fps": 5.0}, "code": "OK"}
            return self._send(200, "application/json", json.dumps(body).encode())
        self._error(404, "NOT_FOUND", "없는 경로")


@pytest.fixture
def server(monkeypatch):
    main_key = secrets.token_urlsafe(16)
    companion_key = secrets.token_urlsafe(16)
    FakeAiServer.main_key = main_key
    FakeAiServer.companion_key = companion_key
    FakeAiServer.seen = []
    httpd = HTTPServer(("127.0.0.1", 0), FakeAiServer)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_port}"
    monkeypatch.setenv("COMPANION_STREAM_BASE_URL", base)
    monkeypatch.setenv("COMPANION_STREAM_API_KEY", companion_key)
    yield base
    httpd.shutdown()


def test_fetches_latest_frame_with_companion_key(server):
    assert sc.fetch_latest_frame("sess-1") == JPEG
    assert FakeAiServer.seen[-1][1] == os.environ["COMPANION_STREAM_API_KEY"]


def test_fetches_status(server):
    status = sc.fetch_status("sess-1")
    assert status["status"] == "LIVE"


def test_wrong_key_is_rejected(server, monkeypatch):
    monkeypatch.setenv("COMPANION_STREAM_API_KEY", "wrong-key")
    with pytest.raises(sc.StreamError) as err:
        sc.fetch_latest_frame("sess-1")
    assert err.value.code == "AUTH_INVALID_KEY"


def test_missing_frame_reports_error_code(server):
    with pytest.raises(sc.StreamError) as err:
        sc.fetch_latest_frame("empty-session")
    assert err.value.code == "STREAM_FRAME_NOT_FOUND"


def test_key_is_only_sent_in_header(server):
    sc.fetch_latest_frame("sess-1")
    sc.fetch_status("sess-1")
    assert all("apiKey" not in path for path, _ in FakeAiServer.seen)


def test_missing_settings_fail_before_any_request(monkeypatch):
    monkeypatch.setenv("COMPANION_STREAM_BASE_URL", "")
    monkeypatch.setenv("COMPANION_STREAM_API_KEY", "x")
    with pytest.raises(sc.StreamError) as err:
        sc.fetch_latest_frame("sess-1")
    assert err.value.code == "CONFIG_MISSING"


def test_unreachable_server_is_reported(monkeypatch):
    monkeypatch.setenv("COMPANION_STREAM_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("COMPANION_STREAM_API_KEY", "x")
    with pytest.raises(sc.StreamError) as err:
        sc.fetch_latest_frame("sess-1")
    assert err.value.code == "UNREACHABLE"
