"""MedGemma 관리자 스위치: 재시작 없이 끄고 켠다. 끄면 올리지 않고 메모리에서 내린다."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.models.medgemma_loader import MedGemmaLoader
from app.routers import medgemma_switch_router as sw


class FakeLoader:
    def __init__(self):
        self._enabled = True
        self.calls = []

        class _State:
            loaded = True
            model_name = "fake"
            load_error = None

        self.state = _State()
        self.quant = "none"

    @property
    def enabled(self):
        return self._enabled

    def set_enabled(self, on):
        self._enabled = bool(on)
        self.calls.append(("set_enabled", on))

    def load(self):
        self.calls.append(("load",))
        self.state.loaded = True

    def warmup(self):
        self.calls.append(("warmup",))

    def unload(self):
        self.calls.append(("unload",))
        self.state.loaded = False
        self.state.load_error = "MEDGEMMA_DISABLED (관리자 스위치로 꺼짐)"


def _client(loader):
    app = FastAPI()
    app.include_router(sw.router)
    app.dependency_overrides[sw.get_loader] = lambda: loader
    return TestClient(app)


def test_status_reports_enabled_and_loaded():
    client = _client(FakeLoader())
    body = client.get("/api/admin/medgemma/status").json()
    assert body["success"] is True
    assert body["data"]["enabled"] is True
    assert body["data"]["loaded"] is True


def test_disable_turns_off_and_unloads():
    loader = FakeLoader()
    body = _client(loader).post("/api/admin/medgemma/disable").json()
    assert body["data"]["enabled"] is False
    assert body["data"]["loaded"] is False
    assert ("set_enabled", False) in loader.calls and ("unload",) in loader.calls


def test_enable_turns_on_loads_and_warms_up():
    loader = FakeLoader()
    loader.set_enabled(False)
    loader.state.loaded = False
    body = _client(loader).post("/api/admin/medgemma/enable").json()
    assert body["data"]["enabled"] is True
    assert ("load",) in loader.calls and ("warmup",) in loader.calls


def test_real_loader_does_not_load_when_disabled():
    loader = MedGemmaLoader(get_settings())
    loader.set_enabled(False)
    loader.load()  # 꺼져 있으면 모델 파일을 읽지 않는다
    assert loader.state.loaded is False
    assert loader.state.load_error == "MEDGEMMA_ENABLED=false"
    assert loader.enabled is False


def test_real_loader_unload_clears_state():
    loader = MedGemmaLoader(get_settings())
    loader.unload()
    assert loader.state.loaded is False
    assert "관리자 스위치" in loader.state.load_error
