"""MedGemma JSON 멈춤·4비트 선택 스위치(MEDGEMMA_QUANT) - 2026-10-23.

적재 테스트는 torch·transformers 를 가짜 모듈로 바꿔 끼워 GPU·모델 없이 로드 인자만 본다.
생성 경로 테스트는 실제 torch 가 있을 때만 돈다(서비스 이미지).
"""

from __future__ import annotations

import logging
import sys
import types

import pytest

from app.core.config import Settings
from app.models import medgemma_loader as ml
from app.models.medgemma_loader import MedGemmaLoader, json_object_complete, resolve_quant


# ---- JSON 완성 판정 ---------------------------------------------------------------------------------


def test_complete_object_stops() -> None:
    assert json_object_complete('{"empathy": "힘드셨겠어요.", "reply": ""}')
    assert json_object_complete('{"a": "괄호 } 가 값 안에 있어도", "b": {"c": 1}}\n')


def test_incomplete_object_continues() -> None:
    assert not json_object_complete('{"empathy": "힘드셨')
    assert not json_object_complete('{"a": "값 안의 닫는 괄호}')  # 닫는 괄호로 끝나도 객체가 안 닫혔으면 계속
    assert not json_object_complete('{"a": {"b": 1}')
    assert not json_object_complete("")


def test_non_object_does_not_stop() -> None:
    assert not json_object_complete("[1, 2]")


def test_resolve_quant_defaults_to_none() -> None:
    assert resolve_quant("nf4") == "nf4"
    assert resolve_quant(" NF4 ") == "nf4"
    for value in ("none", "", None, "int8", "4bit"):
        assert resolve_quant(value) == "none"


def test_quantization_kwargs_none_does_not_import_bitsandbytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "bitsandbytes", raising=False)
    assert ml.quantization_kwargs("none", object()) == {}
    assert "bitsandbytes" not in sys.modules


# ---- 적재 인자 (가짜 torch·transformers) --------------------------------------------------------------


class _FakeModel:
    def __init__(self, kwargs: dict) -> None:
        self.kwargs = kwargs
        self.moved = False

    def to(self, device: str) -> "_FakeModel":
        if "quantization_config" in self.kwargs:
            raise ValueError("`.to` is not supported for 4-bit bitsandbytes models")
        self.moved = True
        return self


class _FakeBnbConfig:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def _install_fakes(monkeypatch: pytest.MonkeyPatch, *, bnb_installed: bool) -> list[dict]:
    calls: list[dict] = []
    torch = types.ModuleType("torch")
    torch.float16, torch.bfloat16, torch.float32 = "fp16", "bf16", "fp32"
    torch.cuda = types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None)

    class _AutoModel:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> _FakeModel:
            calls.append(dict(kwargs))
            return _FakeModel(dict(kwargs))

    class _AutoProcessor:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> object:
            return types.SimpleNamespace(tokenizer=object())

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForImageTextToText = _AutoModel
    transformers.AutoModelForCausalLM = _AutoModel
    transformers.AutoProcessor = _AutoProcessor
    transformers.AutoTokenizer = _AutoProcessor
    transformers.BitsAndBytesConfig = _FakeBnbConfig
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    if bnb_installed:
        monkeypatch.setitem(sys.modules, "bitsandbytes", types.ModuleType("bitsandbytes"))
    else:
        monkeypatch.setitem(sys.modules, "bitsandbytes", None)  # import 하면 ModuleNotFoundError
    return calls


def _settings(quant: str) -> Settings:
    return Settings(medgemma_quant=quant, default_chat_model="google/medgemma-1.5-4b-it", chat_model_path="")


def test_nf4_loads_with_4bit_config_and_skips_to(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fakes(monkeypatch, bnb_installed=True)
    loader = MedGemmaLoader(_settings("nf4"))
    loader.load()

    assert loader.state.loaded and loader.quant == "nf4"
    assert len(calls) == 1
    config = calls[0]["quantization_config"]
    assert config.kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bf16",
    }
    assert calls[0]["device_map"] == {"": 0}
    assert "load_in_8bit" not in calls[0]
    assert loader._model.moved is False  # 4비트 모델에 .to() 를 부르지 않는다


def test_none_loads_like_before_without_bitsandbytes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fakes(monkeypatch, bnb_installed=True)
    monkeypatch.delitem(sys.modules, "bitsandbytes")
    loader = MedGemmaLoader(_settings("none"))
    loader.load()

    assert loader.state.loaded and loader.quant == "none"
    assert len(calls) == 1
    assert "quantization_config" not in calls[0]
    assert calls[0]["dtype"] == "bf16"
    assert loader._model.moved is True
    assert "bitsandbytes" not in sys.modules


def test_nf4_failure_falls_back_to_none_with_one_warn(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    calls = _install_fakes(monkeypatch, bnb_installed=False)
    loader = MedGemmaLoader(_settings("nf4"))
    with caplog.at_level(logging.WARNING, logger="app.models.medgemma_loader"):
        loader.load()

    assert loader.state.loaded and loader.quant == "none"  # 기동을 막지 않는다
    assert len(calls) == 1 and "quantization_config" not in calls[0]
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1
    assert warns[0].getMessage() == "[MEDGEMMA-QUANT-FALLBACK] MEDGEMMA_QUANT=nf4 -> none (ModuleNotFoundError)"


# ---- 생성 경로: 멈춤 조건은 prefill="{" 일 때만 (실제 torch 필요) ----------------------------------------


class _Batch(dict):
    def to(self, *args: object, **kwargs: object) -> "_Batch":
        return self


def _generation_loader(torch: types.ModuleType, out_text: str) -> tuple[MedGemmaLoader, dict]:
    seen: dict = {}

    class _Tok:
        unk_token_id = 0

        def __call__(self, text: str, **kwargs: object) -> dict:
            return {"input_ids": torch.tensor([[7]])}

        def convert_tokens_to_ids(self, token: str) -> int:
            return 0

        def decode(self, ids: object, skip_special_tokens: bool = True) -> str:
            return out_text if len(ids) else ""

    class _Proc:
        tokenizer = _Tok()

        def apply_chat_template(self, messages: object, **kwargs: object) -> _Batch:
            return _Batch(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.tensor([[1, 1, 1]]))

        def decode(self, ids: object, skip_special_tokens: bool = True) -> str:
            return out_text

    class _Model:
        device = "cpu"

        def generate(self, **kwargs: object) -> object:
            seen.update(kwargs)
            return torch.cat([kwargs["input_ids"], torch.tensor([[9, 9]])], dim=-1)

    loader = MedGemmaLoader(_settings("none"))
    loader._model, loader._processor, loader._tokenizer, loader._torch = _Model(), _Proc(), _Proc.tokenizer, torch
    loader._uses_processor = True
    loader._state.loaded = True
    return loader, seen


def test_stop_criteria_only_with_json_prefill() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    loader, seen = _generation_loader(torch, '"a": 1}')
    assert loader.generate_structured_text("sys", "user", prefill="{") == '{"a": 1}'
    criteria = seen["stopping_criteria"]
    assert seen["max_new_tokens"] == loader._settings.max_new_tokens  # 상한은 그대로
    assert bool(criteria[0](torch.tensor([[1, 2, 3, 7, 9]]), None)[0]) is True  # 완성 객체 -> 멈춤

    loader, seen = _generation_loader(torch, '"a": 1')
    loader.generate_structured_text("sys", "user", prefill="{")
    assert bool(seen["stopping_criteria"][0](torch.tensor([[1, 2, 3, 7, 9]]), None)[0]) is False  # 미완성 -> 계속

    loader, seen = _generation_loader(torch, "안녕하세요")
    loader.generate_structured_text("sys", "user")
    assert "stopping_criteria" not in seen
