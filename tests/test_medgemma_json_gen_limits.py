"""MedGemma 구조화(JSON) 생성 상한·반복 억제 스위치 - 2026-10-26.

MEDGEMMA_JSON_MAX_TOKENS·MEDGEMMA_JSON_REPETITION 을 넣지 않으면 생성 인자가 지금과 같아야 한다.
torch 없이 numpy 배열과 가짜 모듈로 generate 인자만 본다.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from app.core.config import Settings
from app.models.medgemma_loader import MedGemmaLoader, resolve_json_repetition

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---- 설정 우선순위 (환경변수는 import 시점에 읽으므로 별도 프로세스로 본다) --------------------------------


def _settings_from_env(env: dict[str, str]) -> dict:
    base = {k: v for k, v in os.environ.items() if not k.startswith(("MEDGEMMA_JSON_", "MAX_NEW_TOKENS"))}
    code = (
        "import json;from app.core.config import Settings;s=Settings();"
        "print(json.dumps([s.max_new_tokens,s.medgemma_json_max_tokens,s.medgemma_json_repetition]))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, env={**base, **env}, capture_output=True, text=True, check=True
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_json_max_tokens_defaults_to_max_new_tokens() -> None:
    assert _settings_from_env({}) == [512, 512, "off"]
    assert _settings_from_env({"MAX_NEW_TOKENS": "384"}) == [384, 384, "off"]
    assert _settings_from_env({"MEDGEMMA_JSON_MAX_TOKENS": ""}) == [512, 512, "off"]


def test_json_max_tokens_overrides_and_out_of_range_falls_back() -> None:
    assert _settings_from_env({"MEDGEMMA_JSON_MAX_TOKENS": "768"})[:2] == [512, 768]
    for bad in ("63", "2049", "abc", "-1"):
        assert _settings_from_env({"MAX_NEW_TOKENS": "400", "MEDGEMMA_JSON_MAX_TOKENS": bad})[:2] == [400, 400]


def test_repetition_env_is_normalized() -> None:
    assert _settings_from_env({"MEDGEMMA_JSON_REPETITION": " Penalty:1.05 "})[2] == "penalty:1.05"


# ---- 반복 억제 해석 -----------------------------------------------------------------------------------


def test_repetition_off_by_default_and_unknown_values() -> None:
    for value in ("off", "", None, "on", "penalty", "penalty:", "penalty:abc", "penalty:0.9", "penalty:1.6",
                  "ngram:1", "ngram:21", "ngram:2.5", "beam:3"):
        assert resolve_json_repetition(value) == ({}, "off")


def test_repetition_penalty_and_ngram() -> None:
    assert resolve_json_repetition("penalty:1.05") == ({"repetition_penalty": 1.05}, "penalty:1.05")
    assert resolve_json_repetition("penalty:1.5") == ({"repetition_penalty": 1.5}, "penalty:1.5")
    assert resolve_json_repetition("ngram:4") == ({"no_repeat_ngram_size": 4}, "ngram:4")


def test_invalid_repetition_warns_once_at_init(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="app.models.medgemma_loader"):
        MedGemmaLoader(_settings(medgemma_json_repetition="penalty:9"))
        MedGemmaLoader(_settings())  # off 는 경고 없음
    warns = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warns == ["[MEDGEMMA-JSON-REPETITION-INVALID] -> off"]  # 값 원문을 남기지 않는다


# ---- 생성 인자 (가짜 torch, numpy 배열) ------------------------------------------------------------------


class _Arr:
    def __init__(self, values: list[list[int]]) -> None:
        self.a = np.array(values)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.a.shape

    def to(self, *args: object, **kwargs: object) -> "_Arr":
        return self


class _Batch(dict):
    def to(self, *args: object, **kwargs: object) -> "_Batch":
        return self


_FAKE_TORCH = types.SimpleNamespace(
    float16="fp16", float32="fp32", inference_mode=contextlib.nullcontext, no_grad=contextlib.nullcontext
)
_SECRET_OUTPUT = '"reply": "상담 본문 비밀"}'


def _settings(**overrides: object) -> Settings:
    values = {"default_chat_model": "google/medgemma-1.5-4b-it", "chat_model_path": ""}
    values.update(overrides)
    return Settings(**values)


def _processor_loader(settings: Settings, new_tokens: int) -> tuple[MedGemmaLoader, dict]:
    seen: dict = {}

    class _Tok:
        unk_token_id = 0

        def convert_tokens_to_ids(self, token: str) -> int:
            return 0

    class _Proc:
        tokenizer = _Tok()

        def apply_chat_template(self, messages: object, **kwargs: object) -> _Batch:
            return _Batch(input_ids=np.array([[1, 2, 3]]))

        def decode(self, ids: object, skip_special_tokens: bool = True) -> str:
            return _SECRET_OUTPUT

    class _Model:
        device = "cpu"

        def generate(self, **kwargs: object) -> np.ndarray:
            seen.update(kwargs)
            return np.array([[1, 2, 3, *([9] * new_tokens)]])

    loader = MedGemmaLoader(settings)
    loader._model, loader._processor, loader._tokenizer, loader._torch = _Model(), _Proc(), None, _FAKE_TORCH
    loader._uses_processor = True
    loader._state.loaded = True
    return loader, seen


def test_defaults_keep_generation_args_identical() -> None:
    settings = _settings()
    loader, seen = _processor_loader(settings, 5)
    loader.generate_structured_text("sys", "user")
    assert seen["max_new_tokens"] == settings.max_new_tokens == 512
    assert seen["do_sample"] is False
    assert "repetition_penalty" not in seen and "no_repeat_ngram_size" not in seen


def test_json_settings_change_processor_generation() -> None:
    loader, seen = _processor_loader(_settings(medgemma_json_max_tokens=320, medgemma_json_repetition="ngram:6"), 5)
    loader.generate_structured_text("sys", "user")
    assert seen["max_new_tokens"] == 320
    assert seen["no_repeat_ngram_size"] == 6
    assert "repetition_penalty" not in seen

    loader, seen = _processor_loader(_settings(medgemma_json_repetition="penalty:1.05"), 5)
    loader.generate_structured_text("sys", "user")
    assert seen["repetition_penalty"] == 1.05


def test_gen_log_has_counts_only(caplog: pytest.LogCaptureFixture) -> None:
    loader, _ = _processor_loader(_settings(medgemma_json_max_tokens=64), 64)
    with caplog.at_level(logging.INFO, logger="app.models.medgemma_loader"):
        loader.generate_structured_text("sys", "user")
    loader, _ = _processor_loader(_settings(medgemma_json_max_tokens=64), 10)
    with caplog.at_level(logging.INFO, logger="app.models.medgemma_loader"):
        loader.generate_structured_text("sys", "user")
    messages = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[MEDGEMMA-GEN]")]
    assert messages == [
        "[MEDGEMMA-GEN] tokens=64 limit=64 hit_limit=True rep=off quant=none",
        "[MEDGEMMA-GEN] tokens=10 limit=64 hit_limit=False rep=off quant=none",
    ]
    assert all("비밀" not in r.getMessage() and "sys" not in r.getMessage() for r in caplog.records)


def test_non_processor_path_unchanged() -> None:
    settings = _settings(medgemma_json_max_tokens=128, medgemma_json_repetition="ngram:4")
    seen: dict = {}

    class _Tok:
        eos_token_id = 2

        def __call__(self, text: str, **kwargs: object) -> dict:
            return {"input_ids": _Arr([[1, 2]]), "attention_mask": _Arr([[1, 1]])}

        def decode(self, ids: object, skip_special_tokens: bool = True) -> str:
            return "답"

    class _Model:
        device = "cpu"

        def generate(self, **kwargs: object) -> np.ndarray:
            seen.update(kwargs)
            return np.array([[1, 2, 9]])

    loader = MedGemmaLoader(settings)
    loader._model, loader._tokenizer, loader._torch = _Model(), _Tok(), _FAKE_TORCH
    loader._uses_processor = False
    loader._state.loaded = True
    loader.generate_structured_text("sys", "user")
    assert seen["max_new_tokens"] == settings.max_new_tokens == 512
    assert seen["repetition_penalty"] == settings.repetition_penalty
    assert "no_repeat_ngram_size" not in seen
