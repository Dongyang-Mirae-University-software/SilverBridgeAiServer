"""위험 판단 어댑터 r1 (11/15): 얹히면 v9 위험 확률을 "risk" 어댑터로 계산하고 기준 0.5, 없으면 기본 모델 + 0.99.

COMPANION_RISK_THRESHOLD 가 있으면 그 값이 우선한다. 응답 riskModel 로 확률 눈금을 알린다 (백엔드 누적 기준용).
"""
from __future__ import annotations

import logging

import pytest
import torch

from app.core.config import get_settings
from app.routers.companion_router import _risk_model_out
from app.services.companion import pipeline
from app.services.companion.llm_client import MOOD_ADAPTER, REPLY_ADAPTER, RISK_ADAPTER, ExaoneLlm, LlmUnavailable
from app.services.companion.risk import (
    RISK_ADAPTER as RISK_NAME_IN_RISK,
    is_risk,
    log_risk_path,
    risk_model,
    risk_probability,
    risk_threshold,
)
from app.services.companion.text_emotion_service import TextEmotionResult

CLASSIFY = lambda _t: TextEmotionResult("평온", 0.9, True, {})  # noqa: E731


@pytest.fixture
def settings(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "companion_risk_prompt", "v9")
    monkeypatch.setattr(s, "companion_risk_threshold", None)
    monkeypatch.setattr(s, "companion_risk_lora_subdir", "companion/llm/lora-risk-r1")

    def set_(**kw):
        for k, v in kw.items():
            monkeypatch.setattr(s, k, v)
    return set_


class _Llm:
    """위험 확률만 주는 가짜 LLM. adapters 에 있는 어댑터만 얹힌 것으로 본다."""

    def __init__(self, p, adapters=(), load_fails=False):
        self.p, self.adapters, self.load_fails, self.calls = p, set(adapters), load_fails, []

    def has_adapter(self, name):
        if self.load_fails:
            raise LlmUnavailable("down", code="LOAD")
        return name in self.adapters

    def generate(self, *a, **k):
        raise LlmUnavailable("off")

    def message_token_probs(self, system, user, prefix, candidates, *, adapter=None, timeout):
        self.calls.append(adapter)
        return {"있": self.p, "없": 1 - self.p}


def test_adapter_names_match():
    assert RISK_ADAPTER == RISK_NAME_IN_RISK == "risk"


# --- 경로·기준 -----------------------------------------------------------------------------

def test_with_risk_adapter_uses_it_and_threshold_half(settings):
    llm = _Llm(0.6, adapters={MOOD_ADAPTER, REPLY_ADAPTER, RISK_ADAPTER})
    assert risk_probability(llm, "요즘 다 귀찮고 그만 살고 싶어") == 0.6
    assert llm.calls == [RISK_ADAPTER]
    assert risk_threshold(llm) == 0.5
    assert is_risk(0.6, llm) is True and is_risk(0.49, llm) is False
    assert risk_model(llm) == "v9+lora-risk-r1"


def test_without_risk_adapter_falls_back_to_base_and_099(settings):
    llm = _Llm(0.6, adapters={MOOD_ADAPTER, REPLY_ADAPTER})
    assert risk_probability(llm, "오늘 공원에 다녀왔어") == 0.6
    assert llm.calls == [None]
    assert risk_threshold(llm) == 0.99
    assert is_risk(0.6, llm) is False
    assert risk_model(llm) == "v9"


def test_load_failure_is_base_path(settings):
    llm = _Llm(0.6, load_fails=True)
    assert risk_threshold(llm) == 0.99 and risk_model(llm) == "v9"


def test_no_llm_keeps_old_default(settings):
    assert risk_threshold() == 0.99  # llm 없이 부르는 곳은 전과 같다


def test_explicit_threshold_wins_on_both_paths(settings):
    settings(companion_risk_threshold=0.8)
    assert risk_threshold(_Llm(0, adapters={RISK_ADAPTER})) == 0.8
    assert risk_threshold(_Llm(0)) == 0.8


def test_out_of_range_threshold_falls_back_to_path_default(settings):
    settings(companion_risk_threshold=1.5)
    assert risk_threshold(_Llm(0, adapters={RISK_ADAPTER})) == 0.5
    assert risk_threshold(_Llm(0)) == 0.99


def test_v4_prompt_ignores_risk_adapter(settings):
    settings(companion_risk_prompt="v4")
    llm = _Llm(0.6, adapters={RISK_ADAPTER})
    assert risk_threshold(llm) == 0.5 and risk_model(llm) == "v4"


def test_start_log_line(settings, caplog):
    with caplog.at_level(logging.INFO, logger="app.services.companion.risk"):
        log_risk_path(True)
        log_risk_path(False)
    lines = [r.getMessage() for r in caplog.records]
    assert "[COMPANION-RISK] path=v9+lora-risk-r1 threshold=0.50" in lines
    assert "[COMPANION-RISK] path=v9 threshold=0.99" in lines


# --- 파이프라인·응답 ------------------------------------------------------------------------

def _turn(llm, text="오늘은 공원에 다녀와서 기분이 좋아요"):
    return pipeline.run_turn(text, {}, CLASSIFY, llm, "q", None, False, 1, stt_logprob=-0.1)


@pytest.mark.parametrize("risk_first", [True, False])
def test_pipeline_flags_at_half_with_adapter(settings, risk_first):
    settings(companion_risk_first=risk_first)
    r = _turn(_Llm(0.7, adapters={RISK_ADAPTER}))
    assert r["riskFlag"] is True and r["replySource"] == "SAFETY"
    assert r["riskProbability"] == 0.7 and r["riskModel"] == "v9+lora-risk-r1"


@pytest.mark.parametrize("risk_first", [True, False])
def test_pipeline_same_probability_without_adapter_is_not_risk(settings, risk_first):
    settings(companion_risk_first=risk_first)
    r = _turn(_Llm(0.7))
    assert r["riskFlag"] is False and r["riskModel"] == "v9"


def test_low_confidence_turn_uses_adapter_threshold(settings):
    settings(companion_confirm_mode="logprob")
    r = pipeline.run_turn("음 그게 뭐더라", {}, CLASSIFY, _Llm(0.7, adapters={RISK_ADAPTER}), "q", None, False, 1,
                          stt_logprob=-9.0)
    assert r["riskFlag"] is True and r["riskModel"] == "v9+lora-risk-r1"


def test_risk_model_out_only_with_probability(settings):
    assert _risk_model_out({"riskProbability": 0.7, "riskModel": "v9+lora-risk-r1"}) == "v9+lora-risk-r1"
    assert _risk_model_out({"riskProbability": None, "riskModel": "v9"}) is None
    settings(companion_risk_prompt="v4")
    assert _risk_model_out({"riskProbability": 0.7, "riskModel": "v4"}) is None  # v4 확률은 내보내지 않는다


# --- 어댑터 폴더 설정 ----------------------------------------------------------------------

def test_adapter_dirs_include_risk_by_default(settings, tmp_path):
    settings(model_base_path=str(tmp_path))
    dirs = ExaoneLlm().adapter_dirs()
    assert dirs[RISK_ADAPTER] == tmp_path / "companion/llm/lora-risk-r1"
    assert {MOOD_ADAPTER, REPLY_ADAPTER} <= set(dirs)


def test_empty_subdir_turns_risk_adapter_off(settings, tmp_path):
    settings(model_base_path=str(tmp_path), companion_risk_lora_subdir="")
    assert RISK_ADAPTER not in ExaoneLlm().adapter_dirs()


# --- 실제 peft 전환 (작은 GPT2) -------------------------------------------------------------

def _tiny_gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(0)
    m = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=40, n_positions=16))
    return m.eval()


def _save_adapter(path, seed):
    from peft import LoraConfig, get_peft_model
    pm = get_peft_model(_tiny_gpt2(), LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"], fan_in_fan_out=True))
    torch.manual_seed(seed)
    for n, p in pm.named_parameters():
        if "lora_" in n:
            p.data.normal_(0, 0.5)
    pm.save_pretrained(str(path))


def _out(model):
    with torch.no_grad():
        return model(input_ids=torch.tensor([[1, 2, 3, 4]])).logits


@pytest.fixture
def three_adapters(tmp_path, settings):
    pytest.importorskip("peft")
    base = tmp_path / "models"
    for name, seed in (("mood", 1), ("reply", 2), ("risk", 3)):
        _save_adapter(base / "lm" / name, seed)
    settings(model_base_path=str(base), companion_mood_lora_subdir="lm/mood", companion_reply_lora_subdir="lm/reply",
             companion_risk_lora_subdir="lm/risk")
    return base


def test_three_adapters_switch_and_return_in_place(three_adapters):
    clean = _out(_tiny_gpt2())
    llm = ExaoneLlm()
    model = llm._attach_adapters(_tiny_gpt2())
    assert llm._adapters == {MOOD_ADAPTER, REPLY_ADAPTER, RISK_ADAPTER}
    llm._model = model
    outs = {}
    for name in (MOOD_ADAPTER, REPLY_ADAPTER, RISK_ADAPTER):
        llm._switch(name)
        outs[name] = _out(model)
    assert not torch.allclose(outs[RISK_ADAPTER], outs[MOOD_ADAPTER], atol=1e-4)
    assert not torch.allclose(outs[RISK_ADAPTER], outs[REPLY_ADAPTER], atol=1e-4)
    # 위험 → 기분·대답으로 돌아가도 처음 값 그대로, 기본 모델로 끄면 깨끗한 모델과 같다
    for name in (MOOD_ADAPTER, REPLY_ADAPTER):
        llm._switch(RISK_ADAPTER)
        llm._switch(name)
        assert torch.allclose(_out(model), outs[name], atol=1e-6)
    llm._switch(None)
    assert torch.allclose(_out(model), clean, atol=1e-6)


def test_broken_risk_adapter_drops_only_risk(three_adapters):
    (three_adapters / "lm" / "risk" / "adapter_model.safetensors").write_bytes(b"broken")
    llm = ExaoneLlm()
    llm._attach_adapters(_tiny_gpt2())
    assert llm._adapters == {MOOD_ADAPTER, REPLY_ADAPTER}
