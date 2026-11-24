"""11/23 마지막 인사도 대답 LoRA 로 (COMPANION_REPLY_LAST_LORA, 기본 끔) + 마지막 인사 판정 시험 세트 형식."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.config import get_settings
from app.services.companion import reply_service as rs

ROOT = Path(__file__).resolve().parents[1]


class _Llm:
    def __init__(self):
        self.calls = []

    def has_adapter(self, name):
        return name == rs.REPLY_ADAPTER  # 꼬리질문 어댑터만 올라 있는 서버 (전용 마지막 인사 어댑터 없음)

    def generate_adapter(self, prompt, adapter, **kw):
        self.calls.append(("adapter", adapter))
        return "오늘도 고맙습니다. 편안한 하루 보내세요."

    def generate(self, prompt, **kw):
        self.calls.append(("base", None))
        return "오늘도 고맙습니다. 편안한 하루 보내세요."


@pytest.fixture
def settings(monkeypatch):
    s = get_settings()
    def set_(**kw):
        for k, v in kw.items():
            monkeypatch.setattr(s, k, v)
    set_(companion_reply_lora=True, companion_reply_last_lora=False, companion_reply_last_lora_subdir="")
    return set_


def test_default_on_with_cc8c():
    """11/24: 마지막 인사는 전용 어댑터 cc8c 로 (사용자 결정)."""
    src = (ROOT / "app/core/config.py").read_text(encoding="utf-8")
    assert '_as_bool("COMPANION_REPLY_LAST_LORA", True)' in src
    assert 'os.getenv("COMPANION_REPLY_LAST_LORA_SUBDIR", "companion/llm/lora-reply-cc8c")' in src


def test_last_turn_uses_base_model_unless_switched_on(settings):
    llm = _Llm()
    rs._generate(llm, "p", True)
    assert llm.calls[-1] == ("base", None)
    rs._generate(llm, "p", False)
    assert llm.calls[-1] == ("adapter", rs.REPLY_ADAPTER)  # 꼬리질문은 그대로 어댑터
    settings(companion_reply_last_lora=True)
    rs._generate(llm, "p", True)
    assert llm.calls[-1] == ("adapter", rs.REPLY_ADAPTER)


def test_reply_lora_off_keeps_base_even_for_last(settings):
    settings(companion_reply_lora=False, companion_reply_last_lora=True)
    llm = _Llm()
    rs._generate(llm, "p", True)
    assert llm.calls[-1] == ("base", None)


def test_closing_set_shape():
    d = json.loads((ROOT / "tests/companion_eval/closing_v1.json").read_text(encoding="utf-8"))
    items = d["items"]
    assert len(items) == 40 and len({x["id"] for x in items}) == 40
    kinds = {k: sum(x["kind"] == k for x in items) for k in ("future", "done", "mix", "feeling")}
    assert kinds == {"future": 15, "done": 10, "mix": 10, "feeling": 5}
    assert set(d["labels"]) == {"ok", "future_as_done", "done_as_future", "off"}
    assert all(len(x["history"]) == 2 and x["elder"] for x in items)


class _Llm2(_Llm):
    def __init__(self, have):
        super().__init__(); self.have = have

    def has_adapter(self, name):
        return name in self.have


def test_last_turn_prefers_dedicated_adapter(settings):
    settings(companion_reply_last_lora=True)
    llm = _Llm2({rs.REPLY_ADAPTER, rs.REPLY_LAST_ADAPTER})
    rs._generate(llm, "p", True)
    assert llm.calls[-1] == ("adapter", rs.REPLY_LAST_ADAPTER)
    rs._generate(llm, "p", False)
    assert llm.calls[-1] == ("adapter", rs.REPLY_ADAPTER)  # 꼬리질문은 그대로
    llm2 = _Llm2({rs.REPLY_ADAPTER})  # 전용 어댑터가 없으면 꼬리질문 어댑터
    rs._generate(llm2, "p", True)
    assert llm2.calls[-1] == ("adapter", rs.REPLY_ADAPTER)


def test_dedicated_adapter_ignored_when_last_lora_off(settings):
    llm = _Llm2({rs.REPLY_ADAPTER, rs.REPLY_LAST_ADAPTER})
    rs._generate(llm, "p", True)
    assert llm.calls[-1] == ("base", None)


def test_adapter_dirs_include_last_only_when_set(monkeypatch):
    from app.services.companion.llm_client import ExaoneLlm, REPLY_LAST_ADAPTER
    s = get_settings()
    monkeypatch.setattr(s, "companion_reply_last_lora_subdir", "")
    assert REPLY_LAST_ADAPTER not in ExaoneLlm().adapter_dirs()
    monkeypatch.setattr(s, "companion_reply_last_lora_subdir", "companion/llm/lora-reply-cc8c")
    assert str(ExaoneLlm().adapter_dirs()[REPLY_LAST_ADAPTER]).endswith("companion/llm/lora-reply-cc8c")

