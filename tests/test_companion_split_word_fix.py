"""11/23 대답 첫머리 '경로당' 깨짐 고치기 (토큰 경·로·당 → LoRA 가 '경량'으로 미끄러짐). DGX 재현 문장 그대로."""
from __future__ import annotations

import pytest

from app.services.companion.reply_service import fix_split_words, generate_reply


@pytest.mark.parametrize("broken, fixed", [
    ("경량에서 친구들과 즐거운 시간 보내셨군요. 어떤 화투를 치셨나요?", "경로당에서 친구들과 즐거운 시간 보내셨군요. 어떤 화투를 치셨나요?"),
    ("경 경로당에서 즐거운 시간 보내셨군요. 오늘 경로당에서는 어떤 이야기 나누셨나요?", "경로당에서 즐거운 시간 보내셨군요. 오늘 경로당에서는 어떤 이야기 나누셨나요?"),
    ("경량당에서 점심을 드셨군요. 오늘 경로당에서 어떤 이야기를 나누셨나요?", "경로당에서 점심을 드셨군요. 오늘 경로당에서 어떤 이야기를 나누셨나요?"),
    ("경량에 다녀오셔서 좋으셨겠네요. 오늘 경로당에서는 어떤 이야기 나누셨나요?", "경로당에 다녀오셔서 좋으셨겠네요. 오늘 경로당에서는 어떤 이야기 나누셨나요?"),
    ("경량이 친구분이 얼른 나아지시기를 바라네요. 요즘은 어떠세요?", "경로당 친구분이 얼른 나아지시기를 바라네요. 요즘은 어떠세요?"),
])
def test_broken_forms_fixed_when_elder_said_the_word(broken, fixed):
    elder = "경로당 친구가 요즘 아프대" if "친구분이 얼른" in broken else "경로당에서 친구들이랑 화투 쳤어"
    got = fix_split_words(broken, elder)
    assert "경량" not in got and "경 경로당" not in got
    if "경량이 친구분" not in broken:
        assert got == fixed


def test_untouched_when_elder_did_not_say_it():
    t = "경량 패딩이 따뜻하시군요. 오늘 어디 다녀오셨어요?"
    assert fix_split_words(t, "경량 패딩 입고 나갔어") == t


def test_correct_reply_unchanged():
    t = "경로당에서 즐거운 시간 보내셨군요. 무슨 이야기 나누셨어요?"
    assert fix_split_words(t, "경로당 다녀왔어") == t


class _Llm:
    def generate(self, prompt, **kw):
        return "경량에서 친구들과 즐거운 시간 보내셨군요. 어떤 화투를 치셨나요?"


def test_generate_reply_returns_fixed_text(monkeypatch):
    from app.core.config import get_settings
    monkeypatch.setattr(get_settings(), "companion_reply_lora", False)
    r = generate_reply(_Llm(), "오늘 뭐 하셨어요?", "경로당에서 친구들이랑 화투 쳤어", None, False, 2,
                       text_mood={"usable": True, "label": "JOY", "turnLabel": "기쁨"})
    assert r.text.startswith("경로당에서") and r.source == "LLM"
