"""최종 전체 점검 수정 PR-3(2026-11-21) - M-3 벼락두통 대안 좁힘 · M-5 묻지 말라는 말 · L-7 119 문장 보존 · L-8 마지막 인사 버전."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.core.config import get_settings
from app.routers import companion_router
from app.services import chat_keywords as ck
from app.services.companion.avoidance import is_avoidance
from app.services.companion.reply_service import reply_prompt_version
from app.services.medical_llm_service import MedicalLlmService


# ---- M-3: "평생·난생" 두통은 "처음·최악"이 함께 있어야 걸린다 --------------------------------------------------

@pytest.mark.parametrize("text", [
    "살면서 두통은 처음이에요 너무 아파요",
    "난생 처음 이렇게 머리가 아파요",
    "평생 이런 최악의 두통은 처음이에요",
    "갑자기 머리가 깨질 듯 아파요",
    "갑자기 머리가 터질 듯이 아파요",
    "살면서 이렇게 머리가 아픈 건 처음이야",
    "이런 두통은 난생 처음이에요",
    "태어나서 가장 최악의 두통이에요",
])
def test_thunderclap_headache_still_caught(text) -> None:
    assert "뇌출혈:두통" in ck.fallback_emergency_signals(text)


@pytest.mark.parametrize("text", [
    "평생 두통약을 먹고 살아요",
    "저는 평생 두통이 있어서 약을 먹어요",
    "살면서 머리가 아파본 적이 없어요",
    "난생 처음 두통약을 먹어 봤어요",
])
def test_chronic_headache_talk_not_caught(text) -> None:
    assert "뇌출혈:두통" not in ck.fallback_emergency_signals(text)


# ---- M-5: 말벗에게 묻지 말라는 말은 회피 ---------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "자꾸 묻지 말라고 했잖아",
    "물어보지 말라고 했지",
    "캐묻지 말라고 하잖아요",
    "자꾸 묻지 말라고 했잖아!",
])
def test_stop_asking_is_avoidance(text) -> None:
    assert is_avoidance(text), text


@pytest.mark.parametrize("text", [
    "그 얘기 하지 말라고 했지",               # 11/17 H-1 - 남 이야기 그대로
    "그런 얘기 그만 좀 하라고 했지",
    "아들이 묻지 말라고 했어",                # 가족이 주어인 남 이야기
    "누구한테 묻지 말고 병원 가래",
    "숨이 차서 묻지 말라고 했잖아",           # 몸 불편은 회피가 아니다
])
def test_other_person_and_body_cases_unchanged(text) -> None:
    assert not is_avoidance(text), text


# ---- L-7: 규칙 문구와 겹치는 문장을 지울 때 자연스러운 119 안내는 남는다 -------------------------------------------

NATURAL_119 = [
    "숨쉬기가 어렵거나 가슴이 조이듯 아프면 바로 119에 연락하세요.",
    "의식이 흐려지거나 말이 어눌해지면 지체하지 말고 119를 불러 주세요.",
    "증상이 갑자기 심해지면 119에 전화해 도움을 요청하세요.",
]


@pytest.mark.parametrize("sentence", NATURAL_119)
def test_natural_119_sentence_is_kept(sentence) -> None:
    assert MedicalLlmService.drop_prompt_leak(sentence) == (sentence, 0)


def test_natural_119_sentence_survives_next_to_a_leaked_one() -> None:
    leaked = "답을 짧게 써야 할 때도 이 첫 문장은 빼지 않고, 진료과 권유나 진료 기준으로 시작하지 않는다."
    out, removed = MedicalLlmService.drop_prompt_leak(f"{NATURAL_119[0]} {leaked}")
    assert removed == 1 and out == NATURAL_119[0]


def test_119_rule_wording_itself_is_dropped_but_not_the_natural_one() -> None:
    out, removed = MedicalLlmService.drop_prompt_leak(f"{NATURAL_119[2]} 이번에는 119 문장은 붙이지 않습니다.")
    assert removed == 1 and out == NATURAL_119[2]


# ---- L-8: 마지막 인사 promptVersion ---------------------------------------------------------------------------

def test_last_greeting_version_follows_prompt_actually_used(monkeypatch) -> None:
    s = get_settings()
    monkeypatch.setattr(s, "companion_reply_history", "summary")
    monkeypatch.setattr(s, "companion_reply_history_last", False)
    assert reply_prompt_version() == "reply-v4"                      # 꼬리질문은 v4
    assert reply_prompt_version(is_last_turn=True) == "reply-v2"     # 마지막 인사는 흐름 없는 v1-last (off 와 같은 파일)
    monkeypatch.setattr(s, "companion_reply_history_last", True)
    assert reply_prompt_version(is_last_turn=True) == "reply-v4"     # v2-last 는 지금처럼
    monkeypatch.setattr(s, "companion_reply_history", "off")
    assert reply_prompt_version(is_last_turn=True) == "reply-v2"


@pytest.mark.parametrize("follow_up,last_flag,expected", [
    (False, False, "reply-v2"), (True, False, "reply-v4"), (False, True, "reply-v4"),
])
def test_review_queue_records_version_of_last_greeting(monkeypatch, follow_up, last_flag, expected) -> None:
    s = get_settings()
    monkeypatch.setattr(s, "companion_reply_history", "summary")
    monkeypatch.setattr(s, "companion_reply_history_last", last_flag)
    seen = {}
    monkeypatch.setattr(companion_router.review_store, "add",
                        lambda *a, **k: seen.update(version=a[3]))
    d = SimpleNamespace(dialogueId="d-l8", turnNo=1, wardId="w-l8")
    companion_router._queue_for_review(d, {"replySource": "LLM", "reply": {"text": "안녕히 계세요.", "isFollowUp": follow_up}})
    assert seen["version"] == expected
