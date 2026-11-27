"""꼬리질문 막연한 질문 다시 만들기 (2026-11-25, COMPANION_REPLY_VAGUE_REDO, 기본 끔).

막연한 질문("함께 하시는 활동은", "그때까지 어떻게 지내실 계획", 내용 있는 대답의 "특별히 ~ 있으신가요")이면 한 번 다시 만들고,
다시 만들 때만 "대답 속 낱말로 구체적으로" 한 줄을 프롬프트 끝에 붙인다. 그래도 막연하면 대체 문장. LLM 은 스텁이다.
"""
from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.services.companion.reply_service import (
    VAGUE_REDO_HINT, generate_reply, is_flat_answer, is_vague_question,
)

VAGUE = "주말에 따님이 오시니 좋으시겠어요. 주말에 함께 하시는 활동은 있으신가요?"
CONCRETE = "주말에 따님이 오시니 좋으시겠어요. 따님 오시면 같이 뭐 드시고 싶으세요?"


@pytest.fixture
def settings(monkeypatch):
    s = get_settings()

    def set_(**kw):
        for k, v in kw.items():
            monkeypatch.setattr(s, k, v)
    # 다시 만들기(VAGUE_REDO)만 본다 - 고쳐 쓰기(11/27 기본 켬)는 끈다 (test_companion_reply_tense_fix 가 본다)
    set_(companion_reply_tense_fix="off", companion_reply_vague_rewrite=False,
         companion_reply_vague_redo=True, companion_reply_strict=False, companion_reply_lora=False, companion_llm_retry=0,
         companion_reply_history="off", companion_reply_similar_ratio=0.65)
    return set_


class _SeqLlm:
    """부를 때마다 차례로 문장을 돌려주고 받은 프롬프트를 남긴다."""

    def __init__(self, *texts):
        self.texts = list(texts)
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.texts[min(len(self.prompts), len(self.texts)) - 1]


@pytest.mark.parametrize("body,transcript", [
    (VAGUE, "주말에 온대"),
    ("다음 달에 손주가 오신다니 기대되시겠어요. 그때까지 어떻게 지내실 계획이세요?", "손주가 다음 달에 온대"),
    ("다음 달에 손주를 만나신다니 기대되시겠어요. 그때 어떤 계획이 있으신가요?", "손주가 다음 달에 온대"),
    ("그런 날도 있으시죠. 요즘 특별히 생각하시는 것이 있으신가요?", "없어"),
    ("반찬 만들 계획이시군요. 반찬 만들 때 특별히 좋아하시는 재료가 있으신가요?", "내일은 반찬 만들려고"),
])
def test_vague_questions_detected(body, transcript):
    assert is_vague_question(body, transcript)


@pytest.mark.parametrize("body,transcript", [
    ("조금 있다 나가시려는군요. 어디로 가실 계획이세요?", "좀 있다 나가려고"),  # 계획이라는 말만으로는 막연하지 않다
    ("국수 삶아 드시려는군요. 어떤 국수를 준비하시나요?", "이따가 국수 삶아 먹으려고"),
    ("그저 그랬다니 아쉽네요. 특별히 기억나는 일은 없으셨나요?", "그냥 그랬어"),  # 내용 없는 대답엔 '특별히'를 보지 않는다
    ("집에서 쉬셨군요. 특별히 읽으신 책이나 보신 프로그램이 있으셨나요?", "그냥 집에 있었어"),
    ("경로당 가시는군요. 거기서 어떤 활동을 하세요?", "경로당에서 활동하는 거 좋아해"),  # 어르신이 먼저 꺼낸 말
    ("네, 알겠어요.", "몰라"),  # 물음이 없으면 보지 않는다
])
def test_concrete_questions_pass(body, transcript):
    assert not is_vague_question(body, transcript)


def test_flat_answer_includes_geuraesseo():
    assert is_flat_answer("그냥 그랬어")
    assert not is_flat_answer("그냥 집에 있었어")


def test_vague_reply_is_redone_once_with_hint(settings):
    llm = _SeqLlm(VAGUE, CONCRETE)
    r = generate_reply(llm, "따님은 언제 오세요?", "주말에 온대", None, False, 1)
    assert r.source == "LLM" and r.text == CONCRETE
    assert len(llm.prompts) == 2
    assert not llm.prompts[0].endswith(VAGUE_REDO_HINT) and llm.prompts[1].endswith(VAGUE_REDO_HINT)


def test_still_vague_after_redo_falls_back(settings):
    llm = _SeqLlm(VAGUE, VAGUE)
    r = generate_reply(llm, "따님은 언제 오세요?", "주말에 온대", None, False, 1)
    assert r.source == "FALLBACK" and r.failures == ["막연한 질문"]
    assert len(llm.prompts) == 2


def test_switch_off_keeps_vague_reply(settings):
    settings(companion_reply_vague_redo=False)
    llm = _SeqLlm(VAGUE, CONCRETE)
    r = generate_reply(llm, "따님은 언제 오세요?", "주말에 온대", None, False, 1)
    assert r.source == "LLM" and r.text == VAGUE and len(llm.prompts) == 1


def test_last_turn_is_not_checked(settings):
    llm = _SeqLlm("주말에 따님이 오신다니 좋으시겠어요. 함께 하시는 활동은 있으신가요?")
    generate_reply(llm, "오늘 이야기 나눠 주셔서 고마워요.", "주말에 온대", None, True, 5)
    assert len(llm.prompts) == 1


def test_similar_redo_does_not_get_hint(settings):
    """지난 문장과 비슷해서 다시 만들 때는 힌트를 붙이지 않는다 (전과 같다)."""
    llm = _SeqLlm(CONCRETE, "따님 오신다니 반가우시겠어요. 오시면 어디 같이 가실 거예요?")
    generate_reply(llm, "따님은 언제 오세요?", "주말에 온대", None, False, 2, past_replies=(CONCRETE,))
    assert len(llm.prompts) == 2 and not llm.prompts[1].endswith(VAGUE_REDO_HINT)
