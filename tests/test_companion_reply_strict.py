"""말벗 대답 검사 보강 (2026-11-21, COMPANION_REPLY_STRICT).

gosky 실측·품질 세트에서 남은 문제: ① 마지막 인사가 앞일을 한 일로 받음 ② 막연한 질문 ③ 계획 캐묻기 ④ 세 문장 꼬리질문.
①~③은 대체 문장, ④는 첫 문장 + 마지막 질문으로 줄인다. ③은 사용자 결정으로 계획·예정·언제쯤·하실 생각만 본다('왜·무슨 이유'는 보지 않음).
LLM 은 스텁이다 (API 키·GPU 없음).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import get_settings
from app.services.companion import pipeline
from app.services.companion.reply_service import (
    FALLBACK_LAST, check_reply, generate_reply, strict_failures, trim_reply,
)


@pytest.fixture
def settings(monkeypatch):
    s = get_settings()

    def set_(**kw):
        for k, v in kw.items():
            monkeypatch.setattr(s, k, v)
    set_(companion_reply_strict=True, companion_reply_lora=False, companion_llm_retry=0, companion_reply_relax_calm=True)
    return set_


class _FixedLlm:
    def __init__(self, text: str):
        self.text = text

    def generate(self, prompt, **kwargs):
        return self.text


# --- ① 마지막 인사: 앞일을 한 일로 ------------------------------------------------------------------------

@pytest.mark.parametrize("elder, closing", [
    ("이따가 김치찌개 남은 거 데워 먹으려고", "오늘도 따뜻한 김치찌개로 하루를 마무리하시는 모습이 보기 좋네요. 편안한 저녁 되세요."),
    ("내일은 시장에 가 보려고", "시장 잘 다녀오셨군요. 편안한 하루 보내세요."),
    ("저녁엔 국수 먹을 거야", "국수 맛있게 드셨네요. 편히 쉬세요."),
])
def test_closing_that_takes_future_as_done_is_caught(settings, elder, closing):
    assert strict_failures(closing, True, elder) == ["앞일을 한 일로"]


@pytest.mark.parametrize("elder, closing", [
    ("이따가 김치찌개 끓여 먹으려고", "김치찌개 끓여 드신다니 든든하시겠어요. 편안한 저녁 보내세요."),  # 앞일을 앞일로
    ("김치찌개 먹었어", "김치찌개 맛있게 드셨군요. 편안한 하루 보내세요."),  # 이미 한 일
])
def test_closing_ok_when_future_kept_or_already_done(settings, elder, closing):
    assert strict_failures(closing, True, elder) == []


def test_closing_falls_back_to_fixed_sentence(settings):
    r = generate_reply(_FixedLlm("김치찌개로 하루를 마무리하시는 모습이 보기 좋네요. 편안한 저녁 되세요."),
                       "저녁은 뭐 드실 거예요?", "이따가 김치찌개 데워 먹으려고", None, True, 8)
    assert r.source == "FALLBACK" and r.text in FALLBACK_LAST and r.failures == ["앞일을 한 일로"]


# --- ② 막연한 질문 · ③ 계획 캐묻기 --------------------------------------------------------------------------

@pytest.mark.parametrize("reply, name", [
    ("김치찌개 맛있게 드셨군요. 오늘 하루 중 가장 좋아하는 시간은 언제세요?", "막연한 질문"),
    ("전화라도 하셔서 다행이네요. 오늘 가장 즐거운 순간은 언제였나요?", "막연한 질문"),
    ("그냥 보내셨군요. 특별히 하고 싶은 활동은 없으셨나요?", "막연한 질문"),
    ("시장 가시는군요. 새로운 것을 사 오실 계획이신가요?", "계획 질문"),
    ("시장 가시는군요. 언제쯤 가실 건가요?", "계획 질문"),
    ("시장 가시는군요. 무엇을 사실 예정이세요?", "계획 질문"),
])
def test_vague_and_plan_questions_are_caught(settings, reply, name):
    assert strict_failures(reply, False, "내일은 시장에 가 보려고") == [name]


@pytest.mark.parametrize("reply, elder", [
    ("시장 가시는군요. 시장에는 자주 가세요?", "내일은 시장에 가 보려고"),
    ("속상하셨겠어요. 왜 그렇게 느끼셨어요?", "내가 귀찮은 존재인가 싶어"),  # '왜'는 보지 않는다 (사용자 결정)
    ("취미로 하시는군요. 취미는 얼마나 하셨어요?", "요즘 취미로 붓글씨를 써"),  # 어르신이 먼저 꺼낸 말
    ("계획이 있으시군요. 무엇을 하실 계획이세요?", "다음 주 계획이 있어"),
    ("가장 즐거운 시간이 있으셨다니 좋네요. 누구와 함께였어요?", "가장 즐거운 시간은 손주랑 놀 때야"),
])
def test_ok_questions_not_caught(settings, reply, elder):
    assert strict_failures(reply, False, elder) == []


def test_vague_word_only_in_empathy_sentence_is_ok(settings):
    # 질문 문장만 본다 - 받아 주는 문장의 '활동'은 괜찮다
    assert strict_failures("산책 같은 활동을 좋아하시는군요. 오늘은 어디까지 걸으셨어요?", False, "오늘 산책했어") == []


def test_vague_question_falls_back_with_reason_code(settings):
    r = generate_reply(_FixedLlm("그냥 보내셨군요. 특별히 하고 싶은 활동은 없으셨나요?"), "오후는 어떻게 보내세요?", "그냥 그래",
                       None, False, 2, text_mood={"usable": True, "label": "CALM", "turnLabel": "평온"})
    assert r.source == "FALLBACK" and "막연한 질문" in r.failures
    assert pipeline.REASON_CODES["막연한 질문"] == "RULE_VAGUE" and pipeline.REASON_CODES["계획 질문"] == "RULE_PLAN"
    assert pipeline.REASON_CODES["앞일을 한 일로"] == "RULE_FUTURE_DONE"


# --- ④ 세 문장 줄이기 ---------------------------------------------------------------------------------------

def test_three_sentences_trimmed_to_empathy_and_question(settings):
    t = "가족분께서 도움을 주시려는 마음이 보기 좋네요. 반찬 보내주신 것도 큰 힘이 되셨겠어요. 오늘 하루는 좀 더 편안하게 보내셨나요?"
    assert trim_reply(t, False) == "가족분께서 도움을 주시려는 마음이 보기 좋네요. 오늘 하루는 좀 더 편안하게 보내셨나요?"


@pytest.mark.parametrize("text, last", [
    ("반찬 맛있게 드셨군요. 어떤 반찬이었어요?", False),  # 두 문장은 그대로
    ("좋으셨겠네요. 다행이에요. 편히 쉬세요.", False),  # 마지막이 질문이 아니면 그대로 (규칙 검사가 따로 본다)
    ("즐거우셨겠네요. 오늘도 고맙습니다. 편안한 하루 보내세요.", True),  # 마지막 인사는 줄이지 않는다
])
def test_trim_leaves_others(settings, text, last):
    assert trim_reply(text, last) == text


def test_generate_reply_returns_trimmed_llm_sentence(settings):
    three = "반찬 보내 주셔서 다행이네요. 큰 힘이 되셨겠어요. 반찬은 맛있게 드셨나요?"
    r = generate_reply(_FixedLlm(three), "누구와 이야기하셨어요?", "그래도 지난번엔 반찬 보내 줬어", None, False, 4,
                       text_mood={"usable": True, "label": "CALM", "turnLabel": "평온"})
    assert r.source == "LLM" and r.text == "반찬 보내 주셔서 다행이네요. 반찬은 맛있게 드셨나요?"


# --- 스위치 -----------------------------------------------------------------------------------------------

def test_switch_off_is_the_old_behavior(settings):
    settings(companion_reply_strict=False)
    three = "그냥 보내셨군요. 오늘은 조용했네요. 특별히 하고 싶은 활동은 없으셨나요?"
    assert trim_reply(three, False) == three
    assert "막연한 질문" not in check_reply(three, False, "그냥 그래 오늘 보내셨")
    assert "앞일을 한 일로" not in check_reply("시장 잘 다녀오셨군요. 편히 쉬세요.", True, "내일 시장에 가 보려고")


def test_switch_default_on():
    from app.core import config
    assert '_as_bool("COMPANION_REPLY_STRICT", True)' in Path(config.__file__).read_text(encoding="utf-8")
