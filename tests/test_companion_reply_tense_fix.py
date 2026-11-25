"""꼬리질문 시제·막연한 질문 고쳐 쓰기 (2026-11-25, COMPANION_REPLY_TENSE_FIX · COMPANION_REPLY_VAGUE_REWRITE, 기본 끔).

어르신이 앞으로 하실 일을 대답이 이미 한 일처럼 받으면 질문 어미를 앞일로 고친다. 막연한 질문은 공감 문장을 두고
어르신 말의 낱말로 만든 틀 질문으로 바꾼다. 고쳐도 규칙에 걸리면 모델 원문을 둔다 (대체 문장으로 바꾸지 않는다).
시험 문장은 판정 세트(followup_v1)와 겹치지 않게 새로 썼다. LLM 은 스텁이다.
"""
from __future__ import annotations

import pytest

from app.core.config import get_settings
from app.services.companion import reply_service
from app.services.companion.reply_service import (
    FALLBACK_AVOID, FALLBACK_EMPATHY, FALLBACK_FOLLOWUP, FALLBACK_NO_RESPONSE, TENSE_REDO_HINT, TOPIC_SWITCH,
    generate_reply, is_tense_error, is_vague_for_rewrite, rewrite_tense, rewrite_vague,
)


@pytest.fixture
def settings(monkeypatch):
    s = get_settings()

    def set_(**kw):
        for k, v in kw.items():
            monkeypatch.setattr(s, k, v)
    set_(companion_reply_tense_fix="off", companion_reply_vague_rewrite=False, companion_reply_vague_redo=False,
         companion_reply_strict=False, companion_reply_lora=False, companion_llm_retry=0,
         companion_reply_history="off", companion_reply_similar_ratio=0.65)
    return set_


class _SeqLlm:
    def __init__(self, *texts):
        self.texts = list(texts)
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.texts[min(len(self.prompts), len(self.texts)) - 1]


# ── 1. 꼬리질문 대체 문장은 두 문장까지 (품질 세트 R8) ──

@pytest.mark.parametrize("sentence", FALLBACK_FOLLOWUP + FALLBACK_EMPATHY + FALLBACK_NO_RESPONSE + FALLBACK_AVOID
                         + tuple(t for _, t in TOPIC_SWITCH))
def test_followup_fixed_sentences_have_at_most_two_sentences(sentence):
    assert len(reply_service._SENT_SPLIT.split(sentence.strip())) <= 2, sentence


# ── 2. 시제 검사 ──

@pytest.mark.parametrize("body,transcript", [
    ("고등어 사러 가시는군요. 고등어는 싸게 사셨나요?", "모레 시장에서 고등어 사려고"),
    ("미용실 가시는군요. 머리는 어떻게 자르셨어요?", "이따 미용실 가서 머리 자를 거야"),
    ("동창분들 만나시는군요. 동창분들과는 무슨 이야기 나누셨어요?", "다음 주에 동창들 만나기로 했어"),
    ("텃밭 가꾸시는군요. 텃밭 물 주기는 어떠셨나요?", "아침은 먹었고 오후엔 텃밭에 물 주려고"),
    ("사촌이 오셨다니 반가우셨겠어요. 무슨 이야기 하셨어요?", "사촌이 다음 주에 온대"),  # 오다 → 오셨다니
])
def test_future_taken_as_done_is_detected(body, transcript):
    assert is_tense_error(body, transcript)


@pytest.mark.parametrize("body,transcript", [
    ("고등어 사러 가시는군요. 어떤 반찬 해 드실 거예요?", "모레 시장에서 고등어 사려고"),  # 앞일로 맞게 받음
    ("미용실 다녀오셨군요. 머리는 어떻게 자르셨어요?", "아까 미용실 가서 머리 잘랐어"),  # 이미 한 일
    ("아침 드셨군요. 아침은 뭘 드셨어요?", "아침은 먹었고 오후엔 텃밭에 물 주려고"),  # 한 일 절을 받은 과거형
    ("동창분들 만나기로 하셨군요. 어디서 만나실 거예요?", "다음 주에 동창들 만나기로 했어"),  # 정한 일의 과거형은 맞다
    ("모임이 있으시군요. 장소는 정하셨어요?", "모레 계모임 하려고"),
    ("비 소식이 있네요. 우산은 챙기셨나요?", "비가 올 거 같아"),  # 짐작은 앞일이 아니다
    ("곧 나으실 거예요. 병원은 다녀오셨어요?", "괜찮아질 거야"),
    ("즐거우셨다니 다행이네요. 오늘 뭐 하셨어요?", "덕분에 즐거웠어요"),  # '즐거'의 '거'는 앞일 표지가 아니다
    ("나가신다니 좋네요. 옷은 따뜻하게 입으셨어요?", "동네 한 바퀴 돌려고"),  # 준비를 묻는 질문 (어르신 낱말 없음)
])
def test_correct_tense_is_not_flagged(body, transcript):
    assert not is_tense_error(body, transcript)


@pytest.mark.parametrize("body,transcript,expected", [
    ("고등어 사러 가시는군요. 고등어 고르기는 어려우셨나요?", "모레 시장에서 고등어 사려고",
     "고등어 사러 가시는군요. 고등어 고르기는 어려우신가요?"),
    ("동창분들 만나시는군요. 동창분들과는 무슨 이야기 나누셨어요?", "다음 주에 동창들 만나기로 했어",
     "동창분들 만나시는군요. 동창분들과는 무슨 이야기 나누실 건가요?"),
    ("텃밭 가꾸시는군요. 텃밭 물 주기는 어떠셨나요?", "오후엔 텃밭에 물 주려고",
     "텃밭 가꾸시는군요. 텃밭 물 주기는 어떠실 것 같으세요?"),
    ("사촌이 오셨다니 반가우셨겠어요. 사촌분은 어디 사세요?", "사촌이 다음 주에 온대",
     "그러시군요. 사촌분은 어디 사세요?"),  # 틀린 공감 문장만 바꾼다
])
def test_rewrite_tense(body, transcript, expected):
    assert rewrite_tense(body, transcript) == expected


def test_rewrite_tense_gives_up_when_ending_unknown():
    assert rewrite_tense("텃밭에 가시는군요. 텃밭 일은 몇 시간이었나요?", "오후엔 텃밭에 물 주려고") is None


# ── 3. 막연한 질문 고쳐 쓰기 ──

@pytest.mark.parametrize("body,transcript,expected", [
    ("텃밭 가꾸실 계획이시군요. 텃밭 일 할 때 특별히 좋아하시는 채소가 있으신가요?", "내일은 텃밭에 상추 심으려고",
     "텃밭 가꾸실 계획이시군요. 텃밭에서는 뭐 하실 거예요?"),
    ("사촌분이 오시면 반가우시겠어요. 사촌분과 함께 하실 계획이 있으신가요?", "사촌이 다음 주에 온대",
     "사촌분이 오시면 반가우시겠어요. 사촌과는 뭐 하실 거예요?"),
    ("화단 정리하셨군요. 특별히 좋아하시는 꽃이 있으신가요?", "아까 화단 정리했어",
     "화단 정리하셨군요. 화단 이야기 조금 더 들려주시겠어요?"),
])
def test_rewrite_vague(body, transcript, expected):
    assert is_vague_for_rewrite(body, transcript)
    assert rewrite_vague(body, transcript) == expected


@pytest.mark.parametrize("body,transcript,expected", [
    ("모레 오신다니 좋으시겠어요. 특별히 하고 싶은 활동이 있으신가요?", "모레 온대", "모레 오신다니 좋으시겠어요. 모레에는 뭐 하실 거예요?"),
    ("복지관 가시는군요. 복지관에서 특별히 즐거운 활동이 있으신가요?", "이따 복지관 가려고",
     "복지관 가시는군요. 복지관에서는 뭐 하실 거예요?"),
])
def test_rewrite_vague_time_and_place_particles(body, transcript, expected):
    assert rewrite_vague(body, transcript) == expected


def test_rewrite_vague_drops_subject_particle_ga():
    # '가'는 조사라 뗀다 ("조카가와는 뭐 하실 거예요?"가 되지 않게)
    assert rewrite_vague("조카가 온다니 반가우시겠어요. 그때 어떤 계획을 세우고 계신가요?", "조카가 내일 온대") == \
        "조카가 온다니 반가우시겠어요. 조카와는 뭐 하실 거예요?"


@pytest.mark.parametrize("body,transcript,expected", [
    # 부사(혼자)는 고르지 않고, 조사 '만'은 뗀다
    ("혼자 텔레비전 보셨군요. 특별히 좋아하시는 프로그램이 있으신가요?", "종일 혼자 텔레비전만 봤어",
     "혼자 텔레비전 보셨군요. 텔레비전 이야기 조금 더 들려주시겠어요?"),
    # 띄어 쓴 '마음이'의 '이'는 조사다
    ("마음이 무거우시겠어요. 특별히 걱정되는 일이 있으신가요?", "허리가 아파서 마음이 무거워",
     "마음이 무거우시겠어요. 마음 이야기 조금 더 들려주시겠어요?"),
    # '~요일'은 시간 낱말
    ("금요일에 오신다니 반가우시겠어요. 그때 어떤 계획을 세우고 계신가요?", "금요일에 온대",
     "금요일에 오신다니 반가우시겠어요. 금요일에는 뭐 하실 거예요?"),
])
def test_rewrite_vague_picks_nouns_only(body, transcript, expected):
    assert rewrite_vague(body, transcript) == expected


@pytest.mark.parametrize("question", ["손자분 선물은 무엇이셨나요?", "준비하시느라 힘드시지 않으셨나요?"])
def test_rewrite_tense_gives_up_on_copula_and_negation(question):
    body = "손자분 오시는군요. " + question
    assert rewrite_tense(body, "다음 주에 손자 오려고") is None


def test_rewrite_vague_skips_one_letter_word_with_particle():
    # "집에" → "집"(한 글자)은 쓰지 않는다. 조사째 쓰면 "집에 이야기 조금 더…"가 된다
    assert rewrite_vague("집에서 쉬셨군요. 특별히 좋아하시는 활동이 있으신가요?", "종일 집에 있었어") is None


def test_rewrite_vague_gives_up_without_a_word():
    # 어르신 말의 낱말이 대답에 하나도 없으면 고치지 않는다
    assert rewrite_vague("편히 쉬셨군요. 특별히 좋아하시는 활동이 있으신가요?", "응 그랬어") is None


def test_plan_question_echoing_elder_is_not_vague():
    assert not is_vague_for_rewrite("계획 세우셨군요. 어떤 계획을 세우셨어요?", "올해 계획을 세웠어")


# ── 4. generate_reply 연결 ──

TENSE_BAD = "고등어 사러 가시는군요. 고등어 고르기는 어려우셨나요?"
TENSE_GOOD = "고등어 사러 가시는군요. 고등어로 무슨 반찬 하실 거예요?"
T = "모레 시장에서 고등어 사려고"


def test_switches_off_keep_model_text(settings):
    out = generate_reply(_SeqLlm(TENSE_BAD), "장 보러 가세요?", T, None, False, 2)
    assert out.text == TENSE_BAD and out.source == "LLM"


def test_rewrite_mode_fixes_ending_without_new_generation(settings):
    settings(companion_reply_tense_fix="rewrite")
    llm = _SeqLlm(TENSE_BAD)
    out = generate_reply(llm, "장 보러 가세요?", T, None, False, 2)
    assert out.text == "고등어 사러 가시는군요. 고등어 고르기는 어려우신가요?" and out.source == "LLM"
    assert len(llm.prompts) == 1


def test_redo_mode_uses_hint_once(settings):
    settings(companion_reply_tense_fix="redo")
    llm = _SeqLlm(TENSE_BAD, TENSE_GOOD)
    out = generate_reply(llm, "장 보러 가세요?", T, None, False, 2)
    assert out.text == TENSE_GOOD and len(llm.prompts) == 2 and llm.prompts[1].endswith(TENSE_REDO_HINT)


def test_redo_still_wrong_falls_back_to_rewrite_not_fallback(settings):
    settings(companion_reply_tense_fix="redo")
    llm = _SeqLlm(TENSE_BAD, TENSE_BAD)
    out = generate_reply(llm, "장 보러 가세요?", T, None, False, 2)
    assert out.text == "고등어 사러 가시는군요. 고등어 고르기는 어려우신가요?" and out.source == "LLM"


def test_unfixable_keeps_original(settings):
    settings(companion_reply_tense_fix="rewrite")
    body = "텃밭에 가시는군요. 텃밭 일은 몇 시간이었나요?"
    out = generate_reply(_SeqLlm(body), "오늘 뭐 하세요?", "오후엔 텃밭에 물 주려고", None, False, 2)
    assert out.text == body and out.source == "LLM"


def test_vague_rewrite_on(settings):
    settings(companion_reply_vague_rewrite=True)
    body = "화단 정리하셨군요. 특별히 좋아하시는 꽃이 있으신가요?"
    out = generate_reply(_SeqLlm(body), "오늘 뭐 하셨어요?", "아까 화단 정리했어", None, False, 2)
    assert out.text == "화단 정리하셨군요. 화단 이야기 조금 더 들려주시겠어요?"


def test_last_turn_is_not_touched(settings):
    settings(companion_reply_tense_fix="rewrite", companion_reply_vague_rewrite=True)
    body = "고등어 맛있게 드셨다니 다행이에요. 오늘도 고맙습니다."
    out = generate_reply(_SeqLlm(body), "장 보러 가세요?", T, None, True, 7)
    assert out.text == body


def test_logs_have_no_text(settings, caplog):
    settings(companion_reply_tense_fix="rewrite", companion_reply_vague_rewrite=True)
    with caplog.at_level("INFO"):
        generate_reply(_SeqLlm(TENSE_BAD), "장 보러 가세요?", T, None, False, 2)
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "[COMPANION-REPLY-TENSE] action=rewrite" in joined and "고등어" not in joined
