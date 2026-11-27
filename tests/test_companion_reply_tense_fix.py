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
    FALLBACK_AVOID, FALLBACK_EMPATHY, FALLBACK_FOLLOWUP, FALLBACK_NO_RESPONSE, TOPIC_SWITCH,
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
    ("즐거우셨다니 다행이네요. 오후에는 무엇을 하셨어요?", "덕분에 즐거웠어요"),  # '즐거'의 '거'는 앞일 표지가 아니다
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


def test_rewrite_tense_state_ending_for_iss():
    # '수 있으셨나요'는 상태로 본다 ("쉬실 수 있으실 건가요?"가 되지 않게)
    assert rewrite_tense("텃밭 가시는군요. 텃밭에서는 좀 쉬실 수 있으셨나요?", "오후엔 텃밭에 물 주려고") == \
        "텃밭 가시는군요. 텃밭에서는 좀 쉬실 수 있으신가요?"


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
    # '과'로 끝나는 이름말은 떼지 않는다
    ("이비인후과 예약이 잡히셨군요. 특별히 걱정되는 점이 있으신가요?", "이비인후과 예약했어",
     "이비인후과 예약이 잡히셨군요. 이비인후과 이야기 조금 더 들려주시겠어요?"),
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


@pytest.mark.parametrize("body,transcript", [
    ("심심하시겠네요. 특별히 좋아하시는 놀이가 있으신가요?", "그래도 하루 종일 심심하고 처지네"),  # '심심하고'는 이름말이 아니다
    ("비가 올까 걱정되시는군요. 특별히 챙기실 물건이 있으신가요?", "비가 올까 걱정이야 우산이 없어서"),  # '올까'는 동사
])
def test_rewrite_vague_skips_verbs_and_adjectives(body, transcript):
    out = rewrite_vague(body, transcript)
    assert out is None or ("심심 이야기" not in out and "올까" not in out.split(". ")[-1])


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




def test_unfixable_keeps_original(settings):
    settings(companion_reply_tense_fix="rewrite")
    body = "텃밭에 가시는군요. 텃밭 일은 몇 시간이었나요?"
    out = generate_reply(_SeqLlm(body), "오늘 뭐 하세요?", "오후엔 텃밭에 물 주려고", None, False, 2)
    assert out.text == body and out.source == "LLM"


def test_vague_rewrite_on(settings):
    settings(companion_reply_vague_rewrite=True)
    body = "화단 정리하셨군요. 특별히 좋아하시는 꽃이 있으신가요?"
    out = generate_reply(_SeqLlm(body), "오전엔 무엇을 하셨어요?", "아까 화단 정리했어", None, False, 2)
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


# ── 5. 11/26 기능 점검 반영 ──

@pytest.mark.parametrize("body,transcript", [
    # H-1: 지난 일 절이 있으면 공감 문장의 과거형은 지난 일을 받은 것일 수 있다
    ("감기에 걸리셨군요. 병원은 어디로 가실 거예요?", "어제 감기 걸렸는데 내일 병원 가려고요"),
    ("손주가 오셨군요. 내일은 손주와 뭐 하실 거예요?", "어제 손주가 왔다 갔는데 내일 또 온대요"),
    ("어제도 가셨군요. 시장에서는 뭘 사실 거예요?", "이따 시장 가려고요, 어제도 갔어요"),
    # H-1: 지금 마음을 받는 "~셨겠어요"는 과거형이 아니다
    ("걱정되셨겠어요. 검사는 어디서 받으세요?", "내일 검사받으려고요 걱정돼요"),
])
def test_correct_empathy_with_past_clause_is_kept(body, transcript):
    assert not is_tense_error(body, transcript)


@pytest.mark.parametrize("body,transcript", [
    # 받침 ㅆ 글자 뒤 연결어미(는데·고·더니)와 앞일 어미 뒤에서 절을 나눈다 - 지난 일 절의 공감 문장은 맞다
    ("손이 많이 시리셨군요. 장갑은 어디서 사실 거예요?", "손이 시렸는데 이따 장갑 사러 갈 거야"),
    ("이사하셨군요. 짐 정리는 누가 도와주실 거예요?", "그저께 이사했는데 내일 짐 정리하려고"),
    ("우유가 떨어지셨군요. 마트에서 또 뭘 사실 거예요?", "이따 마트 가려고 우유가 떨어졌거든"),
    ("다리가 아프셨군요. 오늘은 집에서 쉬실 거예요?", "어제 많이 걸었더니 다리가 아파서 오늘은 쉬려고"),
])
def test_past_clause_split_by_final_ss_and_future_ending(body, transcript):
    assert not is_tense_error(body, transcript)


def test_time_word_only_in_future_clause_marks_question():
    # 지난 일 절이 있어도 질문에 앞일 절의 시간 낱말 + 과거형이면 앞일을 한 일로 받은 것이다
    body = "따님이 다녀가셨군요. 다음 달에 따님과 어디 다녀오셨어요?"
    assert rewrite_tense(body, "주말에 딸이 왔었고 다음 달에 또 오기로 했어") == \
        "따님이 다녀가셨군요. 다음 달에 따님과 어디 다녀오실 건가요?"


def test_question_is_still_checked_when_past_clause_exists():
    # 지난 일 절이 있어도 앞일을 과거로 묻는 질문은 본다
    body = "감기에 걸리셨군요. 병원은 잘 다녀오셨나요?"
    assert is_tense_error(body, "어제 감기 걸렸는데 내일 병원 가려고요")


@pytest.mark.parametrize("question,expected", [
    ("등산은 많이 힘드셨나요?", "등산은 많이 힘드신가요?"),  # M-1 상태 형용사는 지금으로
    ("동창 만나서 행복하셨어요?", "동창 만나서 행복하신가요?"),
    ("병원 다녀오시면 허리가 좀 나으셨나요?", "병원 다녀오시면 허리가 좀 나아지실까요?"),
])
def test_state_adjectives_become_present(question, expected):
    # 재점검 2: 예문을 앞일 행사에 대한 상태 질문으로 바꿨다 ("허리가 아파서 …"의 허리 통증은 지금 상태라 시제 오류가 아니다)
    body = "그러시군요. " + question
    out = rewrite_tense(body, "모레 등산 가서 동창 만나고 병원 가서 허리 치료 받으려고")
    assert out == "그러시군요. " + expected


@pytest.mark.parametrize("question", ["동창분이 아니셨나요?", "동창 만나서 이야기하셨나요?", "동창 만나서 안심되셨나요?"])
def test_unknown_hada_doeda_and_anida_endings_are_not_rewritten(question):
    assert rewrite_tense("동창 만나시는군요. " + question, "내일 동창 만나려고") is None


def test_past_with_yeot_is_detected():
    # M-2 '였'도 과거형 (바꿀 어미가 없어 고치지는 않는다)
    body = "산책 가시는군요. 산책하기 좋은 날씨였나요?"
    assert is_tense_error(body, "이따 공원에 산책 가려고")
    assert rewrite_tense(body, "이따 공원에 산책 가려고") is None


def test_future_template_only_for_time_place_person():
    # M-4 그 밖의 낱말은 "~ 이야기 조금 더"
    out = rewrite_vague("미역국 끓이시는군요. 특별히 좋아하시는 재료가 있으신가요?", "내일 미역국 끓이려고")
    assert out == "미역국 끓이시는군요. 미역국 이야기 조금 더 들려주시겠어요?"


def test_compound_person_words_are_joined():
    out = rewrite_vague("사촌 동생 만나시는군요. 함께 하실 계획이 있으신가요?", "내일 사촌 동생 만나려고")
    assert out == "사촌 동생 만나시는군요. 사촌 동생과는 뭐 하실 거예요?"


def test_redo_value_falls_back_to_off(monkeypatch, caplog):
    # H-2 redo 는 없앴다 - 예전 값은 경고 후 off
    from app.core.config import _as_choice
    monkeypatch.setenv("COMPANION_REPLY_TENSE_FIX", "redo")
    with caplog.at_level("WARNING"):
        assert _as_choice("COMPANION_REPLY_TENSE_FIX", "off", ("off", "rewrite")) == "off"
    assert "redo" not in " ".join(r.getMessage() for r in caplog.records)  # 값은 로그에 남기지 않는다


def test_switches_off_call_llm_once(settings):
    # L-1 스위치를 끄면 LLM 을 한 번만 부른다 (고쳐 쓰기가 다시 부르지 않는다)
    llm = _SeqLlm(TENSE_BAD)
    generate_reply(llm, "장 보러 가세요?", T, None, False, 2)
    assert len(llm.prompts) == 1


def test_rewrite_never_calls_llm_again(settings):
    settings(companion_reply_tense_fix="rewrite", companion_reply_vague_rewrite=True)
    llm = _SeqLlm(TENSE_BAD)
    generate_reply(llm, "장 보러 가세요?", T, None, False, 2)
    assert len(llm.prompts) == 1


def test_rewritten_text_similar_to_past_keeps_original(settings):
    # M-3 고친 문장이 지난 말벗 문장과 비슷하면 원문을 둔다 - 고쳐 쓰기 단계만 직접 본다 (11/26 재점검 N-7:
    #   generate_reply 로 보면 뒤이은 반복 검사가 같은 원문을 다시 만들어 조치를 지워도 통과했다)
    from app.services.companion.reply_service import _apply_followup_fixes
    settings(companion_reply_vague_rewrite=True)
    body = "화단 정리하셨군요. 특별히 좋아하시는 꽃이 있으신가요?"
    past = ("화단 정리하셨군요. 화단 이야기 조금 더 들려주시겠어요?",)
    assert _apply_followup_fixes(body, "아까 화단 정리했어", None, past) == body
    assert _apply_followup_fixes(body, "아까 화단 정리했어", None, ()) == past[0]


def test_negative_mood_turn_skips_vague_rewrite(settings):
    # L-4 글 기분이 부정이면 막연한 질문을 고치지 않는다
    settings(companion_reply_vague_rewrite=True)
    body = "화단 정리하시느라 힘드셨겠어요. 특별히 걱정되는 일이 있으신가요?"
    mood = {"usable": True, "turnLabel": "우울"}
    out = generate_reply(_SeqLlm(body), "오전엔 무엇을 하셨어요?", "아까 화단 정리했어 힘들어", None, False, 2, text_mood=mood)
    assert out.text == body


def test_retry_setting_with_rewrite(settings):
    # L-1 재시도 설정이 있어도 고쳐 쓴 대답이 바로 나간다
    settings(companion_reply_tense_fix="rewrite", companion_llm_retry=1)
    llm = _SeqLlm(TENSE_BAD)
    out = generate_reply(llm, "장 보러 가세요?", T, None, False, 2)
    assert out.text == "고등어 사러 가시는군요. 고등어 고르기는 어려우신가요?" and len(llm.prompts) == 1


def test_vague_redo_and_rewrite_together(settings):
    # L-1 VAGUE_REDO 와 함께 켜면 고쳐 쓰기가 먼저 - 고친 문장은 막연하지 않아 다시 만들지 않는다
    settings(companion_reply_vague_rewrite=True, companion_reply_vague_redo=True)
    body = "화단 정리하셨군요. 특별히 좋아하시는 꽃이 있으신가요?"
    llm = _SeqLlm(body)
    out = generate_reply(llm, "오전엔 무엇을 하셨어요?", "아까 화단 정리했어", None, False, 2)
    assert out.text == "화단 정리하셨군요. 화단 이야기 조금 더 들려주시겠어요?" and len(llm.prompts) == 1


# ── 6. 11/26 재점검 반영 (조치 작성자와 다른 작성자의 문장에서 나온 오탐) ──

@pytest.mark.parametrize("body,transcript", [
    # N-1 지난 의도·목적의 '려고'는 앞일이 아니다
    ("비 때문에 못 가셨군요. 친구분은 만나셨나요?", "친구 만나러 가려고 했는데 비가 와서 못 갔어"),
    ("복지관에 가시려고 하셨군요. 복지관에는 가 보셨나요?", "복지관에 가려고 했어 어제"),
    ("당면을 불리셨군요. 잡채는 완성하셨나요?", "잡채를 하려고 당면을 불렸어"),
    # N-2 '-어서' 연결의 지난 일
    ("약을 드셨군요. 약은 잘 들으셨나요?", "약을 먹었어서 괜찮은데 이따 병원 가 보려고"),
    ("잠을 못 주무셨군요. 어제는 왜 못 주무셨나요?", "어제 잠을 못 자서 오늘 일찍 자려고"),
    # N-3 문장부호 없이 이어진 지난 일
    ("드라마 보셨군요. 어떤 내용이었나요?", "어제 드라마를 봤어 내일도 볼 거야"),
])
def test_recheck_false_positives_are_gone(body, transcript):
    assert not is_tense_error(body, transcript)


def test_purpose_with_future_time_word_is_still_future():
    # 앞일 시간 낱말이 있으면 '려고' 뒤가 지난 일이어도 앞일이다 ("이따 마트 가려고 우유가 떨어졌거든")
    assert is_tense_error("우유가 떨어지셨군요. 마트에서는 뭘 사셨나요?", "이따 마트 가려고 우유가 떨어졌거든")


@pytest.mark.parametrize("body,transcript", [
    # N-4 앞일 표지 확장
    ("아드님이 오신다니 좋으시겠어요. 저녁에 아드님과 무엇을 드셨나요?", "저녁에 아들이 온다고 해서 장을 봤어"),
    ("상추 심으시려는군요. 상추는 잘 자라셨나요?", "주말에 상추를 심을까 해"),
    ("병원에 가셔야 하는군요. 다음 주 병원은 어떠셨나요?", "다음 주에 병원 가야 돼"),
])
def test_recheck_missed_future_markers_are_caught(body, transcript):
    assert is_tense_error(body, transcript)


def test_wish_is_not_future():
    # 소망은 앞일로 보지 않는다
    assert not is_tense_error("꽃구경 좋으시죠. 꽃구경은 다녀오셨나요?", "꽃구경 가고 싶어")


def test_become_ending_is_not_rewritten():
    # N-6 '-어지셨'(상태 변화)은 고치지 않는다
    assert rewrite_tense("눈이 침침하시군요. 눈이 왜 침침해지셨어요?", "눈이 침침해서 안과에 가 보려고") is None


def test_person_followed_by_noun_does_not_use_person_template():
    # N-5 "남편 산소" - 사람 틀("남편과는 뭐 하실 거예요?")을 쓰지 않는다
    out = rewrite_vague("남편 산소에 가시는군요. 그날 특별히 하실 계획이 있으신가요?", "주말에 남편 산소에 가려고")
    assert out is None or "남편과는" not in out


def test_past_intention_with_future_time_word_is_not_future():
    # '려고' 바로 뒤가 '했'이면 지난 의도다 - 앞일 시간 낱말(저녁)이 있어도 ("저녁에 가려고 했는데 비가 와서 못 갔어")
    assert not is_tense_error("비 때문에 못 가셨군요. 저녁에는 집에서 뭐 하셨어요?", "저녁에 가려고 했는데 비가 와서 못 갔어")


def test_agreement_geuraeyaji_is_not_future():
    # 맞장구 "그래야지"는 앞일 표지가 아니다
    assert not is_tense_error("불면증이 있으시군요. 어떻게 심해지셨나요?", "그래야지 근데 불면증이 있는데 요새 더 심해져서 큰일이야")


@pytest.mark.parametrize("transcript,future", [
    ("내일 병원 가야 돼", True), ("이따 고기 구워야겠어", True), ("모레는 꼭 가야지", True),
    ("손주가 아직 아기야 지금은 잘 자", False),  # '야' 뒤 띄어 쓴 '지금'은 앞일이 아니다
    ("그래야 돼 다들 그렇게 해", False),
])
def test_must_ending_marker(transcript, future):
    assert any(f for _, f in reply_service._clause_tags(transcript)) is future


# ── 7. 11/26 재점검 2 반영 ──

@pytest.mark.parametrize("body,transcript", [
    # P-1 지난 일 절이 있어도 앞일 절에만 나온 낱말 + 과거형이면 공감 문장도 잡는다
    ("국수를 드셨군요. 맛은 어떠셨어요?", "점심은 먹었고 저녁엔 국수 삶아 먹으려고"),
    ("김장을 끝내셨군요. 김치 맛은 어떻게 나왔어요?", "배추 사다 놨어 내일 김장할 거야"),
    ("병원에 다녀오셨군요. 의사 선생님은 뭐라고 하셨어요?", "어제부터 배가 아파서 내일 병원 가려고"),
])
def test_empathy_about_future_part_is_caught_with_past_clause(body, transcript):
    assert is_tense_error(body, transcript)


def test_empathy_about_past_part_is_kept():
    # 지난 일 절의 낱말(점심)을 받은 공감은 맞다
    assert not is_tense_error("점심 맛있게 드셨군요. 국수는 무슨 국수로 하실 거예요?", "점심은 먹었고 저녁엔 국수 삶아 먹으려고")


@pytest.mark.parametrize("transcript,future", [
    ("내일 병원에 갔다 올 거야", True),  # P-2 '-다'에서 자르지 않는다
    ("내일 소풍 가", True), ("다음 주에 손주 돌잔치 가", True),  # P-3 앞일 시간 낱말 + 현재형
    ("딸이 주말에 전화한다고 했어", True), ("아들이 용돈 준대", True),  # P-3 '-ㄴ다고·-ㄴ대'
    ("저녁 먹어", False), ("내일이 아니라 어제 갔어", False),  # 저녁은 앞일 시간 낱말에서 뺀다 / 지난 일 표지가 있으면 아니다
])
def test_future_clause_detection_recheck2(transcript, future):
    assert any(f for _, f in reply_service._clause_tags(transcript)) is future


@pytest.mark.parametrize("transcript,pieces", [
    ("시간이 있었어 내일도 갈 거야", 2),  # P-5 '있었'의 '었'에서 자른다
    ("애들이 밥은 먹었겠지 내일 가서 볼 거야", 1),  # '겠'(짐작)에서는 자르지 않는다
    ("시간이 있어 내일 갈 거야", 1),  # '있'은 지난 일이 아니다
])
def test_iss_gess_are_not_past_markers(transcript, pieces):
    assert len(reply_service._clause_tags(transcript)) == pieces


def test_still_past_question_becomes_template_in_pure_future():
    # P-4 전부 앞일인데 질문이 과거형으로 남으면 앞일 절 낱말의 틀 질문으로
    out = rewrite_tense("된장찌개를 맛있게 드셨군요. 간은 잘 맞았어요?", "오늘 저녁엔 된장찌개 끓일 거야")
    assert out == "그러시군요. 된장찌개 이야기 조금 더 들려주시겠어요?"


def test_word_in_both_past_and_future_clause_is_not_future_only():
    # P-1 지난 일 절에도 나온 낱말(병원)은 앞일 절에만 나온 낱말이 아니다
    assert not is_tense_error("병원 다녀오셨군요. 다음 주엔 몇 시에 가세요?", "어제 병원 다녀왔는데 다음 주에 또 병원 가야 해")


def test_compound_gatda_ol_is_one_future_clause():
    # P-2 "갔다 올 거야"를 자르지 않아야 앞일 낱말(병원)을 지킨다
    assert is_tense_error("병원에는 잘 다녀오셨나요? 결과는 괜찮으셨어요?", "내일 병원에 갔다 올 거야")


def test_preparation_question_with_elder_word_is_ok():
    # 나가기 전 준비를 묻는 말은 어르신 낱말이 있어도 맞다
    assert not is_tense_error("비 소식이 있네요. 장화는 챙기셨어요?", "모레 장화 신고 장 보러 가야지")


def test_time_word_rule_is_per_comma_part():
    # 쉼표 뒤 '지치셨어요'는 오늘 일 - 앞일 시간 낱말이 있는 부분만 본다
    assert not is_tense_error("오늘 고단하셨군요. 모레는 괜찮으실 거예요, 많이 지치셨어요?", "오늘은 고단했지만 모레는 괜찮을 거야")


def test_still_past_generic_ss_question_becomes_template():
    # P-4 '말랐나요'처럼 받침 ㅆ 일반형 과거 질문도 틀 질문으로
    out = rewrite_tense("이불까지 빨아 놓으셨다니 개운하시겠어요. 잘 말랐나요?", "이따 이불 빨 거야")
    assert out == "그러시군요. 이불 이야기 조금 더 들려주시겠어요?"



@pytest.mark.parametrize("body,transcript", [
    # 11/26 모델 재생성 측정: 부사는 틀 질문 낱말로 고르지 않는다 (품질 세트 문구와 다른 새 문장)
    ("남은 걸로 대강 차리셨군요. 특별히 좋아하시는 반찬이 있으신가요?", "남은 걸로 대강 차렸어"),
    ("천천히 걸으셨군요. 특별히 좋아하시는 길이 있으신가요?", "공원을 천천히 걸었어"),
    # 11/27 켠 상태 점검 N-3: 목록에 없는 '-히' 부사 (목록 덕에 통과하지 않게)
    ("넉넉히 해 두셨군요. 특별히 신경 쓰신 게 있으신가요?", "넉넉히 해 뒀어"),
])
def test_adverbs_are_not_topic_words(body, transcript):
    out = rewrite_vague(body, transcript)
    assert out is None or not any(w in out.split(". ")[-1] for w in ("대강", "천천히", "넉넉히"))


def test_switch_defaults_are_on(monkeypatch):
    # 11/27 사용자 결정: 시제 rewrite·막연한 질문 고쳐 쓰기 기본 켬 (환경변수가 없을 때)
    from app.core.config import _as_bool, _as_choice
    monkeypatch.delenv("COMPANION_REPLY_TENSE_FIX", raising=False)
    monkeypatch.delenv("COMPANION_REPLY_VAGUE_REWRITE", raising=False)
    assert _as_choice("COMPANION_REPLY_TENSE_FIX", "rewrite", ("off", "rewrite")) == "rewrite"
    assert _as_bool("COMPANION_REPLY_VAGUE_REWRITE", True) is True
    import inspect
    from app.core import config
    src = inspect.getsource(config)
    assert '_as_choice("COMPANION_REPLY_TENSE_FIX", "rewrite"' in src and '_as_bool("COMPANION_REPLY_VAGUE_REWRITE", True)' in src


# 11/27 켠 상태 점검 N-1: 연락만 하는 사람에게는 "~와는 뭐 하실 거예요?"를 쓰지 않는다
def test_person_template_skips_remote_contact():
    from app.services.companion.reply_service import _template_question
    assert _template_question("손자", True, "이따 손자가 영상통화 한대") == "손자 이야기 조금 더 들려주시겠어요?"
    assert _template_question("조카", True, "주말에 조카한테 전화하려고") == "조카 이야기 조금 더 들려주시겠어요?"
    assert _template_question("손자", True, "이따 손자가 온대") == "손자와는 뭐 하실 거예요?"


# 11/27 켠 상태 점검 N-2: 말로 줄인 '낼'(내일)도 앞일 시간 낱말이다 - 동사 '내다'는 아니다
def test_nael_is_tomorrow_only_as_time_word():
    from app.services.companion.reply_service import _nael_to_tomorrow as f
    assert f("낼 아침에 나가") == "내일 아침에 나가"
    assert f("낼은 쉬어") == "내일은 쉬어"
    assert f("그래서 낼 병원 가") == "그래서 내일 병원 가"
    assert f("아들이 낼 들른대") == "아들이 내일 들른대"
    for verb in ("이번 달 회비 낼 거야", "병원비 낼 때 됐어", "내가 낼게", "돈 낼 수 있어", "관리비 낼 날이 다가와", "밥값 낼 사람 정했어"):
        assert f(verb) == verb


def test_nael_future_clause_with_reason_link():
    elder = "고추가 떨어져서 낼 시장에 가서 두 근 사 와야지"
    assert is_tense_error("시장에서 고추를 사 오셨군요. 장 보시면서 즐거우셨나요?", elder)
    assert not is_tense_error("고추가 떨어지셨군요. 시장은 몇 시쯤 가실 거예요?", elder)


# 11/27 #359 점검 S-5: 앞 낱말이 조사로 끝나 앞 낱말 조건을 지나는 동사 '낼'은 뒤 낱말(거·게·수·때)로 가른다
@pytest.mark.parametrize("verb", ["돈은 내가 낼 거야", "밥값도 우리가 낼 수 있어", "회비는 다음에 낼 때 같이 줘"])
def test_nael_verb_after_particle_is_not_tomorrow(verb):
    from app.services.companion.reply_service import _nael_to_tomorrow
    assert _nael_to_tomorrow(verb) == verb


# 11/27 #359 점검 S-5: 연락 낱말마다 사람 틀 질문을 쓰지 않는다
@pytest.mark.parametrize("contact", ["전화", "통화", "문자", "연락", "카톡", "영상"])
def test_each_remote_contact_word_skips_person_template(contact):
    from app.services.companion.reply_service import _template_question
    assert _template_question("조카", True, f"이따 조카가 {contact} 준대") == "조카 이야기 조금 더 들려주시겠어요?"


# 11/27 #359 점검 S-1: 마지막 인사 검사도 '낼'을 앞일로 본다
def test_last_turn_check_sees_nael():
    from app.services.companion.reply_service import strict_failures
    body = "약국 다녀오셨군요. 오늘 하루 수고 많으셨어요. 편히 쉬세요."
    assert strict_failures(body, True, "낼 약국 가") == ["앞일을 한 일로"]
    assert strict_failures(body, True, "관리비 낼 날이야") == []


# 11/27 #359 점검 S-2: 할 일을 말한 어르신에게 시간 낱말 틀 질문("내일에는 뭐 하실 거예요?")을 쓰지 않는다
def test_time_word_template_skipped_when_plan_stated():
    # 대답에 시간 낱말만 있으면 그것밖에 못 고른다 - 어르신이 할 일을 말했으므로 고쳐 쓰지 않는다
    body = "내일 바쁘시겠네요. 특별히 하실 일이 있으신가요?"
    assert rewrite_vague(body, "낼 은행 가서 통장 정리해야 돼") is None
    assert rewrite_vague(body, "내일 은행 가서 통장 정리해야 돼") is None
    # 할 일을 말하지 않았으면 시간 틀 질문을 그대로 쓴다
    assert rewrite_vague("주말엔 푹 쉬시는군요. 특별히 하실 일이 있으신가요?", "주말엔 그냥 쉬어야지") == "주말엔 푹 쉬시는군요. 주말에는 뭐 하실 거예요?"


# 11/27 #359 점검 S-4: '아들 내외'·'딸네 부부'는 한 낱말로 쓴다
def test_person_tail_is_joined():
    body = "아들 내외분이 오시면 반찬 준비하시느라 바쁘시겠네요. 특별히 좋아하시는 반찬이 있으신가요?"
    out = rewrite_vague(body, "낼모레 주말엔 아들 내외가 온다더라")
    assert out is not None and out.endswith("아들 내외와는 뭐 하실 거예요?")
