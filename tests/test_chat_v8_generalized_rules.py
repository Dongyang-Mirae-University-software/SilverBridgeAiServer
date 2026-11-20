"""시험지 v8 holdout-A 1차 측정(2026-11-21) 미달을 일반 규칙으로 고친 것 - 걸려야 하는 말과 걸리면 안 되는 말.

시험지 문장을 그대로 넣지 않고 같은 유형의 다른 문장으로 고정한다(과적합 방지). locked 는 보지 않았다.
"""
from __future__ import annotations

import pytest

from app.services import chat_keywords as ck
from app.services.medical_llm_service import MedicalLlmService as M
from tools.eval.replay_offline import fixed_sentence_in_reply


# ---- E: 라우터 실패 때 응급 판정 ------------------------------------------------------------------------------


@pytest.mark.parametrize("text,name", [
    ("가슴이 찢어질 듯 아파요", "흉통:비유"),
    ("아버지가 가슴이 불에 타는 것처럼 아프대요", "흉통:비유"),
    ("가슴이 터질듯이 아퍼", "흉통:비유"),
    ("갑자기 머리가 터질 듯이 아파요", "뇌출혈:두통"),
    ("살면서 이렇게 머리가 아픈 건 처음이야", "뇌출혈:두통"),
    ("이런 두통은 난생 처음이에요", "뇌출혈:두통"),
    ("아내가 갑자기 말을 못 해요", "뇌졸중:실어"),
    ("할머니가 갑자기 말이 안 나온대", "뇌졸중:실어"),
])
def test_new_fallback_patterns_catch_the_type(text, name) -> None:
    assert name in ck.fallback_emergency_signals(text)


@pytest.mark.parametrize("text", [
    "운동한 뒤로 가슴 근육을 누르면 아파요",      # 근육통
    "가슴이 쓰리고 신물이 올라와요",              # 속쓰림
    "가슴이 아플 것 같아서 걱정이에요",           # 걱정 꼴
    "술을 많이 마셔서 머리가 깨질 것 같아요",      # 숙취(갑자기 없음)
    "머리가 지끈거리는데 어제 잠을 못 자서 그래요",
    "두통약은 처음 먹어 봐요",
    "목이 쉬어서 말을 못 해요",                   # 갑자기 없음
    "손주가 처음으로 말을 했어요",
])
def test_new_fallback_patterns_do_not_catch_ordinary_sentences(text) -> None:
    hits = set(ck.fallback_emergency_signals(text))
    assert not hits & {"흉통:비유", "뇌출혈:두통", "뇌졸중:실어"}, (text, hits)


# ---- G': 약 + 술 규칙의 약 이름·술 이름 ---------------------------------------------------------------------------


@pytest.mark.parametrize("text,rule", [
    ("글루코파지 먹는데 소주 마셔도 돼요?", "diabetes"),
    ("자누비아 먹는 중에 맥주 한 캔 괜찮아요?", "diabetes"),
    ("엘리퀴스 먹는 중인데 와인 괜찮아요?", "anticoagulant"),
    ("아버지가 노바스크 드시는데 저녁마다 반주하세요", "blood_pressure"),
    ("아티반 먹고 위스키 한 잔 해도 돼?", "sedative"),
])
def test_drug_families_and_alcohol_words(text, rule) -> None:
    assert rule in [r.name for r in M.drug_alcohol_rules(text)]


@pytest.mark.parametrize("text", [
    "피아노 반주를 배우는데 손목이 아파요",   # 약 이름 없음
    "메트포르민은 아침에 먹어요",              # 술 없음
    "수술 뒤에 엘리퀴스 다시 먹어도 돼요?",    # "수술"의 술은 술이 아니다
])
def test_drug_alcohol_rules_do_not_fire_without_both(text) -> None:
    assert M.drug_alcohol_rules(text) == []


# ---- I: 저림 한 가지 호소는 되묻는다 ------------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["손이 저려요", "팔이 저려", "발이저려요"])
def test_limb_numbness_alone_gets_followup(text) -> None:
    assert M.needs_followup_question(text)


@pytest.mark.parametrize("text", [
    "다리가 저리고 어지러워요",     # 서로 다른 증상 둘
    "어제부터 발이 저려요",          # 기간이 있다(충분한 정보)
    "한쪽 다리가 저려요",           # 응급 키워드(뇌졸중 의심) - 되묻지 않고 응급
])
def test_numbness_boundaries_kept(text) -> None:
    assert not M.needs_followup_question(text)


# ---- 채점기: 답 모양이 나눈 고정 문장도 찾는다 ----------------------------------------------------------------------


def test_fixed_sentence_found_after_layout_split() -> None:
    rule = next(r for r in M.drug_alcohol_rules("당뇨약 먹는데 술 마셔도 돼요?"))
    head, tail = str(rule.sentence).split(". ", 1)
    laid_out = f"💊 {head}.\n✅  {tail}"
    assert fixed_sentence_in_reply(rule.sentence, laid_out)
    assert not fixed_sentence_in_reply(rule.sentence, f"💊 {head}.")  # 한 조각만 있으면 아니다
    assert fixed_sentence_in_reply(rule.sentence, rule.sentence.proxy)
