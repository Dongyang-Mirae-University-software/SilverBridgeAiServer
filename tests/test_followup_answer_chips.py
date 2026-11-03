"""되묻기 답 칩(2026-11-02): 짧은 증상 한마디의 되묻기 턴에만 눌러서 답하는 고정 칩 4개를 준다.
판정(응급·자해·위험 신호·약 + 술·예약·진료과)은 바꾸지 않고, 칩을 눌러 보낸 문장도 되묻기 다음 턴 이어 붙이기를 그대로 탄다."""
from __future__ import annotations

import json
import logging
import re

import pytest

from app.models.chat_log import ChatLog
from app.services import chat_keywords as kw
from app.services import follow_up_questions as fq
from app.services import followup_answer_chips as fac
from app.services.chat_service import _FOLLOWUP_CLOSE_WORDS
from app.services.medical_llm_service import MedicalLlmService as M
from tests.test_chat_followup_question import INTRO, _send, env  # noqa: F401 - 같은 대화 흐름 하네스

WARD_PROMPTS = ["머리가 아파요", "배가 아파요", "무릎이 아파요", "어지러워요"]  # FE 피보호자 기본 칩


# ---- 표 조건 ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("chip", fac.all_chips())
def test_every_chip_is_short_plain_and_never_changes_judgment(chip: str) -> None:
    assert len(chip) <= fq.MAX_CHARS
    assert fq.sanitize([chip]) == [chip]  # 이어서 물어볼 질문과 같은 정리 기준(숫자 덩어리·제어문자 없음)
    assert not M.emergency_signals(chip) and not M.self_harm_signals(chip)
    assert not kw.has_red_flag(chip)  # 위험 신호 칩은 두지 않는다
    assert not M.drug_alcohol_rules(chip)
    assert not any(k in chip for k in kw.RESERVATION_KEYWORDS)
    # 종료어("됐어요"·"괜찮아요" 등)가 있으면 다음 턴에 앞 질문과 이어 붙이지 않는다 - "며칠 됐어요"를 쓰지 않은 이유.
    assert not _FOLLOWUP_CLOSE_WORDS.search(chip)


def test_each_group_has_four_distinct_self_and_proxy_chips() -> None:
    for name, _, self_chips, proxy_chips in fac._GROUPS:
        assert len(self_chips) == 4 and len(set(self_chips)) == 4, name
        assert len(proxy_chips) == 4 and len(set(proxy_chips)) == 4, name
        assert all(re.search(r"대요$", c) for c in proxy_chips), name  # 대리 상담 말투(전해 들은 말)
        assert not any(c.endswith("대요") for c in self_chips), name


@pytest.mark.parametrize(("message", "group"), [
    ("머리가 아파요", "head"), ("두통이 있어요", "head"), ("배가 아파요", "belly"), ("속이 안 좋아요", "belly"),
    ("무릎이 아파요", "joint"), ("허리가 아파요", "joint"), ("손목이 아파요", "joint"), ("어지러워요", "dizzy"),
    ("머리가 어지러워요", "dizzy"), ("손이 저려요", "numb"), ("기침이 나요", "cough"), ("열이 나요", "cough"),
    ("발열이 있어요", "cough"), ("귀가 아파요", "default"), ("피부가 가려워요", "default"),
    # 2026-11-03 점검 L-1: 손·발 통증·손발가락·목 뻐근함은 관절, 목 통증은 기침·열, 손 떨림·발 부음은 기본
    ("손이 아파요", "joint"), ("발이 아파요", "joint"), ("발가락이 아파요", "joint"), ("목이 뻐근해요", "joint"),
    ("목이 결려요", "joint"), ("목이 아파요", "cough"), ("손이 떨려요", "default"), ("발이 부었어요", "default"),
])
def test_group_matching(message: str, group: str) -> None:
    assert fac.group_name(message) == group


@pytest.mark.parametrize("message", ["어지러워", "기침이 나요", "머리가 아파요", "배가 아파요", "허리가 아파요", "속이 안 좋아요",
                                     "귀가 아파요", "피부가 가려워요", "가려워요", *WARD_PROMPTS])
def test_every_followup_target_gets_four_chips(message: str) -> None:
    assert M.needs_followup_question(message)
    assert len(fac.answer_chips(M.followup_question_response(message), message)) == 4


# ---- 대화 흐름 -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("message", WARD_PROMPTS)
def test_followup_turn_gives_answer_chips(env, message: str) -> None:  # noqa: F811
    out = _send(env, message)
    assert INTRO.split(". ")[0] in out["reply"]
    assert out["suggestedKind"] == "answer"
    assert out["suggestedQuestions"] == fac.chips_for(message, proxy=False)
    assert env.llm.calls == []  # 모델을 부르지 않는다


def test_proxy_followup_gets_proxy_chips(env) -> None:  # noqa: F811
    out = _send(env, "기침이 나요", ward="ab12")
    assert out["suggestedQuestions"] == fac.chips_for("기침이 나요", proxy=True)
    out = _send(env, "엄마가 어지러워해요", session="s2")
    assert out["suggestedQuestions"] == fac.chips_for("어지러워", proxy=True)


@pytest.mark.parametrize("message", ["숨이 좀..", "가슴이 좀 답답", "숨차네"])
def test_red_flag_followup_has_no_chips(env, message: str) -> None:  # noqa: F811
    out = _send(env, message)
    assert "119" in out["reply"]
    assert out["suggestedQuestions"] == [] and "suggestedKind" not in out


@pytest.mark.parametrize("message", ["가슴이 아파요", "죽고 싶어요", "머리가 너무 아파요", "혈압약 먹었어요"])
def test_non_followup_turns_have_no_answer_chips(env, message: str) -> None:  # noqa: F811
    out = _send(env, message)
    assert "suggestedKind" not in out
    assert not set(out["suggestedQuestions"]) & set(fac.all_chips())


def test_router_emergency_has_no_chips(env) -> None:  # noqa: F811
    env.router.emergency = True
    out = _send(env, "머리가 아파요")
    assert out["suggestedQuestions"] == [] and "suggestedKind" not in out


@pytest.mark.parametrize("prompt", WARD_PROMPTS)
def test_each_chip_continues_previous_question(env, prompt: str) -> None:  # noqa: F811
    for i, chip in enumerate(fac.chips_for(prompt, proxy=False)):
        session = f"s-{i}"
        first = _send(env, prompt, session=session)
        assert chip in first["suggestedQuestions"]
        env.router.route_value = "general"  # 짧은 칩 문장을 라우터가 일반 대화로 보내도
        out = _send(env, chip, session=session)
        env.router.route_value = "symptom"
        call = env.llm.calls[-1]
        assert call["message"] == f"{prompt} / {chip}"  # 앞 질문과 이어 상담
        assert call["intent"] == "medical_advice"
        assert INTRO.split(". ")[0] not in out["reply"]  # 되묻기를 또 하지 않는다
        assert "suggestedKind" not in out  # 상담 답 턴에는 답 칩이 없다
    assert len(env.llm.calls) == 4


@pytest.mark.parametrize("prompt", WARD_PROMPTS)
def test_chip_does_not_change_department(prompt: str) -> None:
    base = kw.symptom_department(prompt)
    for chip in fac.chips_for(prompt, proxy=False) + fac.chips_for(prompt, proxy=True):
        joined = kw.symptom_department(f"{prompt} / {chip}")
        # 열이 함께 있는 두통·어지럼은 내과 - 기존 규칙(I-2) 그대로, 그 밖에는 앞 질문의 진료과
        expected = "내과" if base == "신경과" and kw.FEVER.search(chip) else base
        assert joined == expected, chip


def test_chips_are_not_saved_and_log_has_count_only(env, caplog) -> None:  # noqa: F811
    with caplog.at_level(logging.INFO):
        out = _send(env, "머리가 아파요")
    row = env.db.query(ChatLog).order_by(ChatLog.id.desc()).first()
    stored = json.dumps({"ctx": row.context_json, "reply": row.reply}, ensure_ascii=False)
    assert not any(chip in stored for chip in out["suggestedQuestions"])
    assert "suggestedKind" not in stored
    lines = [r.getMessage() for r in caplog.records if "source=answer" in r.getMessage()]
    assert lines == ["[CHAT-SUGGEST] source=answer count=4"]


def test_each_guard_blocks_on_its_own() -> None:
    base = M.followup_question_response("머리가 아파요")
    assert fac.answer_chips(base, "머리가 아파요")
    flagged = {**base, "meta": {**base["meta"], "redFlag119": True}}  # 119 문장이 없어도 표시만으로 막는다
    assert fac.answer_chips(flagged, "머리가 아파요") == []
    with_119 = {**base, "reply": base["reply"] + "\n갑자기 심해지면 119에 연락하세요."}  # 표시가 없어도 답의 119로 막는다
    assert fac.answer_chips(with_119, "머리가 아파요") == []
    assert fac.answer_chips({**base, "riskLevel": "high"}, "머리가 아파요") == []
    assert fac.answer_chips({**base, "emergencyWarning": ["x"]}, "머리가 아파요") == []
    assert fac.answer_chips({**base, "meta": {}}, "머리가 아파요") == []


# 2026-11-03 점검 M-2: 증상 낱말("속·열·배")이 들어 있어도 증상이 아닌 말은 되묻지 않는다 - 엉뚱한 답 칩이 붙지 않게.
@pytest.mark.parametrize("message", ["속상해요", "너무 속상해", "열받아요", "열이 받쳐요", "배고파요", "배가 고파요", "배불러요", "배가 불러요"])
def test_not_symptom_words_are_not_followup(message: str) -> None:
    assert not M.needs_followup_question(message)


@pytest.mark.parametrize("message", ["배가 아파요", "속이 쓰려요", "열이 나요", "배아퍼"])
def test_real_symptoms_still_followup(message: str) -> None:
    assert M.needs_followup_question(message)


def test_no_answer_chips_right_after_emergency_reply(env) -> None:  # noqa: F811
    env.router.emergency = True
    _send(env, "갑자기 머리가 깨질 듯 아파요")
    env.router.emergency = False
    out = _send(env, "머리가 아파요")  # 응급 답 다음 턴은 되묻지 않으므로 답 칩도 없다
    assert "suggestedKind" not in out
    assert not set(out["suggestedQuestions"]) & set(fac.all_chips())
