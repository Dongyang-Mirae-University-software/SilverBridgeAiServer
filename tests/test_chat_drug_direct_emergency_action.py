"""약 질문 직접 안내(F-1)·응급 행동 한 줄(F-2) (2026-10-29).
F-1: 약 질문 유형별로 핵심이 없으면 고정 문장 / 응급·약 + 술·과다 복용 질문은 대상 밖.
F-2: 응급 답의 119 안내는 그대로 두고 그 뒤에 유형별 행동 한 줄 / 경고 박스·자해 고정 응답은 그대로."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.config import Settings
from app.services.medical_llm_service import MedicalLlmService

PLAIN = "약을 더 드시거나 바꾸기 전에 꼭 의사나 약사에게 물어봐 주세요."
DOUBLE = "놓친 약을 다음에 두 번 몫으로 한꺼번에 드시지는 마세요."
PAINKILLER = "진통제는 정해진 양과 간격을 넘겨 드시지 마세요."
ANTIBIOTIC = "남은 항생제를 임의로 드시지 마세요. 열이 나면 진료를 받아 보세요."
BP = "혈압약을 임의로 끊으면 혈압이 다시 오를 수 있어요."
CALL_119 = "지금은 병원 예약보다 119 연락이 먼저예요. 바로 119에 전화해 주세요."
DRIVE = "직접 운전하거나 혼자 움직이지 마세요."
SEIZURE = "억지로 붙잡거나 입에 물건을 넣지 마세요."
COLLAPSE = "무리하게 일으키지 말고 편한 자세로 눕혀 두세요."
SUGAR = "의식이 있으면 사탕이나 단 음료를 드시고, 의식이 흐리면 억지로 먹이지 마세요."
ACTIONS = (DRIVE, SEIZURE, COLLAPSE, SUGAR)


class _Loader:
    def __init__(self, **fields: str) -> None:
        self.fields = {"empathy": "걱정되셨겠어요.", "symptoms": "약 복용 문의", "features": "", "advice": PLAIN, "reply": "", **fields}
        self.state = SimpleNamespace(model_name="fake-medgemma")

    def ensure_loaded(self) -> None:
        return None

    def generate_structured_text(self, system_prompt: str, user_prompt: str, prefill: str = "", **kwargs) -> str:  # noqa: ARG002
        return json.dumps(self.fields, ensure_ascii=False)


def _gen(message: str, intent: str = "medical_advice", **fields: str) -> dict[str, Any]:
    return MedicalLlmService(Settings(), _Loader(**fields)).generate(message, {}, [], intent=intent)


# ---- F-1 약 질문 직접 안내 ------------------------------------------------------------------------------


@pytest.mark.parametrize("message, sentence", [
    ("혈압약을 깜빡하고 안 먹었어요", DOUBLE),
    ("약 먹는 걸 자꾸 잊어서 하루치를 한 번에 몰아서 먹어도 되나요", DOUBLE),
    ("혈압약을 아침에 깜빡하고 안 먹었는데 저녁에 두 알 먹어도 돼?", DOUBLE),
    ("진통제는 아프면 아무 때나 많이 먹어도 되죠?", PAINKILLER),
    ("무릎이 너무 아파서 진통제를 하루에 몇 알까지 먹어도 돼?", PAINKILLER),
    ("열이 나는데 병원 가지 말고 남은 항생제를 먹어도 되죠?", ANTIBIOTIC),
    ("혈압이 정상으로 나왔으니 약을 끊어도 되죠?", BP),
    ("혈압약 이제 끊어도 될까요?", BP),
])
def test_drug_question_gets_direct_sentence_once(message: str, sentence: str) -> None:
    out = _gen(message)
    assert out["reply"].count(sentence) == 1


def test_antibiotic_sentence_replaces_generic_clinic_line() -> None:
    # F-1 문장에 "진료"가 있어 C-2 일반 권유가 한 번 더 붙지 않는다.
    out = _gen("열이 나는데 병원 가지 말고 남은 항생제를 먹어도 되죠?")
    assert "증상이 계속되거나 걱정되시면 병원 진료를 받아 보세요." not in out["reply"]


def test_bp_stop_keeps_consult_sentence() -> None:
    out = _gen("혈압이 정상으로 나왔으니 약을 끊어도 되죠?")
    assert "약을 줄이거나 끊거나 거르는 것은 꼭 의사·약사와 상의한 뒤에 하세요." in out["reply"]


def test_bp_sentence_skipped_when_answer_says_bp_rises() -> None:
    out = _gen("혈압약 이제 끊어도 될까요?", advice="임의로 끊으면 혈압이 다시 오를 수 있으니 의사와 상의해 주세요.")
    assert BP not in out["reply"]


def test_skipped_when_answer_already_says_it() -> None:
    out = _gen("하루치를 몰아서 먹어도 되나요?", advice="두 번 몫을 한꺼번에 드시지 마세요. 의사나 약사에게 물어봐 주세요.")
    assert DOUBLE not in out["reply"]


@pytest.mark.parametrize("message", ["혈압약을 두 번 먹었는데 괜찮을까요?", "약을 깜빡해서 두 알 더 먹었어요"])
def test_overdose_told_gets_no_double_dose_sentence(message: str) -> None:
    assert DOUBLE not in _gen(message)["reply"]


def test_drug_alcohol_question_untouched() -> None:
    out = _gen("혈압약 깜빡했는데 술 마셔도 돼요?")
    assert DOUBLE not in out["reply"]
    assert "혈압약을 드시는 중에 술을 마시면" in out["reply"]  # 약 + 술 규칙은 그대로


@pytest.mark.parametrize("message", ["감기약이랑 혈압약을 같이 먹어도 되나요", "아스피린을 매일 먹어도 되나요", "약속을 깜빡했어요"])
def test_other_questions_get_no_direct_sentence(message: str) -> None:
    reply = _gen(message)["reply"]
    assert not any(s in reply for s in (DOUBLE, PAINKILLER, ANTIBIOTIC, BP))


def test_no_drug_direct_on_emergency() -> None:
    out = _gen("약을 한꺼번에 많이 먹었어요")  # 응급 키워드(과다 복용)
    assert out["emergencyWarning"]
    assert not any(s in out["reply"] for s in (DOUBLE, PAINKILLER, ANTIBIOTIC, BP))


# ---- F-2 응급 행동 한 줄 --------------------------------------------------------------------------------


@pytest.mark.parametrize("message, intent, action", [
    ("가슴이 조이고 식은땀이 나요", "medical_advice", DRIVE),
    ("갑자기 말이 어눌해지고 입이 한쪽으로 돌아가요", "emergency_guidance", DRIVE),
    ("몸에서 경련이 계속 나요", "medical_advice", SEIZURE),
    ("할머니가 쓰러지셨어요", "medical_advice", COLLAPSE),
    ("넘어졌는데 허벅지가 너무 아파서 일어나지 못하겠어요", "emergency_guidance", COLLAPSE),
    ("혈당이 40밖에 안 나오고 몸이 떨려요", "emergency_guidance", SUGAR),
    ("저혈당 온 것 같아요, 식은땀이 나요", "emergency_guidance", SUGAR),
])
def test_emergency_gets_action_after_119(message: str, intent: str, action: str) -> None:
    out = _gen(message, intent=intent, symptoms="응급 증상", advice="x")
    lines = out["reply"].split("\n")
    assert CALL_119 in lines and action in lines
    assert lines.index(CALL_119) < lines.index(action)  # 119 안내가 먼저
    assert out["emergencyWarning"] == [MedicalLlmService.EMERGENCY_WARNING]  # 경고 박스 그대로
    assert out["riskLevel"] == "high"


@pytest.mark.parametrize("message", ["혈당이 300이에요 너무 높아요", "숨을 못 쉬겠어요", "피를 토했어요"])
def test_no_action_or_no_sugar_when_not_matching(message: str) -> None:
    out = _gen(message, intent="emergency_guidance", symptoms="응급 증상", advice="x")
    assert CALL_119 in out["reply"]
    assert SUGAR not in out["reply"]
    if "혈당" not in message:
        assert not any(a in out["reply"] for a in ACTIONS)


def test_no_action_on_non_emergency_answer() -> None:
    out = _gen("혈당이 50 정도 나와요", symptoms="혈당 문의", advice="내과 진료를 권해드려요.")
    assert not out["emergencyWarning"]
    assert not any(a in out["reply"] for a in ACTIONS)


def test_self_harm_fixed_reply_untouched() -> None:
    reply = MedicalLlmService.self_harm_response()["reply"]
    assert not any(a in reply for a in ACTIONS)


def test_safety_log_has_rule_names_only(caplog) -> None:
    import logging
    caplog.set_level(logging.INFO)
    _gen("몸에서 경련이 계속 나요", symptoms="경련", advice="x")
    _gen("혈압약을 깜빡하고 안 먹었어요")
    msgs = [r.getMessage() for r in caplog.records if "[CHAT-SAFETY]" in r.getMessage()]
    assert "[CHAT-SAFETY] emergency_action=seizure" in msgs
    assert "[CHAT-SAFETY] drug_direct=missed_dose_timing,double_dose" in msgs
