"""되묻기 도중 새 증상·위험 신호(2026-11-11).

피보호자 단계 되묻기 도중 지금 질문과 무관한 다른 묶음 증상("무릎" 단계 중 "기침이 나요")은 앞 증상에 이어 붙이지 않고
새 증상으로 처음부터(그 묶음의 위험 신호 질문부터) 묻는다. 진행 중 답에 위험 신호 낱말이 부정 없이 들면 묶음·순번과 상관없이
그 자리에서 119. 같은 묶음 증상·지금 질문이 묻는 증상·답 꼴로 시작하는 말은 지금 흐름 그대로.
보호자 옛 되묻기는 질문이 "같이 나타나는 증상이 있나요?"라 다른 증상은 그 답이다 - 앞 증상이 나아졌다는 전환 문장만 새 증상으로.
"""
from __future__ import annotations

import logging

import pytest

from app.services import followup_steps as fs
from tests.test_followup_steps import _send, env, flow  # noqa: F401

START = {"joint": "무릎이 아파요", "head": "머리가 아파요", "belly": "배가 아파요", "cough": "기침이 나요",
         "dizzy": "어지러워요", "default": "피곤해요"}
ADVANCE = ["아니요", "오늘부터요"]  # 1번 → 2번 → 3번 질문으로


def _at(group: str, index: int = 0) -> dict:
    state = fs.new_state(START[group], False)
    assert state["group"] == group
    return {**state, "index": index, "answers": ["x"] * index, "turns": index + 1}


def _walk(env, group: str, depth: int, session: str) -> None:
    _send(env, START[group], session=session)
    for answer in ADVANCE[:depth]:
        _send(env, answer, session=session)


# ---- 판별 표 ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("group,index,text,expected", [
    ("joint", 0, "기침이 나요", "기침이 나요"),
    ("joint", 1, "배도 아파요", "배도 아파요"),
    ("joint", 2, "그리고 어지러워요", "그리고 어지러워요"),
    ("head", 0, "배도 아파요", "배도 아파요"),
    ("belly", 1, "허리가 아파요", "허리가 아파요"),
    ("default", 0, "기침이 나요", "기침이 나요"),
    ("joint", 0, "아까 무릎은 괜찮아졌고 기침이 나요", "기침이 나요"),  # 전환 - 뒤 증상 부분
    ("joint", 1, "무릎은 이제 안 아프고 머리가 아파요", "머리가 아파요"),
])
def test_new_symptom_is_detected(group: str, index: int, text: str, expected: str) -> None:
    assert fs.switch_text(_at(group, index), text) == expected


@pytest.mark.parametrize("group,index,text", [
    ("head", 2, "열이 나요"),  # 머리 묶음 3번 질문("열도 나세요?")이 묻는 증상
    ("head", 0, "열이 나요"),
    ("dizzy", 2, "속도 메스꺼워요"),  # 어지럼 묶음 3번 질문
    ("joint", 0, "허리가 아파요"),  # 같은 묶음
    ("belly", 1, "배도 아파요"),
    ("belly", 0, "설사를 해요"),
    ("joint", 0, "붓기도 있어요"),  # 어느 묶음 낱말도 아님 - 지금 흐름
    ("dizzy", 0, "어제부터요, 열도 조금 있어요"),  # 기간 말로 시작 - 앞 증상의 설명
    ("joint", 0, "아니요, 기침이 나요"),  # 답 꼴로 시작
    ("joint", 0, "네"),
    ("joint", 1, "며칠 됐어요"),
    ("joint", 0, "무릎은 괜찮아요"),  # 전환 아님(뒤가 없다)
    ("joint", 0, "가슴이 답답해요"),  # 위험 신호는 새 증상이 아니라 그 자리 119
    ("joint", 0, "모르겠어요"),
])
def test_answer_or_same_group_keeps_flow(group: str, index: int, text: str) -> None:
    assert fs.switch_text(_at(group, index), text) is None


def test_red_flag_flow_is_never_switched() -> None:
    # 위험 신호 낱말로 시작한 흐름(chest)은 위험 신호 질문을 잃지 않게 넘기지 않는다.
    state = fs.new_state("숨이 좀..", False)
    assert state["group"] == "chest"
    assert fs.switch_text(state, "배가 아파요") is None


@pytest.mark.parametrize("text,expected", [
    ("가슴이 답답해요", True), ("숨이 차요", True), ("가슴이 조여요", True),
    ("가슴은 괜찮아요", False), ("숨이 안 차요", False), ("기침이 나요", False), ("아니요", False),
])
def test_red_flag_answer(text: str, expected: bool) -> None:
    assert fs.red_flag_answer(_at("joint", 1), text) is expected


@pytest.mark.parametrize("text", ["배도 아파요", "열이 나요", "기침이 나요", "설사를 해요"])
def test_onset_question_symptom_answer_is_not_asked_again(text: str) -> None:
    # 기간 질문에 증상 말로 답하면 "'오늘부터'처럼 알려 주세요"를 되풀이하지 않는다.
    verdict, _ = fs.interpret(_at("belly", 1), text)
    assert verdict == "describe"


def test_onset_question_unclear_is_still_asked_again() -> None:
    assert fs.interpret(_at("belly", 1), "음")[0] == "unclear"


# ---- 대화 흐름 ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_new_symptom_restarts_steps_for_that_symptom(flow, depth: int) -> None:
    session = f"knee{depth}"
    _walk(flow, "joint", depth, session)
    calls = len(flow.llm.calls)
    out = _send(flow, "기침이 나요", session=session)
    assert len(flow.llm.calls) == calls  # 질문 턴 - MedGemma 를 부르지 않는다
    assert out["layout"]["questions"] == ["숨쉬기 힘드세요?"]  # 기침 묶음 위험 신호 질문부터
    assert out["suggestedQuestions"] == ["네", "아니요"]
    # 새 흐름을 끝까지 - 결과 문장은 기침으로 시작하고 무릎을 섞지 않는다.
    _send(flow, "아니요", session=session)
    _send(flow, "오늘부터요", session=session)
    _send(flow, "아니요", session=session)
    message = flow.llm.calls[-1]["message"]
    assert message.startswith("기침이 나요")
    assert "무릎" not in message


def test_relief_switch_result_uses_new_symptom_only(flow) -> None:
    _send(flow, "무릎이 아파요", session="r")
    out = _send(flow, "아까 무릎은 괜찮아졌고 기침이 나요", session="r")
    assert out["layout"]["questions"] == ["숨쉬기 힘드세요?"]
    for answer in ("아니요", "오늘부터요", "아니요"):
        _send(flow, answer, session="r")
    assert flow.llm.calls[-1]["message"] == "기침이 나요 / 숨쉬기는 괜찮아요 / 오늘부터요 / 열은 없어요"


def test_new_symptom_with_detail_goes_straight_to_answer(flow) -> None:
    # 이미 기간까지 말한 새 증상("어제부터"로 시작하면 답 꼴이라 넘기지 않는다 - 그 밖의 자세한 말)은 그 문장만으로 답한다.
    _send(flow, "무릎이 아파요", session="d")
    _send(flow, "기침이 나고 목도 아파요", session="d")
    assert flow.llm.calls and "무릎" not in flow.llm.calls[-1]["message"]


@pytest.mark.parametrize("group", ["joint", "head", "belly", "cough", "dizzy", "default"])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_red_flag_answer_mid_flow_is_119_right_away(flow, group: str, depth: int) -> None:
    session = f"rf{group}{depth}"
    _walk(flow, group, depth, session)
    calls = len(flow.llm.calls)
    out = _send(flow, "가슴이 답답해요", session=session)
    assert out["riskLevel"] == "high"
    assert "119" in out["reply"]
    assert len(flow.llm.calls) == calls


def test_negated_red_flag_answer_is_not_119(flow) -> None:
    _send(flow, "기침이 나요", session="n")
    out = _send(flow, "가슴은 괜찮아요", session="n")
    assert out["riskLevel"] != "high"


def test_same_group_answer_keeps_flow(flow) -> None:
    _send(flow, "머리가 아파요", session="h")
    _send(flow, "아니요", session="h")
    _send(flow, "오늘부터요", session="h")
    _send(flow, "열이 나요", session="h")  # 3번 질문 "열도 나세요?"의 답
    assert flow.llm.calls[-1]["message"].startswith("머리가 아파요 / ")


def test_self_harm_mid_flow_still_crisis_first(flow) -> None:
    _send(flow, "무릎이 아파요", session="s")
    out = _send(flow, "죽고 싶어요", session="s")
    assert out["riskLevel"] == "high"
    assert "109" in out["reply"]


def test_switch_log_has_no_message_text(flow, caplog) -> None:
    caplog.set_level(logging.INFO)
    _send(flow, "무릎이 아파요", session="l")
    _send(flow, "기침이 나요", session="l")
    lines = [r.getMessage() for r in caplog.records if "followup_new_symptom" in r.getMessage()]
    assert lines and all("기침" not in line for line in lines)


# ---- 보호자 옛 되묻기 ---------------------------------------------------------------------------------


def _guardian(env, message: str, session: str = "g") -> dict:
    return _send(env, message, session=session, ward="ab12", role="GUARDIAN")


def test_guardian_other_symptom_is_answer_to_companion_question(flow) -> None:
    # 옛 질문이 "같이 나타나는 증상(열·통증 등)이 있나요?"라 다른 증상은 그 답 - 지금처럼 이어 붙인다.
    _guardian(flow, "무릎이 아프다고 하세요")
    _guardian(flow, "기침도 하세요")
    assert flow.llm.calls[-1]["message"] == "무릎이 아프다고 하세요 / 기침도 하세요"


def test_guardian_relief_switch_is_new_symptom(flow) -> None:
    _guardian(flow, "무릎이 아프다고 하세요", session="g2")
    out = _guardian(flow, "무릎은 괜찮아졌고 기침이 나요", session="g2")
    assert not flow.llm.calls  # 이어 붙여 생성하지 않고 새 증상으로 한 번 되묻는다
    assert "내과" in out["reply"]
    assert "정형외과" not in out["reply"]


# ---- 기능 점검(B) 반영 (2026-11-11) ---------------------------------------------------------------------


@pytest.mark.parametrize("text", ["숨이 차고 안 나아요", "가슴이 답답한데 괜찮겠죠?", "가슴이 답답하고 안 나아져요", "숨이 차는 게 멈추지 않아요"])
def test_persisting_red_flag_is_119_right_away(flow, text: str) -> None:
    # F-1: 증상이 계속된다는 말("안 나아요", "괜찮겠죠?")은 위험 신호를 부정하지 않는다.
    session = f"p{abs(hash(text))}"
    _walk(flow, "joint", 1, session)
    calls = len(flow.llm.calls)
    out = _send(flow, text, session=session)
    assert out["riskLevel"] == "high"
    assert len(flow.llm.calls) == calls


@pytest.mark.parametrize("text", ["가슴은 괜찮아요", "숨이 안 차요", "가슴이 답답하지는 않아요"])
def test_real_negation_still_not_119(text: str) -> None:
    assert fs.red_flag_answer(_at("joint", 1), text) is False


def test_guardian_answer_after_relief_switch_uses_new_symptom(flow) -> None:
    # F-2: 보호자 전환 뒤 다음 턴도 전환 문장 원문(무릎)을 붙이지 않는다.
    _guardian(flow, "무릎이 아프다고 하세요", session="g3")
    _guardian(flow, "무릎은 괜찮아졌고 기침이 나요", session="g3")
    _guardian(flow, "며칠 되셨대요", session="g3")
    assert flow.llm.calls[-1]["message"] == "기침이 나요 / 며칠 되셨대요"


@pytest.mark.parametrize("text", ["손가락에서 피가 조금 나요", "무릎에 출혈이 있어요"])
def test_bleeding_answer_is_not_immediate_119(flow, text: str) -> None:
    # F-3(사용자 결정 A): 출혈 낱말만 든 진행 중 답은 즉시 119가 아니라 설명으로 결과(생성 후처리가 119 기준 한 줄을 붙인다).
    assert fs.red_flag_answer(_at("joint", 1), text) is False
    session = f"b{abs(hash(text))}"
    _walk(flow, "joint", 1, session)
    out = _send(flow, text, session=session)
    assert out["riskLevel"] != "high"
    assert text in flow.llm.calls[-1]["message"]


def test_chest_injury_answer_stays_119(flow) -> None:
    # 가슴 부딪힘은 어르신 흉부 외상이라 보수적으로 즉시 119 유지.
    _send(flow, "무릎이 아파요", session="ci")
    out = _send(flow, "넘어져서 가슴을 부딪혔어요", session="ci")
    assert out["riskLevel"] == "high"
