"""2026-11-11 사용자 실사용 반영.
① 보호자 상담(WARD 아님)은 #196 이전 한 번 되묻기 그대로(질문 3개 + 진료과 한 줄 + 답 칩 4개, 결과는 긴 답) - 이모지만 없다.
   단계 문답·짧은 결과(3문장·120자)는 피보호자 본인(role=WARD)만. 응급·자해·위험 신호 119 는 양쪽 그대로.
② 관절 위험 신호 질문이 다쳤다고 단정하지 않는다.
③ 진료과 이름이 들어 있어도 검사·문진·준비·비용을 묻는 말은 진료과 고정 답이 아니고, 증상 요약 틀도 아닌 일반 대화 틀로 답한다."""
from __future__ import annotations

import json
from typing import Any

import pytest

from app.models.chat_log import ChatLog
from app.services import chat_keywords as kw
from app.services import followup_answer_chips as fac
from app.services import followup_steps as fs
from app.services.chat_service import ChatService
from app.services.medical_llm_service import MedicalLlmService as M
from tests.test_chat_followup_question import _send, env  # noqa: F401 - 같은 대화 흐름 하네스


class _Llm:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, message, ctx, history, intent=None, processed_symptom=None, proxy=False, **kwargs) -> dict[str, Any]:  # noqa: ARG002
        self.calls.append({"message": message, "intent": intent, "processed_symptom": processed_symptom, "proxy": proxy,
                           "concise": kwargs.get("concise")})
        risk = "high" if intent == "emergency_guidance" else "medium"
        return {"reply": "첫째 문장이에요. 둘째 문장이에요. 셋째 문장이에요. 넷째 문장이에요. 신경과 진료를 받아 보세요.", "riskLevel": risk,
                "recommendedAction": "guardian_contact", "reservationRequired": False, "intent": intent, "engine": "medgemma",
                "modelName": "m", "summary": "s", "possibleCauses": [], "homeCare": [], "visitHospitalIf": [],
                "emergencyWarning": [M.EMERGENCY_WARNING] if intent == "emergency_guidance" else []}


@pytest.fixture()
def flow(env):  # noqa: F811
    env.llm = _Llm()
    import app.services.chat_service as cs
    cs.get_medical_llm_service = lambda: env.llm  # env 픽스처의 monkeypatch 가 원래 값으로 되돌린다
    return env


def _guardian(env, message: str, session: str = "s1") -> dict[str, Any]:
    return _send(env, message, session=session, ward="ab12", role="GUARDIAN")


def _ward(env, message: str, session: str = "s1") -> dict[str, Any]:
    return _send(env, message, session=session, role="WARD")


def _texts(out: dict[str, Any]) -> list[str]:
    layout = out.get("layout") or {}
    texts = [out["reply"], *(out.get("suggestedQuestions") or []), layout.get("headline") or ""]
    return texts + [x for k in ("body", "questions", "homeCare", "visitHospitalIf") for x in layout.get(k) or []]


# ---- ① 보호자 = #196 이전 모양 -----------------------------------------------------------------------------

# 증상 묶음별 대표 문장(옛 답 칩 표의 묶음 순서) - 묶음마다 옛 대리 칩 4개가 그대로 나간다.
GROUP_SAMPLES = [("어깨가 아파요", "joint"), ("머리가 아파요", "head"), ("배가 아파요", "belly"), ("어지러워요", "dizzy"),
                 ("저려요", "numb"), ("기침이 나요", "cough"), ("가려워요", "default")]


@pytest.mark.parametrize(("message", "group"), GROUP_SAMPLES)
def test_guardian_first_turn_is_the_old_three_questions(flow, message: str, group: str) -> None:
    out = _guardian(flow, message)
    assert flow.llm.calls == []
    assert len(out["layout"]["questions"]) == 3 and "어르신께서 언제부터 그러셨나요?" in out["reply"]
    assert "진료를 생각해 보셔도 좋아요." in out["reply"]  # 진료과 한 줄(Q2)
    assert "followupStep" not in out.get("meta", {}) or not out["meta"].get("followupStep")
    assert fac.legacy_group_name(message) == group
    assert out["suggestedQuestions"] == fac.legacy_chips_for(message, proxy=True) and out["suggestedKind"] == "answer"
    assert len(out["suggestedQuestions"]) == 4
    assert all(kw.strip_layout_emoji(t) == t for t in _texts(out))  # 이모지 없음(CHAT_LAYOUT_EMOJI 끔)


@pytest.mark.parametrize(("message", "group"), GROUP_SAMPLES)
def test_guardian_answer_turn_joins_and_is_not_concise(flow, message: str, group: str) -> None:  # noqa: ARG001
    _guardian(flow, message)
    chip = fac.legacy_chips_for(message, proxy=True)[0]
    out = _guardian(flow, chip)
    assert len(flow.llm.calls) == 1
    call = flow.llm.calls[0]
    assert call["message"] == f"{message} / {chip}" and call["intent"] == "medical_advice"
    assert call["concise"] is None and call["proxy"] is True  # 긴 답(#173 짧게는 피보호자만)
    assert out["engine"] == "medgemma" and not out.get("suggestedKind")
    assert "신경과 진료를 받아 보세요." in out["reply"]  # 3문장·120자로 자르지 않는다


def test_guardian_session_with_saved_step_state_uses_the_old_join(flow) -> None:
    # 배포 직전에 보호자 단계 문답이 시작된 기록(followupStep 있음) - 보호자는 단계 흐름을 잇지 않고 옛 이어 붙이기로 답한다.
    _guardian(flow, "머리가 아파요")
    row = flow.db.query(ChatLog).order_by(ChatLog.id.desc()).first()
    ctx = json.loads(row.context_json)
    ctx["upstreamMeta"]["followupStep"] = fs.new_state("머리가 아파요", True)
    row.context_json = json.dumps(ctx, ensure_ascii=False)
    flow.db.commit()
    out = _guardian(flow, "네")
    assert len(flow.llm.calls) == 1 and flow.llm.calls[0]["message"] == "머리가 아파요 / 네"
    assert out["riskLevel"] != "high" and flow.llm.calls[0]["concise"] is None


def test_guardian_without_ward_id_uses_self_wording(flow) -> None:
    out = _send(flow, "어깨가 아파요", role="GUARDIAN")
    assert "언제부터 그러셨나요?" in out["reply"] and "어르신께서" not in out["reply"]
    assert out["suggestedQuestions"] == fac.legacy_chips_for("어깨가 아파요", proxy=False)


def test_role_missing_is_the_guardian_shape(flow) -> None:
    out = _send(flow, "어깨가 아파요")
    assert len(out["layout"]["questions"]) == 3


def test_ward_keeps_the_step_flow(flow) -> None:
    out = _ward(flow, "어깨가 아파요")
    assert out["layout"]["questions"] == ["다치셨거나 많이 부었어요?"] and out["suggestedQuestions"] == ["네", "아니요"]
    _ward(flow, "아니요")
    _ward(flow, "며칠 됐어요")
    _ward(flow, "아니요")
    assert len(flow.llm.calls) == 1 and flow.llm.calls[0]["concise"] is True


def test_guardian_red_flag_word_keeps_the_119_closing(flow) -> None:
    message = "숨이 좀.."
    if not M.needs_followup_question(message):
        pytest.skip("되묻기 대상이 아닌 문장")
    out = _guardian(flow, message)
    assert "119" in out["reply"] and out["suggestedQuestions"] == []  # 옛 기준: 위험 신호 되묻기엔 답 칩 없음


@pytest.mark.parametrize("message", ["가슴이 아파요", "식은땀이 나고 가슴이 조여요", "갑자기 말이 어눌해졌어요"])
def test_emergency_is_the_same_for_guardian_and_ward(flow, message: str) -> None:
    g = _guardian(flow, message, session="g")
    w = _ward(flow, message, session="w")
    for out in (g, w):
        assert "119" in out["reply"] and out["riskLevel"] == "high"
        assert not (out.get("layout") or {}).get("questions")


def test_self_harm_is_the_same_for_guardian_and_ward(flow) -> None:
    g = _guardian(flow, "죽고 싶어요", session="g")
    w = _ward(flow, "죽고 싶어요", session="w")
    for out in (g, w):
        assert "109" in out["reply"] and flow.llm.calls == []


# ---- ② 단정하지 않는 질문 ------------------------------------------------------------------------------------


def test_no_question_presumes_an_injury() -> None:
    for q in fs.all_questions():
        for text in (q.text, q.proxy):
            assert "다친 뒤" not in text and "다치신 뒤" not in text, text
    joint = next(g for g in fs.all_groups() if g.name == "joint").red_flag
    assert (joint.text, joint.proxy) == ("다치셨거나 많이 부었어요?", "다치셨거나 많이 부으셨대요?")


@pytest.mark.parametrize(("answer", "verdict"), [
    ("네", "yes"), ("넘어졌어요", "yes"), ("많이 부었어요", "yes"), ("아니요", "no"), ("안 부었어요", "no"),
    ("다치진 않았어요", "no"),
])
def test_joint_question_reads_answers(answer: str, verdict: str) -> None:
    state = fs.new_state("어깨가 아파요", False)
    assert fs.interpret(state, answer)[0] == verdict


# ---- ③ 진료과 질문 vs 진료과 정보 질문 -------------------------------------------------------------------------

_HISTORY = [{"role": "user", "content": "어지러워요"}, {"role": "assistant", "content": "신경과 진료를 권해드려요."}]

DEPT_TABLE = [
    # (문장, 진료과 고정 답인가, 진료과 정보 질문인가)
    ("무슨 과 가야 해요?", True, False),
    ("어떤 과로 가요?", True, False),
    ("어느 과 가면 돼요?", True, False),
    ("신경과 말고 어떤 과 가요?", True, False),
    ("검사 받으려면 어느 과 가요?", True, False),  # 묻는 말이 과 앞 - 어디로 가나
    ("신경과는 어떤 검사를 받게 되나요?", False, True),
    ("신경과에서 어떤 문진을 하나요?", False, True),
    ("내과는 어떤 검사를 하나요?", False, True),
    ("정형외과 가면 뭐 해요?", False, True),
    ("내과 진료비 얼마예요?", False, True),
    ("어느 과에서 검사를 받나요?", False, True),
    ("어떤 과에 가서 무엇을 준비해야 하나요?", False, True),
    ("신경과 예약해 줘", False, False),  # 예약
    ("가슴이 아픈데 내과에서 무슨 검사 해요?", False, False),  # 위험 신호 - 응급 판정이 먼저
    ("머리가 아파요", False, False),
]


@pytest.mark.parametrize(("message", "fixed", "info"), DEPT_TABLE, ids=[r[0] for r in DEPT_TABLE])
def test_department_question_table(message: str, fixed: bool, info: bool) -> None:
    assert bool(ChatService._department_question_reply(message, _HISTORY)) is fixed
    assert kw.is_department_info_question(message) is info


@pytest.mark.parametrize("ask", ["신경과는 어떤 검사를 받게 되나요?", "신경과에서 어떤 문진을 하나요?"])
@pytest.mark.parametrize("who", ["ward", "guardian"])
def test_department_info_question_after_a_result_is_general_chat(flow, ask: str, who: str) -> None:
    send = _ward if who == "ward" else _guardian
    for m in (["어지러워요", "아니요", "오늘부터요", "속은 괜찮아요"] if who == "ward" else ["어지러워요", "오늘부터 그러시대요"]):
        send(flow, m)
    flow.llm.calls.clear()
    out = send(flow, ask)
    assert len(flow.llm.calls) == 1
    call = flow.llm.calls[0]
    assert call["message"] == ask and call["intent"] == "general_chat" and call["processed_symptom"] is None
    assert out["engine"] == "medgemma"


def test_department_where_question_keeps_the_fixed_answer(flow) -> None:
    _guardian(flow, "어지러워요")
    _guardian(flow, "오늘부터 그러시대요")
    flow.llm.calls.clear()
    out = _guardian(flow, "무슨 과 가야 해요?")
    assert out["engine"] == "rule" and out["reply"].endswith("찾아드릴까요?") and flow.llm.calls == []


def test_department_info_with_red_flag_stays_emergency(flow) -> None:
    flow.router.emergency = True
    out = _ward(flow, "가슴이 아픈데 내과에서 무슨 검사 해요?")
    assert "119" in out["reply"] and out["riskLevel"] == "high"
    assert all(c["intent"] != "general_chat" for c in flow.llm.calls)
