"""되묻기 단계 문답(2026-11-09): 한 턴에 질문 하나 + 답 버튼, 최대 3질문(위험 신호 → 언제부터 → 동반 증상).
위험 신호 "네"는 그 자리에서 끝(119 묶음은 규칙 119, 관절·기본은 오늘 진료). 질문 턴은 MedGemma 0회, 결과 턴에 한 번.
응급·자해·위기는 단계 처리보다 먼저(판정 순서 그대로). 진행 상태는 서버 기록에서만 복원한다. 답 문장에 이모지는 없다."""
from __future__ import annotations

import json
from typing import Any

import pytest

from app.models.chat_log import ChatLog
from app.services import chat_keywords as kw
from app.services import followup_answer_chips as fac
from app.services import followup_steps as fs
from app.services.chat_service import _is_closing_message
from app.services.medical_llm_service import MedicalLlmService as M
from tests.test_chat_followup_question import _send, env  # noqa: F401 - 같은 대화 흐름 하네스

WARD_PROMPTS = ["머리가 아파요", "배가 아파요", "무릎이 아파요", "어지러워요"]  # FE 피보호자 기본 칩
YN = ["네", "아니요"]


class _Llm:
    """결과 턴의 concise 인자까지 받는 가짜 MedGemma(호출 문장만 기록)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, message, ctx, history, intent=None, processed_symptom=None, proxy=False, **kwargs) -> dict[str, Any]:  # noqa: ARG002
        self.calls.append({"message": message, "intent": intent, "proxy": proxy, "concise": kwargs.get("concise")})
        risk = "high" if intent == "emergency_guidance" else "medium"
        return {"reply": "쉬시면서 물을 자주 드세요. 증상이 계속되면 병원 진료를 받아 보세요.", "riskLevel": risk,
                "recommendedAction": "guardian_contact", "reservationRequired": False, "intent": intent, "engine": "medgemma",
                "modelName": "m", "summary": "s", "possibleCauses": [], "homeCare": [], "visitHospitalIf": [],
                "emergencyWarning": [M.EMERGENCY_WARNING] if intent == "emergency_guidance" else []}


@pytest.fixture()
def flow(env):  # noqa: F811
    env.llm = _Llm()
    import app.services.chat_service as cs
    cs.get_medical_llm_service = lambda: env.llm  # env 픽스처의 monkeypatch 가 테스트 끝에 원래 값으로 되돌린다
    return env


def _no_emoji(out: dict[str, Any]) -> bool:
    texts = [out["reply"], *(out.get("suggestedQuestions") or [])]
    layout = out.get("layout") or {}
    texts += [layout.get("headline") or "", *[x for k in ("body", "questions", "homeCare", "visitHospitalIf") for x in layout.get(k) or []]]
    return all(kw.strip_layout_emoji(t) == t for t in texts)


def _question(out: dict[str, Any]) -> str:
    qs = out["layout"]["questions"]
    assert len(qs) == 1
    return qs[0]


# ---- 표 조건 ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("q", fs.all_questions(), ids=lambda q: q.text)
def test_every_question_is_short_and_chips_are_plain(q) -> None:
    assert len(q.text) <= 22 and len(q.proxy) <= 22 and q.text.endswith("?") and q.proxy.endswith("?")
    for chip in (*q.chips, *q.proxy_chips):
        # 답 버튼은 이어서 물어볼 질문(sanitize, 2자 이상)을 거치지 않는다 - BE 정리 기준(1~30자)·제어문자·숫자 없음.
        assert 1 <= len(chip) <= 12 and chip == " ".join(chip.split()) and not any(ch.isdigit() for ch in chip)
        assert not M.emergency_signals(chip) and not M.self_harm_signals(chip) and not kw.has_red_flag(chip)
        assert not M.drug_alcohol_rules(chip) and not any(k in chip for k in kw.RESERVATION_KEYWORDS)
        assert not _is_closing_message(chip)  # 인사·종료로 읽히면 단계 흐름이 끝난다
    assert 2 <= len(q.chips) <= 4 and len(q.chips) == len(q.proxy_chips)


@pytest.mark.parametrize("q", fs.all_questions(), ids=lambda q: q.text)
def test_answer_phrases_never_change_judgment(q) -> None:
    # MedGemma 에 이어 붙이는 답 문장 - 응급·자해·위험 신호 낱말이 섞이면 결과 턴 판정이 바뀐다.
    for phrase in (q.yes_phrase, q.no_phrase):
        if phrase:
            assert not M.emergency_signals(phrase) and not M.self_harm_signals(phrase) and not kw.has_red_flag(phrase), phrase


@pytest.mark.parametrize(("message", "group"), [
    ("머리가 아파요", "head"), ("두통이 있어요", "head"), ("배가 아파요", "belly"), ("속이 안 좋아요", "belly"),
    ("무릎이 아파요", "joint"), ("허리가 아파요", "joint"), ("손목이 아파요", "joint"), ("어지러워요", "dizzy"),
    ("머리가 어지러워요", "dizzy"), ("손이 저려요", "numb"), ("기침이 나요", "cough"), ("열이 나요", "cough"),
    ("귀가 아파요", "default"), ("피부가 가려워요", "default"), ("목이 아파요", "cough"), ("발이 부었어요", "default"),
    ("숨이 좀..", "chest"), ("가슴이 좀 답답", "chest"), ("숨차네", "chest"),
])
def test_group_matching(message: str, group: str) -> None:
    assert fs.group_for(message).name == group


@pytest.mark.parametrize("message", ["어지러워", "기침이 나요", "귀가 아파요", "피부가 가려워요", "숨이 좀..", *WARD_PROMPTS])
def test_first_turn_is_one_question_with_yes_no(message: str) -> None:
    assert M.needs_followup_question(message)
    r = M.followup_question_response(message)
    assert r["reply"].count("· ") == 1 and r["engine"] == "rule"
    assert fac.answer_chips(r, message) == YN


@pytest.mark.parametrize(("text", "verdict"), [
    ("네", "yes"), ("예", "yes"), ("응", "yes"), ("맞아요", "yes"), ("네 갑자기 너무 아파요", "yes"), ("심해요", "yes"),
    ("아니요", "no"), ("아뇨", "no"), ("없어요", "no"), ("심하지는 않아요", "no"), ("그렇지 않아요", "no"), ("열은 없어요", "describe"),
    ("아프지 않아요", "no"),
    ("어제부터요", "describe"), ("배가 아파요", "describe"),  # "어"로 시작해도 "네"가 아니다 - 다른 질문의 답·새 증상은 설명
    ("몰라요", "unknown"), ("음", "unclear"), ("뭐라고요?", "unclear"),
    ("어제부터요, 열도 조금 있어요", "describe"), ("아니요, 어제부터 아파요", "describe"),
])
def test_interpret_red_flag_question(text: str, verdict: str) -> None:
    assert fs.interpret(fs.new_state("머리가 아파요", False), text)[0] == verdict


@pytest.mark.parametrize(("text", "verdict"), [
    ("오늘부터요", "answer"), ("며칠 됐어요", "answer"), ("어제부터요", "answer"), ("사흘째", "answer"),
    ("일주일째인데 점점 심해져요", "describe"), ("모르겠어요", "unknown"), ("음", "unclear"),
])
def test_interpret_onset_question(text: str, verdict: str) -> None:
    state = {**fs.new_state("머리가 아파요", False), "index": 1, "answers": ["x"], "turns": 2}
    assert fs.interpret(state, text)[0] == verdict


@pytest.mark.parametrize("bad", [
    None, "x", {}, {"group": "nope", "index": 0, "answers": [], "turns": 1},
    {"group": "head", "index": 3, "answers": ["a", "b", "c"], "turns": 4},
    {"group": "head", "index": 1, "answers": [], "turns": 2},  # 답 수와 순번이 어긋남
    {"group": "head", "index": 0, "answers": [], "turns": 99},
    {"group": "head", "index": 1, "answers": ["a" * 41], "turns": 2},
])
def test_damaged_state_is_ignored(bad) -> None:
    assert fs.valid_state(bad) is None


# ---- 대화 흐름 -----------------------------------------------------------------------------------------


def test_three_steps_then_one_model_call(flow) -> None:
    out = _send(flow, "머리가 아파요")
    assert _question(out) == "갑자기 아주 심하게 아프세요?" and out["suggestedQuestions"] == YN and out["suggestedKind"] == "answer"
    out = _send(flow, "아니요")
    assert _question(out) == "언제부터 그러셨어요?" and out["suggestedQuestions"] == ["오늘부터요", "며칠 됐어요", "오래됐어요"]
    out = _send(flow, "며칠 됐어요")
    assert _question(out) == "열도 나세요?" and out["suggestedQuestions"] == YN
    assert flow.llm.calls == []  # 질문 턴은 MedGemma 0회
    out = _send(flow, "네")
    assert len(flow.llm.calls) == 1
    call = flow.llm.calls[0]
    assert call["message"] == "머리가 아파요 / 갑자기 아주 심한 건 아니에요 / 며칠 됐어요 / 열도 나요"
    assert call["intent"] == "medical_advice" and call["concise"] is True
    assert "suggestedKind" not in out and out["layout"]["questions"] == []
    row = flow.db.query(ChatLog).order_by(ChatLog.id.desc()).first()
    meta = json.loads(row.context_json)["upstreamMeta"]
    assert meta.get("followupResult") is True and not meta.get("followupAsked")


@pytest.mark.parametrize("prompt", ["머리가 아파요", "배가 아파요", "어지러워요", "저려요", "기침이 나요", "숨이 좀.."])
def test_red_flag_yes_is_119_right_away(flow, prompt: str) -> None:
    _send(flow, prompt)
    out = _send(flow, "네")
    assert out["riskLevel"] == "high" and out["emergencyWarning"]
    first = out["reply"].split("\n")[0]
    assert "119" in first and "전화" in first
    assert flow.llm.calls == []  # 라우터·MedGemma 를 기다리지 않는 규칙 답
    assert "suggestedKind" not in out and _no_emoji(out)
    out = _send(flow, "아니요")  # 응급 답 다음 턴은 지금 규칙(119 한 줄 덧붙임) - 단계 흐름이 이어지지 않는다
    assert "suggestedKind" not in out


@pytest.mark.parametrize(("prompt", "department"), [
    ("무릎이 아파요", "정형외과"), ("귀가 아파요", "이비인후과"), ("눈이 아파요", "안과"),  # 2026-11-09 점검 L-2: 증상 문장의 진료과
])
def test_joint_and_default_red_flag_yes_is_visit_today(flow, prompt: str, department: str) -> None:
    _send(flow, prompt)
    out = _send(flow, "네")
    assert out["riskLevel"] != "high" and not out["emergencyWarning"]
    assert f"오늘 바로 {department} 진료" in out["reply"]
    assert flow.llm.calls == []


def test_free_text_description_goes_to_result_with_red_flag_line(flow) -> None:
    _send(flow, "어지러워요")
    out = _send(flow, "어제부터요, 열도 조금 있어요")
    assert flow.llm.calls[-1]["message"] == "어지러워요 / 어제부터요, 열도 조금 있어요"
    assert "suggestedKind" not in out
    # 위험 신호 질문에 답하지 않았으니 그 묶음의 119 기준 한 줄
    assert "말이 어눌해지거나 팔다리 힘이 빠지면 바로 119에 전화하세요." in out["reply"]


def test_long_free_text_still_gets_red_flag_line(flow) -> None:
    # "아니요"로 시작해도 긴 설명이면 위험 신호 질문에 분명히 답한 것으로 보지 않는다 - 119 기준 한 줄을 넣는다.
    _send(flow, "머리가 아파요")
    out = _send(flow, "아니요, 어제부터 아파요")
    assert flow.llm.calls[-1]["message"] == "머리가 아파요 / 아니요, 어제부터 아파요"
    assert "갑자기 아주 심하게 아프거나 말이 어눌해지면 바로 119에 전화하세요." in out["reply"]


def test_short_clear_no_has_no_extra_119_line(flow) -> None:
    _send(flow, "머리가 아파요")
    _send(flow, "아니요")
    out = _send(flow, "일주일째인데 점점 심해져요")  # 기간 질문에 설명 - 위험 신호는 이미 "아니요"
    assert flow.llm.calls[-1]["message"] == "머리가 아파요 / 갑자기 아주 심한 건 아니에요 / 일주일째인데 점점 심해져요"
    assert "119" not in out["reply"]


def test_unclear_answer_is_asked_again_once(flow) -> None:
    _send(flow, "머리가 아파요")
    out = _send(flow, "음")
    assert _question(out) == "갑자기 아주 심하게 아프세요?" and out["suggestedQuestions"] == YN
    assert "'네'나 '아니요'로 알려 주세요." in out["reply"]
    out = _send(flow, "음")  # 두 번째도 못 알아들으면 모름으로 보고 넘어간다(무한 반복 없음)
    assert _question(out) == "언제부터 그러셨어요?"
    out = _send(flow, "몰라요")
    assert _question(out) == "열도 나세요?"
    _send(flow, "아니요")
    assert flow.llm.calls[-1]["message"] == "머리가 아파요 / 열은 없어요"


def test_closing_mid_flow_ends_steps(flow) -> None:
    _send(flow, "머리가 아파요")
    _send(flow, "아니요")
    out = _send(flow, "고마워요")
    assert "suggestedKind" not in out and out["layout"] is None or not (out.get("layout") or {}).get("questions")
    row = flow.db.query(ChatLog).order_by(ChatLog.id.desc()).first()
    assert "followupStep" not in json.loads(row.context_json)["upstreamMeta"]
    calls = len(flow.llm.calls)
    _send(flow, "네")  # 흐름이 끝났으니 "네"는 단계 답이 아니다(119 로 가지 않는다)
    assert len(flow.llm.calls) == calls + 1


def test_proxy_flow_uses_carer_wording(flow) -> None:
    out = _send(flow, "기침이 나요", ward="ab12")
    assert _question(out) == "숨쉬기 힘들어하세요?"
    out = _send(flow, "아니요", ward="ab12")
    assert _question(out) == "언제부터 그러셨대요?" and out["suggestedQuestions"] == ["오늘부터래요", "며칠 되셨대요", "오래되셨대요"]
    _send(flow, "며칠 되셨대요", ward="ab12")
    _send(flow, "아니요", ward="ab12")
    assert flow.llm.calls[-1]["proxy"] is True and flow.llm.calls[-1]["concise"] is True


def test_emergency_mid_flow_goes_emergency_first(flow) -> None:
    _send(flow, "머리가 아파요")
    _send(flow, "아니요")
    out = _send(flow, "가슴이 아파요")  # 응급 키워드 - 단계 답으로 읽지 않는다
    assert flow.llm.calls[-1]["intent"] == "emergency_guidance"
    assert out["riskLevel"] == "high" and "suggestedKind" not in out


def test_router_emergency_mid_flow_goes_emergency_first(flow) -> None:
    _send(flow, "머리가 아파요")
    flow.router.emergency = True
    out = _send(flow, "네")
    assert flow.llm.calls and flow.llm.calls[-1]["intent"] == "emergency_guidance"
    assert out["riskLevel"] == "high"


def test_self_harm_mid_flow_goes_crisis_reply(flow) -> None:
    _send(flow, "머리가 아파요")
    out = _send(flow, "죽고 싶어요")
    row = flow.db.query(ChatLog).order_by(ChatLog.id.desc()).first()
    assert json.loads(row.context_json)["upstreamMeta"].get("selfHarm") is True
    assert "109" in out["reply"] and "suggestedKind" not in out


def test_other_session_does_not_continue_steps(flow) -> None:
    _send(flow, "머리가 아파요", session="a")
    out = _send(flow, "네", session="b")  # 다른 상담의 "네"는 위험 신호 답이 아니다
    assert out["riskLevel"] != "high"


def test_every_reply_in_flows_has_no_emoji(flow) -> None:
    for i, prompt in enumerate(WARD_PROMPTS):
        s = f"e{i}"
        for text in (prompt, "아니요", "며칠 됐어요", "네"):
            assert _no_emoji(_send(flow, text, session=s))
    assert _no_emoji(_send(flow, "가슴이 아파요", session="x"))
    assert _no_emoji(_send(flow, "죽고 싶어요", session="y"))


def test_old_followup_record_keeps_old_one_answer_flow(flow) -> None:
    # 배포 직전에 시작된 옛 되묻기(followupStep 없음) - 답 한 번에 "앞 질문 / 이번 답"으로 결과(옛 흐름 그대로).
    _send(flow, "머리가 아파요")
    row = flow.db.query(ChatLog).order_by(ChatLog.id.desc()).first()
    ctx = json.loads(row.context_json)
    ctx["upstreamMeta"].pop("followupStep")
    row.context_json = json.dumps(ctx, ensure_ascii=False)
    flow.db.commit()
    _send(flow, "네")
    assert flow.llm.calls[-1]["message"] == "머리가 아파요 / 네"


def test_chips_are_not_saved_and_log_has_count_only(flow, caplog) -> None:
    import logging
    with caplog.at_level(logging.INFO):
        _send(flow, "머리가 아파요")
    row = flow.db.query(ChatLog).order_by(ChatLog.id.desc()).first()
    assert "suggestedKind" not in json.dumps({"ctx": row.context_json}, ensure_ascii=False)
    lines = [r.getMessage() for r in caplog.records if "source=answer" in r.getMessage()]
    assert lines == ["[CHAT-SUGGEST] source=answer count=2"]
    with caplog.at_level(logging.INFO):
        _send(flow, "아니요")
    step_logs = [r.getMessage() for r in caplog.records if "followup_step=" in r.getMessage()]
    assert step_logs == ["[CHAT-QUALITY] followup_step=1 verdict=no group=head"]  # 본문 없이 단계·판정만


# 2026-11-03 점검 M-2(옛 test_followup_answer_chips 에서 옮김): 증상 낱말이 있어도 증상이 아닌 말은 되묻지 않는다.
@pytest.mark.parametrize("message", ["속상해요", "너무 속상해", "열받아요", "열이 받쳐요", "배고파요", "배가 고파요", "배불러요", "배가 불러요"])
def test_not_symptom_words_are_not_followup(message: str) -> None:
    assert not M.needs_followup_question(message)


@pytest.mark.parametrize("message", ["배가 아파요", "속이 쓰려요", "열이 나요", "배아퍼"])
def test_real_symptoms_still_followup(message: str) -> None:
    assert M.needs_followup_question(message)



# ---- 2026-11-09 기능 점검 반영(H-1·M-1·M-2·M-3·L-1·L-2) ---------------------------------------------------------


def _at(message: str, index: int = 0) -> dict:
    state = fs.new_state(message, False)
    state.update(index=index, answers=["x"] * index, turns=index + 1)
    return state


@pytest.mark.parametrize(("prompt", "text", "verdict"), [
    # H-1: 위험 신호 질문의 자연스러운 긍정·강조 말·그 질문의 증상 말은 "네"
    ("머리가 아파요", "너무 아파요", "yes"), ("머리가 아파요", "많이 아파요", "yes"), ("머리가 아파요", "엄청 심해요", "yes"),
    ("머리가 아파요", "넹", "yes"), ("머리가 아파요", "웅", "yes"), ("머리가 아파요", "엉", "yes"), ("머리가 아파요", "어", "yes"),
    ("머리가 아파요", "그려", "yes"), ("머리가 아파요", "맞어요", "yes"), ("머리가 아파요", "ㅇㅇ", "yes"),
    ("머리가 아파요", "머리가 깨질 것 같아요", "yes"),
    ("머리가 아파요", "아파요", "unclear"),  # 강조 없는 한마디는 "아주 심하게"인지 몰라 한 번 더 묻는다
    ("어지러워요", "말이 좀 어눌해요", "yes"), ("어지러워요", "팔에 힘이 빠져요", "yes"), ("어지러워요", "힘이 빠지진 않아요", "no"),
    ("숨이 좀..", "식은땀 나요", "yes"), ("숨이 좀..", "가슴은 괜찮아요", "no"), ("기침이 나요", "숨이 차요", "yes"),
    ("무릎이 아파요", "식은땀 나요", "describe"),
    ("머리가 아파요", "일주일째인데 점점 심해져요", "describe"),  # 기간·점점 - 강조 말 규칙을 쓰지 않는다(fq.13)
    ("무릎이 아파요", "계단 오를 때 심해요, 붓기도 있어요", "describe"),  # 다친 말이 없으면 설명(fq.19)
    ("무릎이 아파요", "넘어져서 그래요", "yes"),  # 오늘 진료 묶음 - 위험 신호 낱말은 설명(결과에 119 줄)
    # L-1: 정도 말 - 심한 정도를 묻는 질문은 "아니요", 증상이 있는지 묻는 질문은 "네"
    ("머리가 아파요", "조금요", "no"), ("머리가 아파요", "그냥 그래요", "no"), ("어지러워요", "조금요", "yes"), ("숨이 좀..", "약간요", "yes"),
    # M-1: 되묻는 말은 "네"가 아니다
    ("머리가 아파요", "네?", "unclear"), ("머리가 아파요", "예?", "unclear"), ("머리가 아파요", "진짜요?", "unclear"),
    # M-3: 예/아니요 질문의 "괜찮아요"는 "아니요"
    ("머리가 아파요", "괜찮아요", "no"), ("머리가 아파요", "네 괜찮아요", "no"), ("기침이 나요", "지금은 괜찮아요", "no"),
    ("머리가 아파요", "ㄴㄴ", "no"), ("머리가 아파요", "no", "no"), ("머리가 아파요", "yes", "yes"),
])
def test_interpret_audit_cases(prompt: str, text: str, verdict: str) -> None:
    assert fs.interpret(_at(prompt), text)[0] == verdict


@pytest.mark.parametrize(("text", "verdict"), [("38도요", "yes"), ("미열 있어요", "yes"), ("조금요", "yes"), ("없어", "no"), ("잘 모르겠어요", "unknown")])
def test_interpret_fever_question(text: str, verdict: str) -> None:
    assert fs.interpret(_at("머리가 아파요", 2), text)[0] == verdict


def test_interpret_onset_unknown_memory() -> None:
    assert fs.interpret(_at("머리가 아파요", 1), "기억 안 나요")[0] == "unknown"


@pytest.mark.parametrize(("prompt", "answer"), [("머리가 아파요", "너무 아파요"), ("숨이 좀..", "식은땀 나요"), ("어지러워요", "팔에 힘이 빠져요")])
def test_natural_yes_to_red_flag_is_119_right_away(flow, prompt: str, answer: str) -> None:
    _send(flow, prompt)
    out = _send(flow, answer)
    assert out["riskLevel"] == "high" and out["emergencyWarning"] and flow.llm.calls == []


def test_emergency_keyword_answer_goes_emergency_path_first(flow) -> None:
    # "말이 어눌해요"는 응급 키워드 - 단계 처리보다 먼저 응급 경로(판정 순서 그대로).
    _send(flow, "어지러워요")
    out = _send(flow, "말이 어눌해요")
    assert out["riskLevel"] == "high" and out["emergencyWarning"]


def test_pardon_is_asked_again_not_119(flow) -> None:
    _send(flow, "머리가 아파요")
    out = _send(flow, "네?")
    assert out["riskLevel"] != "high" and _question(out) == "갑자기 아주 심하게 아프세요?"


def test_okay_answer_continues_steps(flow) -> None:
    _send(flow, "머리가 아파요")
    out = _send(flow, "괜찮아요")  # 종료 인사가 아니라 "아니요" - 다음 질문
    assert _question(out) == "언제부터 그러셨어요?" and flow.llm.calls == []


def test_department_question_mid_flow_gets_fixed_answer(flow) -> None:
    _send(flow, "머리가 아파요")
    out = _send(flow, "무슨 과에 가야 돼요?")
    assert "신경과로 가시면 돼요" in out["reply"] and flow.llm.calls == []
