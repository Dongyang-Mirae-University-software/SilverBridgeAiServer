"""2026-11-11 태블릿 실사용 반영(피보호자 화면).

② 증상 + 먹을 것·할 일 질문: 그 질문일 때만 프롬프트에 한 줄(advice 첫 문장 = 방법, 둘째 = 진료과 + 기준). 다른 질문 프롬프트는 글자 그대로.
③ 확인 카드 [고치기 → 진료과]: 같은 병원·날짜를 두고 그 병원 진료과만 고른다. 끝에 [다른 병원에서 찾기], 목록을 못 받으면 예전 동작.
④ 응급 다음 턴: 라우터가 응급이 아니라고 하면 119 한 줄만(10/29) - 서버 이력이 다음 턴을 응급으로 묶지 않는다.
⑤ 이어서 물어볼 질문: 서버 고정 안전 줄(응급 다음 턴 줄·단계 결과 119 기준 줄)만으로는 비우지 않는다. 응급·자해·위기는 그대로 빈 목록.
합성 예문만 쓴다(실제 상담 없음)."""
from __future__ import annotations

import re
from typing import Any

import pytest

from app.core.config import Settings
from app.prompts.medical_chat_prompts import _CARE_QUESTION_LINE, build_user_prompt
from app.schemas.chat_schema import ChatContext, ChatRequest
from app.services import follow_up_questions as fq
from app.services.chat_service import _EMERGENCY_FOLLOWUP
from app.services.medical_llm_service import MedicalLlmService
from app.services.reservation_orchestrator import OTHER_HOSPITAL_OPTION, ReservationOrchestrator, _clear_stale_links, _linked_fields
from app.services.reservation_intake import ReservationDraft
from tests.reservation_flow_helpers import FUTURE, FakeReservation, make_orchestrator
from tests.test_chat_history_server_side import _router
from tests.test_chat_reply_layout import _Loader
from tests.test_reservation_confirm_cancel import _confirm_card, _turn, _value
from tests.test_reservation_confirm_chat_service import env  # noqa: F401

SPLIT = re.compile(r"(?<=[.!?。])\s+")
FULL = {"reservation_date": FUTURE, "reservation_time": "14:00"}

# ---- ② 먹을 것·할 일 질문 ---------------------------------------------------------------------------------

CARE = ["설사할 때 무엇을 먹으면 좋을까요?", "변비에 뭘 먹으면 좋아요?", "기침할 때 어떻게 하면 좋아요?",
        "설사할 때 피해야 할 음식은요?", "배가 아플 때 죽 먹어도 괜찮아요?", "무릎 아플 때 집에서 할 수 있는 거 있어요?"]
NOT_CARE = ["설사해요", "머리가 아파요", "허리가 일주일째 아파요", "어떤 과에 가야 해요?", "밥을 못 먹겠어요",
            "혈압약 먹으면 좋을까요?", "감기약 뭐 먹어요?", "술 마셔도 괜찮아요?", "영양제 뭘 먹으면 좋아요?",
            "어떻게 해야 할지 모르겠어요", "어쩔 줄 모르겠는데 어떻게 하면 돼요?",
            # 애매한 위기 표현(기능 점검 F-2) - 라우터가 놓쳐도 "방법"을 쓰라는 줄이 붙지 않는다
            "어떻게 하면 편하게 갈 수 있을까", "어떻게 하면 안 아프게 죽어요", "어떻게 하면 잠들고 안 깰 수 있어요",
            "다 끝내려면 어떻게 해야 해요", "어떻게 하면 사라질 수 있어요"]


@pytest.mark.parametrize("text", CARE)
def test_care_question_detected(text: str) -> None:
    assert MedicalLlmService.is_care_question(text, "medical_advice", False)


@pytest.mark.parametrize("text", NOT_CARE)
def test_not_care_question(text: str) -> None:
    assert not MedicalLlmService.is_care_question(text, "medical_advice", False)


def test_care_question_not_for_emergency_or_general_chat() -> None:
    assert not MedicalLlmService.is_care_question(CARE[0], "medical_advice", True)
    assert not MedicalLlmService.is_care_question(CARE[0], "emergency_guidance", True)
    assert not MedicalLlmService.is_care_question(CARE[0], "general_chat", False)


def _prompt(**kw) -> str:
    return build_user_prompt(message="설사해요", user_context={}, history_text="(없음)", intent="medical_advice",
                             emergency_hint=False, department_hint="내과", **kw)


def test_prompt_unchanged_unless_care_question() -> None:
    base = _prompt()
    assert _prompt(care_question=False) == base and _CARE_QUESTION_LINE.strip() not in base
    assert _prompt(care_question=True) == base + _CARE_QUESTION_LINE
    # 피보호자 줄과 함께면 관리 질문 줄이 먼저, 그 뒤 피보호자 줄
    both = _prompt(care_question=True, concise=True)
    assert both.index(_CARE_QUESTION_LINE.strip()) < both.index("어르신 본인")


class _PromptLoader(_Loader):
    def __init__(self, **fields: str) -> None:
        super().__init__(**fields)
        self.prompts: list[str] = []

    def generate_structured_text(self, system_prompt: str, user_prompt: str, prefill: str = "", **kwargs) -> str:
        self.prompts.append(user_prompt)
        return super().generate_structured_text(system_prompt, user_prompt, prefill, **kwargs)


DIET = dict(empathy="설사로 불편하셨겠어요.", symptoms="설사",
            advice="미음이나 죽처럼 부드러운 음식과 물을 조금씩 자주 드세요. 내과 진료를 권해드리고, 사흘 넘게 계속되면 꼭 진료를 받아 보세요.")


def _ward(env, message: str, session: str) -> dict[str, Any]:  # noqa: F811
    return env.svc.process_message(env.db, ChatRequest(userId="u1", message=message, sessionId=session, context=ChatContext(role="WARD")))


@pytest.mark.parametrize("prev", [None, "어제부터 설사를 세 번 했어요", "설사해요"])
def test_ward_care_answer_keeps_what_to_do_within_budget(env, monkeypatch, prev) -> None:  # noqa: F811
    _router(env, monkeypatch, route="symptom")
    loader = _PromptLoader(**DIET)
    real = MedicalLlmService(Settings(), loader)
    monkeypatch.setattr(env.llm, "generate", lambda m, c, h, **k: real.generate(m, c, h, **k))
    session = f"s-care-{prev}"
    if prev:
        _ward(env, prev, session)
    loader.prompts.clear()
    out = _ward(env, CARE[0], session)
    assert loader.prompts and _CARE_QUESTION_LINE.strip() in loader.prompts[-1]
    layout = out["layout"]
    content = " ".join([*(layout.get("body") or []), *(layout.get("homeCare") or [])])
    assert "죽" in content, layout  # 묻는 내용(먹을 것)이 피보호자 답에 남는다
    assert "내과" in " ".join([layout.get("headline") or "", *(layout.get("visitHospitalIf") or [])])  # 진료과 권유도 남는다
    # 모델 문장 길이 규칙(3문장·120자)은 그대로 - 119·고정 문장은 세지 않는다
    model = [s for line in [layout.get("headline") or "", *(layout.get("body") or []), *(layout.get("homeCare") or [])]
             for s in SPLIT.split(re.sub(r"^\W+\s", "", line)) if s and "119" not in s]
    assert len(model) <= 3 and sum(len(s) for s in model) <= 120 + 30  # 이모지·머리 기호 여유


def test_non_care_question_prompt_has_no_care_line(env, monkeypatch) -> None:  # noqa: F811
    _router(env, monkeypatch, route="symptom")
    loader = _PromptLoader(**DIET)
    real = MedicalLlmService(Settings(), loader)
    monkeypatch.setattr(env.llm, "generate", lambda m, c, h, **k: real.generate(m, c, h, **k))
    _ward(env, "어제부터 설사를 세 번 했어요", "s-plain")
    assert loader.prompts and _CARE_QUESTION_LINE.strip() not in loader.prompts[-1]


def test_no_care_line_during_crisis_history() -> None:
    """위기 대화 중(이력의 자해 신호)에는 관리 질문 줄을 붙이지 않는다 - 위기 답 틀(109·119)을 상담 틀로 바꾸지 않게(DGX cf.08)."""
    loader = _PromptLoader(**DIET)
    svc = MedicalLlmService(Settings(), loader)
    crisis = [{"role": "user", "content": "죽고 싶어요"}, {"role": "assistant", "content": "109에 이야기해 보세요."}]
    assert MedicalLlmService.self_harm_in_history(crisis)
    svc.generate("잠이 안 올 때 어떻게 하면 좋아요?", {}, crisis, intent="medical_advice")
    svc.generate("잠이 안 올 때 어떻게 하면 좋아요?", {}, [], intent="medical_advice")
    assert _CARE_QUESTION_LINE.strip() not in loader.prompts[0] and _CARE_QUESTION_LINE.strip() in loader.prompts[1]


def test_care_line_steers_away_from_fasting_and_unknown_department() -> None:
    assert "굶기" in _CARE_QUESTION_LINE and "가까운 병원" in _CARE_QUESTION_LINE


def test_care_line_answers_what_is_asked_and_fits_period_to_symptom() -> None:
    # 2026-11-12 기능 점검: 급한 증상에도 "2주"를 쓰던 답, 괄호 예시 낱말이 질문과 상관없이 첫 줄로 베껴지던 답
    assert "진료과 권유나 진료 기준으로 시작하지 않는다" in _CARE_QUESTION_LINE
    assert "짧게 써야 할 때도 이 첫 문장은 빼지 않" in _CARE_QUESTION_LINE
    assert "2~3일" in _CARE_QUESTION_LINE and "1~2주" in _CARE_QUESTION_LINE
    assert "119" not in _CARE_QUESTION_LINE
    for copied in ("물·", "부드러운 음식", "설사", "감기", "목이나", "눈의"):
        assert copied not in _CARE_QUESTION_LINE


@pytest.mark.parametrize("bad", ["설사할 때는 가급적 굶고 물만 드세요.", "가장 먼저 식사를 거르거나 굶어보세요.",
                                 "율무주를 조금씩 드셔 보세요.", "따뜻한 술 한 잔이 도움이 돼요.", "민간요법으로 쑥을 달여 드세요."])
def test_unsafe_care_sentence_dropped(bad: str) -> None:
    reply = f"설사로 불편하셨겠어요.\n· 증상: 설사\n{bad} 며칠 넘게 계속되면 내과 진료를 받아 보세요."
    out, n = MedicalLlmService.drop_unsafe_care_sentences(reply)
    assert n == 1 and bad not in out and "내과 진료를 받아 보세요." in out and "· 증상: 설사" in out


@pytest.mark.parametrize("ok", ["물을 자주 조금씩 마셔 주세요.", "술은 피하세요.", "미음이나 죽처럼 부드러운 음식을 드세요."])
def test_safe_care_sentence_kept(ok: str) -> None:
    assert MedicalLlmService.drop_unsafe_care_sentences(ok) == (ok, 0)


def test_unsafe_filter_only_on_care_question(env, monkeypatch) -> None:  # noqa: F811
    fasting = dict(DIET, advice="가급적 굶으세요. 내과 진료를 권해드리고, 사흘 넘게 계속되면 꼭 진료를 받아 보세요.")
    svc = MedicalLlmService(Settings(), _Loader(**fasting))
    assert "굶" not in svc.generate(CARE[0], {}, [], intent="medical_advice")["reply"]
    assert "굶" in svc.generate("어제부터 설사를 세 번 했어요", {}, [], intent="medical_advice")["reply"]  # 다른 질문은 손대지 않는다


# ---- ③ 고치기 → 진료과 ------------------------------------------------------------------------------------


def _edit_department(fake: FakeReservation) -> tuple[Any, dict[str, Any]]:
    orch = make_orchestrator(fake, FULL)
    card = _confirm_card(orch, fake)
    edit = _turn(orch, card, ui=("reservation_confirm", _value(card, "edit:")))
    return orch, _turn(orch, edit, ui=("reservation_edit", _value(edit, "department:")))


def test_edit_department_offers_only_this_hospitals_departments() -> None:
    fake = FakeReservation()
    fake.departments_by_hospital = {1: ["내과", "가정의학과"]}
    _, asked = _edit_department(fake)
    assert asked["ui"].field == "department"
    assert [o.value for o in asked["ui"].options] == ["내과", "가정의학과", OTHER_HOSPITAL_OPTION]
    assert [o.label for o in asked["ui"].options][-1] == "다른 병원에서 찾기"
    assert "소아청소년과" not in [o.value for o in asked["ui"].options]
    assert "강남메디내과" in asked["ui"].label
    assert fake.searched_regions == [] and "create" not in fake.names()  # 병원을 다시 찾지 않는다


def test_edit_department_keeps_hospital_and_date_and_reselects_time() -> None:
    fake = FakeReservation()
    fake.departments_by_hospital = {1: ["내과", "가정의학과"]}
    orch, asked = _edit_department(fake)
    picked = _turn(orch, asked, ui=("department", "가정의학과"))
    assert picked["ui"].field == "time" and picked["ui"].options[0].value.startswith(FUTURE)
    assert fake.slot_departments[-1] == "가정의학과" and fake.searched_regions == []
    card = _turn(orch, picked, ui=("time", f"{FUTURE} 10:00"))
    assert card["ui"].field == "reservation_confirm"
    for text in ("강남메디내과", "가정의학과", "10:00"):
        assert text in card["reply"]
    assert "14:00" not in card["reply"]
    _turn(orch, card, ui=("reservation_confirm", _value(card, "confirm:")))
    args = fake.calls[-1][1]
    assert fake.calls[-1][0] == "create" and args["hospital_id"] == 1 and args["department"] == "가정의학과"


def test_edit_department_other_hospital_goes_to_full_department_list() -> None:
    fake = FakeReservation()
    fake.departments_by_hospital = {1: ["내과"]}
    orch, asked = _edit_department(fake)
    other = _turn(orch, asked, ui=("department", OTHER_HOSPITAL_OPTION))
    assert other["ui"].field == "department"
    assert [o.value for o in other["ui"].options] == list(ReservationOrchestrator.STANDARD_DEPARTMENTS)
    picked = _turn(orch, other, ui=("department", "정형외과"))
    assert fake.searched_regions  # 이제 새 병원을 찾는다
    assert OTHER_HOSPITAL_OPTION not in picked["reply"]


def test_edit_department_falls_back_when_department_list_unavailable(monkeypatch) -> None:
    fake = FakeReservation()
    monkeypatch.setattr(fake, "list_departments", lambda *a, **k: {"httpStatus": 500, "data": None})
    _, asked = _edit_department(fake)
    # 진료과 없이 그 병원으로 가지 않는다 - 고치기 전처럼 전체 진료과 목록
    assert asked["ui"].field == "department"
    assert [o.value for o in asked["ui"].options] == list(ReservationOrchestrator.STANDARD_DEPARTMENTS)


def test_typed_department_change_still_clears_hospital_1028() -> None:
    """10/28 규칙(R-8 M-1) 회귀: 이어 쓰는 초안에서 진료과를 바꾸면 병원을 비운다 - 고치기 버튼 흐름과 따로."""
    draft = ReservationDraft(message="", hospital_name="강남메디내과", hospital_id=1, department="내과", reservation_date=FUTURE, reservation_time="14:00")
    before = _linked_fields(draft)
    draft.department = "정형외과"
    meta: dict[str, Any] = {}
    _clear_stale_links(before, draft, meta)
    assert draft.hospital_id is None and draft.hospital_name is None and draft.reservation_time is None


def test_department_first_set_with_hospital_is_not_a_change() -> None:
    draft = ReservationDraft(message="", hospital_name="강남메디내과", hospital_id=1, department=None, reservation_date=FUTURE)
    before = _linked_fields(draft)
    draft.department = "가정의학과"
    _clear_stale_links(before, draft, {})
    assert draft.hospital_id == 1 and draft.hospital_name == "강남메디내과"


# ---- ④ 응급 다음 턴 ---------------------------------------------------------------------------------------


def test_emergency_next_turn_not_bound_by_history(env, monkeypatch) -> None:  # noqa: F811
    _router(env, monkeypatch, route="symptom")
    real = MedicalLlmService(Settings(), _Loader(empathy="불편하셨겠어요.", symptoms="증상",
                                                 advice="내과 진료를 권해드려요. 사흘 넘게 계속되면 꼭 진료를 받아 보세요."))
    monkeypatch.setattr(env.llm, "generate", lambda m, c, h, **k: real.generate(m, c, h, **k))
    first = _ward(env, "머리가 아파요", "s-em")
    assert first.get("suggestedKind") == "answer"
    yes = _ward(env, "네", "s-em")
    assert yes["riskLevel"] == "high" and "119" in yes["reply"]
    dizzy = _ward(env, "어지러워요", "s-em")
    assert dizzy["riskLevel"] != "high" and dizzy["reply"].count("119") == 1 and _EMERGENCY_FOLLOWUP in dizzy["reply"]
    knee = _ward(env, "허리가 일주일째 아파요", "s-em")
    assert knee["riskLevel"] != "high" and _EMERGENCY_FOLLOWUP not in knee["reply"]


# ---- ⑤ 이어서 물어볼 질문 조건표 ----------------------------------------------------------------------------

ANSWER = {"reply": "내과 진료를 권해드려요.\n사흘 넘게 계속되면 꼭 진료를 받아 보세요.", "riskLevel": "medium",
          "intent": "medical_advice", "meta": {}}
SYMPTOM = {"route": "symptom", "emergency": False}
RED_LINE = "참기 힘들 만큼 아프거나 피를 토하면 바로 119에 전화하세요."


def _ok(result: dict, lines: tuple[str, ...] = (), route: dict | None = None, **kw) -> bool:
    return fq.should_suggest(result, route or SYMPTOM, "무릎이 아파요", emergency_text=kw.get("et", False),
                             self_harm_text=kw.get("st", False), safety_lines=lines)


def _with(line: str, **meta) -> dict:
    return {**ANSWER, "reply": f"{ANSWER['reply']}\n{line}", "meta": meta}


def test_fixed_safety_line_alone_does_not_block() -> None:
    assert _ok(_with(_EMERGENCY_FOLLOWUP, emergencyFollowup=True), (_EMERGENCY_FOLLOWUP,))
    assert _ok(_with(RED_LINE, followupResult=True), (RED_LINE,))


def test_other_119_still_blocks_even_with_safety_line() -> None:
    model_119 = {**ANSWER, "reply": f"어지러우면 바로 119에 연락하세요.\n{_EMERGENCY_FOLLOWUP}", "meta": {"emergencyFollowup": True}}
    assert not _ok(model_119, (_EMERGENCY_FOLLOWUP,))
    # 줄을 넘기지 않은 옛 호출은 지금처럼 막힌다
    assert not _ok(_with(_EMERGENCY_FOLLOWUP, emergencyFollowup=True))


@pytest.mark.parametrize("override", [
    {"riskLevel": "high"}, {"intent": "emergency_guidance"}, {"meta": {"selfHarm": True, "emergencyFollowup": True}},
    {"meta": {"crisisFollowup": True, "emergencyFollowup": True}}, {"meta": {"redFlag119": True, "emergencyFollowup": True}},
    {"meta": {"drugAlcohol": True, "emergencyFollowup": True}},
])
def test_emergency_crisis_red_flag_answers_stay_empty(override: dict) -> None:
    result = {**_with(_EMERGENCY_FOLLOWUP, emergencyFollowup=True), **override}
    assert not _ok(result, (_EMERGENCY_FOLLOWUP,))


def test_emergency_route_or_text_stays_empty() -> None:
    result = _with(_EMERGENCY_FOLLOWUP, emergencyFollowup=True)
    assert not _ok(result, (_EMERGENCY_FOLLOWUP,), route={"route": "symptom", "emergency": True})
    assert not _ok(result, (_EMERGENCY_FOLLOWUP,), et=True)
    assert not _ok(result, (_EMERGENCY_FOLLOWUP,), st=True)


def test_service_emergency_next_turn_gets_suggestions(env, monkeypatch) -> None:  # noqa: F811
    _router(env, monkeypatch, route="symptom")
    real = MedicalLlmService(Settings(), _Loader(empathy="불편하셨겠어요.", symptoms="증상",
                                                 advice="정형외과 진료를 권해드려요. 사흘 넘게 계속되면 꼭 진료를 받아 보세요."))
    monkeypatch.setattr(env.llm, "generate", lambda m, c, h, **k: real.generate(m, c, h, **k))
    verdicts: list[bool] = []

    def fake_suggest(settings, result, route_info, message, **k):  # 모델(Claude) 호출 없이 붙일지 판정만 본다
        verdicts.append(fq.should_suggest(result, route_info, message, emergency_text=k["emergency_text"],
                                          self_harm_text=k["self_harm_text"], safety_lines=k.get("safety_lines", ())))
        return []

    monkeypatch.setattr(fq, "suggested_questions", fake_suggest)
    _ward(env, "머리가 아파요", "s-sq")
    yes = _ward(env, "네", "s-sq")
    assert yes["riskLevel"] == "high" and verdicts[-1] is False  # 응급 답 자체는 지금처럼 빈 목록
    back = _ward(env, "허리가 일주일째 아파요", "s-sq")
    assert _EMERGENCY_FOLLOWUP in back["reply"] and verdicts[-1] is True


# ---- 기능 점검 F-1: 관리 질문은 되묻지 않고 바로 답 + 피보호자 답에 그 묶음의 119 기준 한 줄 ----------------------

from app.services import chat_keywords, chat_service as cs, followup_steps  # noqa: E402

CARE_WOULD_ASK = ["무릎 아플 때 집에서 할 수 있는 거 있어요?", "감기 걸렸을 때 뭐 먹어요?", "변비에 좋은 음식 알려줘"]


@pytest.mark.parametrize("text", CARE_WOULD_ASK)
def test_care_question_skips_followup(text: str) -> None:
    assert MedicalLlmService.needs_followup_question(text, skip_care=False)  # 관리 질문이 아니었다면 되물었을 말
    assert not MedicalLlmService.needs_followup_question(text)


def _fake_gen(env, monkeypatch, advice: str = "물을 자주 드세요. 사흘 넘게 계속되면 꼭 진료를 받아 보세요."):  # noqa: F811
    real = MedicalLlmService(Settings(), _Loader(empathy="불편하셨겠어요.", symptoms="증상", advice=advice))
    monkeypatch.setattr(env.llm, "generate", lambda m, c, h, **k: real.generate(m, c, h, **k))
    monkeypatch.setattr(cs, "_medgemma_enabled", lambda: True)


@pytest.mark.parametrize("text", CARE_WOULD_ASK)
def test_ward_care_question_answers_with_red_flag_line(env, monkeypatch, text: str) -> None:  # noqa: F811
    _router(env, monkeypatch, route="symptom")
    _fake_gen(env, monkeypatch)
    verdicts: list[bool] = []
    monkeypatch.setattr(fq, "suggested_questions", lambda s, r, ri, m, **k: verdicts.append(fq.should_suggest(
        r, ri, m, emergency_text=k["emergency_text"], self_harm_text=k["self_harm_text"], safety_lines=k.get("safety_lines", ()))) or [])
    out = _ward(env, text, f"s-f1-{text}")
    assert out.get("suggestedKind") != "answer" and "여쭤볼게요" not in out["reply"]  # 되묻기 아님
    line = followup_steps.skipped_red_flag_line(followup_steps.new_state(text, False))
    if line:
        assert line in out["reply"] and out["reply"].count("119") == 1
        assert verdicts == [True]  # 고정 안전 줄이라 이어서 물어볼 질문은 막지 않는다


def test_guardian_care_question_has_no_extra_line(env, monkeypatch) -> None:  # noqa: F811
    _router(env, monkeypatch, route="symptom")
    _fake_gen(env, monkeypatch)
    text = CARE_WOULD_ASK[0]
    out = env.svc.process_message(env.db, ChatRequest(userId="u1", message=text, sessionId="s-f1-g", context=ChatContext(role="GUARDIAN")))
    line = followup_steps.skipped_red_flag_line(followup_steps.new_state(text, False))
    assert "여쭤볼게요" not in out["reply"] and (line is None or line not in out["reply"])


def test_ward_plain_symptom_still_asks_step(env, monkeypatch) -> None:  # noqa: F811
    _router(env, monkeypatch, route="symptom")
    _fake_gen(env, monkeypatch)
    out = _ward(env, "무릎이 아파요", "s-f1-plain")
    assert out.get("suggestedKind") == "answer"  # 관리 질문이 아닌 짧은 증상은 지금처럼 단계 되묻기


# ---- 기능 점검 F-3: 라우터 실패 때 죽음·사라짐의 방법을 묻는 말은 위기 응답 ------------------------------------------

HOW_CRISIS = ["어떻게 하면 편하게 갈 수 있을까", "어떻게 하면 안 아프게 죽어요", "어떻게 하면 잠들고 안 깰 수 있어요",
              "다 끝내려면 어떻게 해야 해요", "어떻게 하면 사라질 수 있어요", "어떻게 하면 죽을 수 있어요", "죽는 방법 알려줘"]
HOW_NOT = ["아파 죽겠어요", "배가 아파서 죽을 것 같아요", "어떻게 하면 통증이 사라질까요", "어떻게 하면 기침이 사라져요",
           "어떻게 하면 병원에 편하게 갈 수 있어요", "숙제 다 끝내려면 어떻게 해야 해", "죽 먹는 방법", "어떻게 하면 죽을 맛있게 끓여요",
           "어떻게 하면 죽을 드실 수 있어요", "이 약 먹으면 죽어요?", "어떻게 하면 잠이 잘 와요"]


@pytest.mark.parametrize("text", HOW_CRISIS)
def test_fallback_self_harm_how(text: str) -> None:
    assert chat_keywords.fallback_self_harm_signals(text)


@pytest.mark.parametrize("text", HOW_NOT)
def test_fallback_self_harm_how_not_symptom_talk(text: str) -> None:
    assert not chat_keywords.fallback_self_harm_signals(text)


class _FailingRouter:
    def route(self, *a, **k):
        raise RuntimeError("router down")


@pytest.mark.parametrize("role", ["WARD", "GUARDIAN"])
def test_router_failure_how_crisis_gets_crisis_reply(env, monkeypatch, role: str) -> None:  # noqa: F811
    monkeypatch.setattr(cs, "get_chat_router_service", lambda: _FailingRouter())
    _fake_gen(env, monkeypatch)
    out = env.svc.process_message(env.db, ChatRequest(userId="u1", message=HOW_CRISIS[1], sessionId=f"s-f3-{role}",
                                                      context=ChatContext(role=role)))
    assert "109" in out["reply"] and out["suggestedQuestions"] == []
