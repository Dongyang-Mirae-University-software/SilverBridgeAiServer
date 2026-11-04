"""답 모양(2026-11-01 가독성): 핵심 한 줄·한 줄 한 문장·칸별 카드·줄 머리 이모지.

모양만 바뀌어야 한다 - 판정 값(위험도·119·응급 경고·진료과·약 + 술 빼기)과 문장(상투 문장 하나 말고)은 그대로.
합성 예문만 쓴다(실제 상담 없음)."""
from __future__ import annotations

import json
import re
from collections import Counter
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.config import Settings
from app.services import chat_keywords as kw
from app.services import chat_layout
from app.services import medical_llm_service as mls
from app.services.chat_service import _CRISIS_FOLLOWUP, _EMERGENCY_FOLLOWUP
from app.services.medical_llm_service import DRUG_ALCOHOL_SENTENCES, FIXED_SENTENCES, MedicalLlmService

EMOJI = re.compile(r"[\U0001F000-\U0001FAFF⌀-⏿☀-➿⬀-⯿]️?")
SPLIT = re.compile(r"(?<=[.!?。])\s+")
FILLER = "말씀해 주신 내용을 정리했어요."


class _Loader:
    def __init__(self, **fields: str) -> None:
        self.fields = {"empathy": "걱정되셨겠어요. 말씀해 주신 내용을 정리했어요.", "symptoms": "증상", "features": "",
                       "advice": "", "reply": "", **fields}
        self.state = SimpleNamespace(model_name="fake-medgemma")

    def ensure_loaded(self) -> None:
        return None

    def generate_structured_text(self, system_prompt: str, user_prompt: str, prefill: str = "", **kwargs) -> str:  # noqa: ARG002
        return json.dumps(self.fields, ensure_ascii=False)


def _gen(message: str, intent: str = "medical_advice", proxy: bool = False, **fields: str) -> dict[str, Any]:
    return MedicalLlmService(Settings(), _Loader(**fields)).generate(message, {}, [], intent=intent, proxy=proxy)


def _apply(result: dict[str, Any]) -> dict[str, Any]:
    return chat_layout.apply(dict(result), DRUG_ALCOHOL_SENTENCES)  # apply 는 맨 위 칸만 바꾼다


def _sentences(reply: str) -> list[str]:
    out = []
    for line in reply.split("\n"):
        line = line.strip()
        m = re.match(r"^·\s*(증상|특징)\s*:\s*(.+)$", line)
        if m:
            out.append(f"{m.group(1)}: {m.group(2).strip()}")
        elif line.startswith("· "):
            out.append(line[2:].strip())  # 되묻기 질문 줄은 한 줄 그대로
        else:
            out.extend(s.strip() for s in SPLIT.split(line) if s.strip())
    return out


def _new_sentences(reply: str) -> list[str]:
    return [kw.strip_layout_emoji(line).strip() for line in reply.split("\n") if line.strip()]


# 합성 예문 - 일반·되묻기 다음·약·약 + 술·응급·대리·위험 신호·이어지는 증상·일반 대화
CORPUS: list[tuple[str, dict[str, Any]]] = [
    ("무릎이 일주일째 아프고 계단 내려갈 때 더 아파요", dict(
        empathy="무릎이 아파 계단 다니기 힘드셨겠어요. 말씀해 주신 내용을 정리했어요.", symptoms="무릎 통증 (일주일째)",
        features="계단 내려갈 때 더 아픔", advice="정형외과 진료를 권해드려요. 무릎이 붓거나 열감이 있으면 꼭 진료를 받아 보세요.")),
    ("감기약 먹고 있는데 진통제 같이 먹어도 돼요?", dict(
        empathy="같이 드셔도 될지 걱정되셨겠어요.", symptoms="약 복용 문의 (감기약·진통제)",
        advice="약을 더 드시거나 바꾸기 전에 꼭 의사나 약사에게 물어봐 주세요. 다니시는 병원 진료 때 함께 여쭤보셔도 좋아요.")),
    ("혈압약 먹는데 오늘 저녁에 소주 마셔도 될까요?", dict(
        empathy="술자리가 있어 고민되셨겠어요.", symptoms="약 복용 문의 (혈압약·음주)",
        advice="술을 드시기 전에 다니시는 병원 의사나 약사에게 꼭 물어봐 주세요.")),
    ("당뇨약 먹는데 막걸리 한 잔 괜찮아요?", dict(
        empathy="걱정되셨겠어요.", symptoms="약 복용 문의", advice="한두 잔은 괜찮아요. 내과 진료 때 여쭤보세요.")),
    ("가슴이 답답하고 식은땀이 나요", dict(
        empathy="많이 불편하시겠어요.", symptoms="가슴 답답함, 식은땀", advice="내과 진료를 권해드려요.")),
    ("어제부터 기침이 계속 나고 가래가 있어요", dict(
        empathy="기침 때문에 힘드시겠어요.", symptoms="기침·가래 (어제부터)", features="계속됨",
        advice="내과 진료를 권해드려요. 숨이 차거나 열이 나면 꼭 진료를 받아 보세요.")),
    ("혈당이 60이에요 저혈당인 것 같아요", dict(
        empathy="놀라셨겠어요.", symptoms="저혈당", advice="내과 진료를 권해드려요.")),
    ("약을 깜빡하고 안 먹었어요", dict(empathy="걱정되셨겠어요.", symptoms="약 복용 문의", advice="")),
    ("관절염인데 무슨 운동이 좋아요?", dict(
        empathy="운동을 하고 싶으시군요.", symptoms="관절염 운동 문의", advice="정형외과 진료 때 맞는 운동을 여쭤보세요.")),
    ("허리가 아파요", dict(
        empathy="허리가 아프시군요. 많이 불편하시겠어요. 오래 앉아 계시기도 힘드시겠어요.", symptoms="허리 통증",
        features="앉아 있기 힘듦", advice="정형외과 진료를 권해드려요. 물을 자주 드시고 푹 쉬세요. 다리가 저리면 꼭 진료를 받으세요.")),
]
EMERGENCY_CORPUS = [
    "갑자기 가슴이 쥐어짜듯 아프고 식은땀이 나요",
    "엄마가 갑자기 쓰러지셨어요",
    "숨을 못 쉬겠어요",
    "피를 토했어요",
]


def _corpus_results() -> list[tuple[str, dict[str, Any]]]:
    out = [(m, _gen(m, **f)) for m, f in CORPUS]
    out += [(m, _gen(m, "emergency_guidance", empathy="많이 놀라셨겠어요.", symptoms="응급 증상")) for m in EMERGENCY_CORPUS]
    out += [(m, _gen(m, "medical_advice", proxy=True, **f)) for m, f in CORPUS[:3]]
    out.append(("안녕하세요", _gen("안녕하세요", "general_chat", reply="네, 안녕하세요. 무엇을 도와드릴까요?")))
    out.append(("어지러워요", MedicalLlmService.followup_question_response("어지러워요")))
    out.append(("어지러워요", MedicalLlmService.followup_question_response("어지러워요", proxy=True)))
    out.append(("죽고 싶어요", MedicalLlmService.self_harm_response()))
    out.append(("엄마가 죽고 싶대요", MedicalLlmService.self_harm_response(family=True)))
    return out


# ── 응급·자해·위기: 🚨 하나만, 첫 줄 119(위기는 109) ─────────────────────────────────────────────────

@pytest.mark.parametrize("message", EMERGENCY_CORPUS)
def test_emergency_answer_has_only_siren_and_119_first(message: str) -> None:
    out = _apply(_gen(message, "emergency_guidance", empathy="많이 놀라셨겠어요.", symptoms="응급 증상"))
    first = out["reply"].split("\n")[0]
    assert first.startswith("🚨 ") and "119" in first
    assert EMOJI.findall(out["reply"]) == ["🚨"]
    assert out["layout"]["headline"] == first and "전화" in first
    assert not out["layout"]["homeCare"] and not out["layout"]["visitHospitalIf"]


@pytest.mark.parametrize("family", [False, True])
def test_self_harm_answer_has_only_siren_and_119_first(family: bool) -> None:
    raw = MedicalLlmService.self_harm_response(family=family)
    out = _apply(raw)
    first = out["reply"].split("\n")[0]
    assert first.startswith("🚨 ") and "119" in first
    assert EMOJI.findall(out["reply"]) == ["🚨"]
    assert "109" in out["reply"]
    assert Counter(_new_sentences(out["reply"])) == Counter(_sentences(raw["reply"]))  # 문장은 그대로, 순서만


def test_crisis_followup_answer_has_only_siren_and_109_first() -> None:
    raw = _gen("손주가 놀러 와서 기분이 좋아요", "general_chat", reply="좋으시겠어요. 즐거운 시간 보내세요.")
    raw["reply"] = f"{_CRISIS_FOLLOWUP}\n{raw['reply']}"
    raw["meta"]["crisisFollowup"] = True
    out = _apply(raw)
    first = out["reply"].split("\n")[0]
    assert first.startswith("🚨 ") and "109" in first
    assert EMOJI.findall(out["reply"]) == ["🚨"]


def test_emergency_warning_is_split_per_sentence_with_same_text() -> None:
    raw = _gen("숨을 못 쉬겠어요", "emergency_guidance", symptoms="호흡 곤란")
    out = _apply(raw)
    assert len(out["emergencyWarning"]) >= 2
    assert " ".join(out["emergencyWarning"]) == " ".join(raw["emergencyWarning"])  # 내용 그대로, 문장만 나눔


# ── 줄마다 이모지 하나·줄 머리·허용 목록 ───────────────────────────────────────────────────────────

def test_every_line_has_at_most_one_allowed_emoji_at_head() -> None:
    for _, raw in _corpus_results():
        out = _apply(raw)
        for line in out["reply"].split("\n"):
            found = EMOJI.findall(line)
            assert len(found) <= 1, line
            if found:
                assert line.startswith(found[0] + " ")
                assert found[0].rstrip("️") in {e.rstrip("️") for e in kw.LAYOUT_EMOJI_ALLOWLIST}


def test_non_urgent_answers_never_use_siren() -> None:
    for _, raw in _corpus_results():
        out = _apply(raw)
        urgent = bool(out.get("emergencyWarning")) or out["meta"].get("selfHarm")
        if not urgent:
            assert "🚨" not in out["reply"]


def test_allowlist_has_about_twenty_and_no_faces() -> None:
    assert 15 <= len(kw.LAYOUT_EMOJI_ALLOWLIST) <= 22
    for emoji in kw.LAYOUT_EMOJI_ALLOWLIST:
        assert not (0x1F600 <= ord(emoji[0]) <= 0x1F64F)  # 웃음·얼굴 이모지 블록 없음


def test_strip_removes_every_allowed_emoji() -> None:
    for emoji in kw.LAYOUT_EMOJI_ALLOWLIST:
        assert kw.strip_layout_emoji(f"{emoji} 병원 진료를 받아 보세요.") == "병원 진료를 받아 보세요."


# ── 판정은 이모지·순서와 무관 ───────────────────────────────────────────────────────────────────────

JUDGE_KEYS = ("riskLevel", "recommendedAction", "reservationRequired", "intent", "engine", "summary", "suggestedQuestions")


def test_layout_does_not_touch_judgement_values() -> None:
    for _, raw in _corpus_results():
        out = _apply(raw)
        for key in JUDGE_KEYS:
            assert out.get(key) == raw.get(key), key
        assert bool(out["emergencyWarning"]) == bool(raw["emergencyWarning"])
        assert out["meta"] == raw["meta"]


def test_judgements_on_stripped_reply_match_old_reply() -> None:
    """이모지를 지운 새 답으로 다시 판정해도 옛 답과 같다(119·위험 신호·응급·자해·진료과·약 + 술 빼기·약 119 빼기)."""
    for message, raw in _corpus_results():
        old = raw["reply"]
        new = kw.strip_layout_emoji(_apply(raw)["reply"])
        assert ("119" in new) == ("119" in old)
        assert kw.has_red_flag(new) == kw.has_red_flag(old)
        assert MedicalLlmService.emergency_signals(new) == MedicalLlmService.emergency_signals(old) or \
            sorted(MedicalLlmService.emergency_signals(new)) == sorted(MedicalLlmService.emergency_signals(old))
        assert bool(MedicalLlmService.self_harm_signals(new)) == bool(MedicalLlmService.self_harm_signals(old))
        assert set(mls._ANSWER_DEPARTMENT.findall(new)) == set(mls._ANSWER_DEPARTMENT.findall(old))
        assert MedicalLlmService.needs_red_flag_sentence(message, new) == MedicalLlmService.needs_red_flag_sentence(message, old)
        assert MedicalLlmService.needs_clinic_sentence(message, new) == MedicalLlmService.needs_clinic_sentence(message, old)
        _, dropped_new, _ = MedicalLlmService.apply_drug_alcohol_rules(message, new)
        _, dropped_old, _ = MedicalLlmService.apply_drug_alcohol_rules(message, old)
        assert dropped_new == dropped_old
        assert MedicalLlmService.drop_medication_119(message, new)[1] == MedicalLlmService.drop_medication_119(message, old)[1]


def test_emoji_in_history_does_not_change_history_judgement() -> None:
    raw = MedicalLlmService.self_harm_response()
    new = _apply(raw)["reply"]
    history_new = [{"role": "user", "content": "죽고 싶어요"}, {"role": "assistant", "content": kw.strip_layout_emoji(new)}]
    history_old = [{"role": "user", "content": "죽고 싶어요"}, {"role": "assistant", "content": raw["reply"]}]
    assert MedicalLlmService.self_harm_in_history(history_new) == MedicalLlmService.self_harm_in_history(history_old)
    assert not EMOJI.search(kw.strip_layout_emoji(new))


# ── 문장은 버리지 않는다(상투 문장 하나만) / 고정 문장은 정한 자리로 ────────────────────────────────

def test_no_sentence_is_lost_except_filler() -> None:
    for _, raw in _corpus_results():
        out = _apply(raw)
        expected = Counter(s for s in _sentences(raw["reply"]) if s != FILLER)
        assert Counter(_new_sentences(out["reply"])) == expected
        assert FILLER not in out["reply"]


def _slot_of(layout: dict[str, Any], sentence: str) -> str:
    for key in ("headline", "body", "questions", "homeCare", "visitHospitalIf"):
        items = [layout[key]] if key == "headline" else layout[key]
        if any(item and kw.strip_layout_emoji(item).strip() == sentence for item in items):
            return key
    return "missing"


NON_URGENT_FIXED = [f for f in FIXED_SENTENCES if not (
    f.name.startswith("_EMERGENCY_ACTIONS.") or f.name in {"_EMERGENCY_REPLY", "_SELF_HARM_REPLY", "_SELF_HARM_REPLY_FAMILY",
                                                          "_SELF_HARM_WARNING", "_FOLLOWUP_INTRO", "_FOLLOWUP_CLOSING"}
)]


@pytest.mark.parametrize("fixed", NON_URGENT_FIXED, ids=lambda f: f.name)
def test_every_fixed_sentence_lands_in_its_slot(fixed) -> None:
    """고정 문장마다(본인·대리): 병원·의사·119 문장은 '이럴 땐 병원으로'(또는 핵심 한 줄), 약 + 술 설명은 본문,
    그 밖의 안내는 '집에서 이렇게'. 어떤 문장도 빠지지 않는다."""
    for text in filter(None, [str(fixed), fixed.proxy]):
        text = text.replace("{department}", "내과")
        raw = _gen("허리가 아파요", empathy="불편하시겠어요.", symptoms="허리 통증", advice="정형외과 진료를 권해드려요.")
        raw["reply"] = f"{raw['reply']}\n{text}"
        layout = _apply(raw)["layout"]
        for sentence in [s for s in SPLIT.split(text) if s.strip()]:
            slot = _slot_of(layout, sentence)
            if fixed.name.startswith("_DRUG_ALCOHOL_RULES."):
                assert slot in {"headline", "body", "homeCare", "visitHospitalIf"}, (fixed.name, sentence)
                if not kw.LAYOUT_HOSPITAL.search(sentence):
                    assert slot in {"body", "homeCare"}
            elif len([x for x in SPLIT.split(text) if x.strip()]) > 1:
                # 여러 문장 고정 문장은 한 카드에 묶인다 - 병원 문장이 하나라도 있으면 '이럴 땐 병원으로'
                group_hospital = any(kw.LAYOUT_HOSPITAL.search(x) for x in SPLIT.split(text))
                assert slot == ("visitHospitalIf" if group_hospital else "homeCare"), (fixed.name, sentence)
            elif kw.LAYOUT_HOSPITAL.search(sentence):
                assert slot in {"headline", "visitHospitalIf"}, (fixed.name, sentence)
            else:
                assert slot == "homeCare", (fixed.name, sentence)


def test_drug_alcohol_headline_is_the_consult_sentence() -> None:
    out = _apply(_gen("혈압약 먹는데 오늘 저녁에 소주 마셔도 될까요?", **CORPUS[2][1]))
    assert out["layout"]["headline"].startswith("🩺 ") and "상의" in out["layout"]["headline"]


def test_general_headline_is_department_advice() -> None:
    out = _apply(_gen(CORPUS[0][0], **CORPUS[0][1]))
    assert out["layout"]["headline"] == "🏥 정형외과 진료를 권해드려요."
    assert out["reply"].split("\n")[0] == out["layout"]["headline"]


def test_red_flag_119_sentence_goes_to_hospital_card_with_warning_emoji() -> None:
    out = _apply(_gen(CORPUS[4][0], **CORPUS[4][1]))
    hits = [x for x in out["layout"]["visitHospitalIf"] if "119" in x]
    assert hits and all(x.startswith("⚠️ ") for x in hits)
    assert "119" not in (out["layout"]["headline"] or "")


def test_emergency_followup_sentence_goes_to_hospital_card() -> None:
    raw = _gen("이제 괜찮아졌어요", "general_chat", reply="다행이에요.")
    raw["reply"] = f"{raw['reply']}\n{_EMERGENCY_FOLLOWUP}"
    out = _apply(raw)
    assert any(kw.strip_layout_emoji(x) == _EMERGENCY_FOLLOWUP for x in out["layout"]["visitHospitalIf"])


def test_followup_layout_questions_and_department_headline() -> None:
    out = _apply(MedicalLlmService.followup_question_response("어지러워요"))
    layout = out["layout"]
    assert layout["headline"].startswith("🏥 ") and "신경과" in layout["headline"]
    assert len(layout["questions"]) == 3 and all(q.startswith("❓ ") for q in layout["questions"])
    assert "(참을 만한지, 점점 심해지는지)" in layout["questions"][1]  # 질문 줄은 쪼개지 않는다
    assert any("갑자기 심해지면" in x for x in layout["visitHospitalIf"])


# ── 본문 줄 수 상한 ─────────────────────────────────────────────────────────────────────────────────

def test_body_is_at_most_four_lines_one_sentence_each() -> None:
    for _, raw in _corpus_results():
        out = _apply(raw)
        layout = out["layout"]
        if out.get("emergencyWarning") or out["meta"].get("selfHarm"):
            continue
        assert len(layout["body"]) <= chat_layout.BODY_MAX_LINES
        for key in ("body", "homeCare", "visitHospitalIf"):
            for line in layout[key]:
                text = kw.strip_layout_emoji(line)
                assert len([s for s in SPLIT.split(text) if s.strip()]) == 1, line


def test_overflow_body_moves_to_home_card_not_dropped() -> None:
    raw = _gen("허리가 아파요", empathy="허리가 아프시군요. 많이 불편하시겠어요. 오래 앉아 계시기도 힘드시겠어요. 걱정되시죠.",
               symptoms="허리 통증", features="앉아 있기 힘듦", advice="정형외과 진료를 권해드려요.")
    out = _apply(raw)
    assert len(out["layout"]["body"]) == chat_layout.BODY_MAX_LINES
    assert any("걱정되시죠" in x for x in out["layout"]["homeCare"])


# ── 빈 칸·옛 형식·내부 값 ───────────────────────────────────────────────────────────────────────────

def test_empty_cards_are_empty_lists_and_legacy_list_fields_stay_empty() -> None:
    out = _apply(_gen("안녕하세요", "general_chat", reply="네, 안녕하세요."))
    assert out["layout"]["homeCare"] == [] and out["layout"]["visitHospitalIf"] == [] and out["layout"]["headline"] is None
    for _, raw in _corpus_results():
        out = _apply(raw)
        assert out["possibleCauses"] == [] and out["homeCare"] == [] and out["visitHospitalIf"] == []  # 옛 FE 중복 방지


@pytest.mark.parametrize("result", [
    {"reply": "날짜를 골라 주세요.", "intent": "hospital_reservation", "type": "ui", "ui": {"field": "date"}},
    {"reply": "예약했어요.", "intent": "hospital_reservation", "type": "tool_result", "tool": "make_appointment"},
    {"reply": "어느 병원으로 할까요?", "intent": "hospital_reservation", "type": "message"},
])
def test_reservation_answers_are_untouched(result: dict[str, Any]) -> None:
    out = _apply(result)
    assert out["layout"] is None and out["reply"] == result["reply"]


def test_layout_parts_never_leave_apply() -> None:
    raw = _gen(CORPUS[0][0], **CORPUS[0][1])
    assert "layoutParts" in raw
    assert "layoutParts" not in _apply(raw)


def test_old_log_without_layout_reads_back_unchanged(tmp_path) -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.database.base import Base
    from app.models.chat_log import ChatLog
    from app.services.chat_service import ChatService

    engine = create_engine(f"sqlite:///{tmp_path}/t.db")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    db.add(ChatLog(chat_no="CHAT-old", user_id="u1", message="m", context_json=json.dumps({"emergencyWarning": []}),
                   reply="옛 답\n· 증상: 기침", risk_level="low", recommended_action="home_monitoring", reservation_required=False))
    db.commit()
    log = ChatService.list_logs(db, "u1")[0] if hasattr(ChatService, "list_logs") else None
    detail = ChatService.get_log(db, 1, "u1")
    assert detail["layout"] is None and detail["reply"] == "옛 답\n· 증상: 기침"
    if log is not None:
        assert log["layout"] is None


# ── 2026-11-01 점검 반영: 모델이 쓴 이모지·따옴표 안 낱말·줄바꿈·중복 키 ───────────────────────────────

FACE = "😊"


def _with_model_emoji(raw: dict[str, Any]) -> dict[str, Any]:
    """모델이 문장 안·끝에 이모지를 쓴 답(합성). 첫 문장 뒤와 줄 끝에 넣는다."""
    out = dict(raw)
    lines = raw["reply"].split("\n")
    out["reply"] = "\n".join(
        (line.replace(". ", f". {FACE} ", 1) + f" {FACE}") if line.strip() else line for line in lines
    )
    return out


def test_model_emoji_is_removed_so_one_allowed_emoji_per_line() -> None:
    """M-1: 모델 문장 속 이모지가 남아 한 줄 두 개·허용 목록 밖·응급 답 🚨 외 이모지가 되지 않는다."""
    for _, raw in _corpus_results():
        out = _apply(_with_model_emoji(raw))
        assert FACE not in out["reply"]
        assert FACE not in json.dumps(out["layout"], ensure_ascii=False)
        urgent = bool(out.get("emergencyWarning")) or out["meta"].get("selfHarm")
        for line in out["reply"].split("\n"):
            found = EMOJI.findall(line)
            assert len(found) <= 1, line
            if found:
                assert line.startswith(found[0] + " ")
                assert found[0].rstrip("️") in {e.rstrip("️") for e in kw.LAYOUT_EMOJI_ALLOWLIST}
                if urgent:
                    assert found[0] == kw.LAYOUT_EMERGENCY_EMOJI


def test_model_emoji_removal_keeps_every_sentence_and_119() -> None:
    for _, raw in _corpus_results():
        out = _apply(_with_model_emoji(raw))
        clean = _apply(raw)
        assert out["reply"] == clean["reply"]  # 이모지만 빠지고 나머지는 이모지 없던 답과 같다
        assert out["layout"] == clean["layout"]


def test_strip_keeps_line_breaks_after_line_end_emoji() -> None:
    """L-1: 줄 끝 이모지 뒤 줄바꿈을 먹어 다음 줄과 합치지 않는다."""
    assert kw.strip_layout_emoji(f"물을 드세요 {FACE}\n쉬세요.") == "물을 드세요 \n쉬세요."
    assert kw.strip_layout_emoji("💧 물을 드세요.\n🛏️ 쉬세요.").split("\n") == ["물을 드세요.", "쉬세요."]


def test_quoted_hospital_word_does_not_become_headline() -> None:
    """M-2: "무슨 과로 가야 해?" 규칙 답 - 진료과 안내가 첫 문장이다.
    2026-11-04부터 따옴표 안내("병원 찾아줘"라고 말씀해 주세요) 대신 버튼(ui)을 붙여, ui 가 있는 답처럼 layout 없이 문장 그대로 나간다."""
    from app.services.chat_service import ChatService

    history = [{"role": "user", "content": "무릎이 아파요"}, {"role": "assistant", "content": "정형외과 진료를 권해드려요."}]
    raw = ChatService._department_question_reply("무슨 과로 가야 해?", history)
    assert raw is not None
    out = _apply(raw)
    assert out["layout"] is None and out["reply"] == raw["reply"]
    assert out["reply"].startswith("정형외과로 가시면 돼요.") and '"' not in out["reply"]


def test_quoted_word_still_counts_outside_quotes() -> None:
    """따옴표 안만 뺀다 - 같은 문장의 따옴표 밖 병원 낱말은 그대로 병원 문장이다."""
    sentence = '"병원은 나중에"라고 미루지 마시고 다리가 저리면 진료를 받아 보세요.'
    out = _apply(_gen("허리가 아파요", empathy="불편하시겠어요.", symptoms="허리 통증",
                      advice=f"정형외과 진료를 권해드려요. {sentence}"))
    assert out["layout"]["headline"] == "🏥 정형외과 진료를 권해드려요."
    assert any(kw.strip_layout_emoji(x) == sentence for x in out["layout"]["visitHospitalIf"])


def test_chat_service_dicts_have_no_duplicate_keys() -> None:
    """L-2: 기록 응답 dict 에 같은 키가 두 번 들어가지 않는다(뒤 값이 앞 값을 조용히 덮는다)."""
    import ast
    from pathlib import Path

    tree = ast.parse(Path("app/services/chat_service.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
            dup = {k for k in keys if keys.count(k) > 1}
            assert not dup, (node.lineno, dup)
