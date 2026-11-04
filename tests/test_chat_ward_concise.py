"""2026-11-04 피보호자(WARD) 본인 답 간결화.

- WARD 일반 답: 공감·증상·특징 줄을 빼고 모델 문장은 핵심 줄 포함 3문장·120자까지.
- 지키는 문장: 119·109 안내, 서버 고정 문장(약·약 + 술·진료 권유 보장·위험 신호), 과 안내 핵심 줄, 되묻기 질문·답 칩.
- GUARDIAN(대리 상담 포함)·role 없음: 프롬프트·답 모두 지금과 같다.
합성 예문만 쓴다(실제 상담 없음)."""
from __future__ import annotations

import re
from typing import Any

import pytest

from app.core.config import Settings
from app.prompts.medical_chat_prompts import _WARD_CONCISE_LINE, build_user_prompt
from app.schemas.chat_schema import ChatContext, ChatRequest
from app.services import chat_layout
from app.services.chat_service import _EMERGENCY_FOLLOWUP, ChatService
from app.services.medical_llm_service import DRUG_ALCOHOL_SENTENCES, FIXED_SENTENCES, MedicalLlmService
from tests.test_chat_history_server_side import _router
from tests.test_chat_reply_layout import _Loader, _gen
from tests.test_reservation_confirm_chat_service import env  # noqa: F401

EMOJI_HEAD = re.compile(r"^\S+\s")
SPLIT = re.compile(r"(?<=[.!?。])\s+")

_BACK = dict(
    empathy="허리가 아파 오래 앉아 계시기 힘드셨겠어요. 말씀해 주신 내용을 정리했어요.",
    symptoms="허리 통증 (일주일째)",
    features="오래 앉아 있기 힘듦",
    advice="정형외과 진료를 권해드려요. 허리 통증이 계속되거나 다리가 저려 오면 꼭 진료를 받아 보세요.",
)


def _copy(result: dict[str, Any]) -> dict[str, Any]:
    # apply 는 reply·layout·meta·emergencyWarning 만 바꾼다 - 그 칸만 따로 복사한다(FixedSentence 는 deepcopy 불가).
    return {**result, "meta": dict(result.get("meta") or {}), "emergencyWarning": list(result.get("emergencyWarning") or [])}


def _both(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    guardian = chat_layout.apply(_copy(result), DRUG_ALCOHOL_SENTENCES)
    ward = chat_layout.apply(_copy(result), DRUG_ALCOHOL_SENTENCES, concise=True)
    return guardian, ward


def _lines(reply: str) -> list[str]:
    return [EMOJI_HEAD.sub("", x, count=1) if x[:1] and not x[:1].isalnum() else x for x in reply.split("\n") if x.strip()]


def _fixed_sentences() -> list[str]:
    out = []
    for f in FIXED_SENTENCES:
        for text in filter(None, (str(f), f.proxy)):
            out.extend(s for s in SPLIT.split(text) if s and "{department}" not in s)
    return out


def _model_lines(reply: str) -> list[str]:
    fixed = _fixed_sentences()
    return [x for x in _lines(reply) if "119" not in x and "109" not in x and x not in fixed
            and not any(x in d for d in DRUG_ALCOHOL_SENTENCES)
            and not re.match(r"^말씀하신 상태라면 미루지 말고 [가-힣]+ 진료를 받아 보세요\.$", x)]


# ---- 프롬프트 ------------------------------------------------------------------------------------------------


def _prompt(**kw) -> str:
    return build_user_prompt(message="허리가 아파요", user_context={"age": 80}, history_text="(없음)",
                             intent="medical_advice", emergency_hint=False, department_hint="정형외과", **kw)


def test_guardian_prompt_unchanged() -> None:
    base = _prompt()
    assert _prompt(concise=False) == base
    assert _WARD_CONCISE_LINE.strip() not in base
    assert base.endswith("(예약 안내와 같은 과여야 한다).\n")  # 마지막 지시 줄 뒤에 아무것도 붙지 않는다


def test_ward_prompt_adds_one_line_only() -> None:
    base, ward = _prompt(), _prompt(concise=True)
    assert ward == base + _WARD_CONCISE_LINE
    assert "쉬운 말" in ward and "괄호" in ward


# ---- 답 모양: 일반 상담 ----------------------------------------------------------------------------------------


def test_ward_general_drops_empathy_and_summary_keeps_department_headline() -> None:
    guardian, ward = _both(_gen("허리가 일주일째 아파요", **_BACK))
    assert ward["layout"]["headline"].endswith("정형외과 진료를 권해드려요.")
    assert ward["layout"]["headline"] == guardian["layout"]["headline"]
    text = ward["reply"]
    assert "힘드셨겠어요" not in text and "증상:" not in text and "특징:" not in text
    assert "힘드셨겠어요" in guardian["reply"] and "증상:" in guardian["reply"]
    assert len(text) < len(guardian["reply"])
    assert ward["meta"]["wardConcise"]["kind"] == "general"
    assert ward["meta"]["wardConcise"]["dropped"] >= 3
    assert "wardConcise" not in (guardian.get("meta") or {})


def test_ward_model_sentences_within_budget() -> None:
    long_reply = "천만에요. 언제든 편하게 물어봐 주세요. 오늘도 건강하게 지내세요. 물도 자주 드시고요. 산책도 좋아요."
    guardian, ward = _both(_gen("고마워요", intent="general_chat", reply=long_reply))
    assert len(_lines(guardian["reply"])) == 5
    model = _model_lines(ward["reply"])
    assert len(model) <= chat_layout.WARD_MODEL_MAX_SENTENCES
    assert sum(len(x) for x in model) <= chat_layout.WARD_MODEL_MAX_CHARS
    assert _lines(ward["reply"])[0] == "천만에요."  # 앞에서부터 남긴다


def test_ward_keeps_long_first_sentence_when_no_headline() -> None:
    first = "가" * 150 + "요."
    _, ward = _both(_gen("고마워요", intent="general_chat", reply=f"{first} 두 번째 문장이에요."))
    assert _lines(ward["reply"]) == [first]


@pytest.mark.parametrize("message", ["당뇨약 먹는데 술 마셔도 돼요?", "혈압약 먹는데 술 한잔 해도 돼요?"])
def test_ward_keeps_drug_alcohol_sentences(message: str) -> None:
    guardian, ward = _both(_gen(message, symptoms="약 복용 문의", advice="의사나 약사에게 물어봐 주세요."))
    pool = [s for d in DRUG_ALCOHOL_SENTENCES for s in SPLIT.split(str(d))]
    g = [x for x in _lines(guardian["reply"]) if x in pool]
    assert g, "시험 예문이 약 + 술 규칙에 걸려야 한다"
    assert [x for x in _lines(ward["reply"]) if x in pool] == g


def test_ward_keeps_drug_direct_fixed_sentences() -> None:
    guardian, ward = _both(_gen("혈압약을 깜빡하고 안 먹었어요", symptoms="약 복용 문의 (혈압약)",
                                advice="약을 더 드시기 전에 의사나 약사에게 물어봐 주세요."))
    fixed = _fixed_sentences()
    g = [x for x in _lines(guardian["reply"]) if x in fixed]
    assert g, "시험 예문이 약 직접 안내 고정 문장에 걸려야 한다"
    assert [x for x in _lines(ward["reply"]) if x in fixed] == g


def test_ward_keeps_red_flag_119_sentence() -> None:
    result = _gen("기침이 나고 가슴이 답답해요", symptoms="기침", advice="내과 진료를 권해드려요.")
    guardian, ward = _both(result)
    g119 = [x for x in _lines(guardian["reply"]) if "119" in x]
    assert [x for x in _lines(ward["reply"]) if "119" in x] == g119


def test_ward_keeps_emergency_followup_line() -> None:
    result = _gen("허리가 일주일째 아파요", **_BACK)
    result["reply"] = f"{result['reply']}\n{_EMERGENCY_FOLLOWUP}"
    _, ward = _both(result)
    assert _EMERGENCY_FOLLOWUP in _lines(ward["reply"])


# ---- 답 모양: 응급·위기·되묻기·예약 ---------------------------------------------------------------------------------


def test_ward_emergency_keeps_119_actions_and_warning_box() -> None:
    result = _gen("아버지가 갑자기 쓰러져서 숨을 안 쉬어요", intent="emergency_guidance",
                  empathy="많이 놀라셨겠어요.", symptoms="쓰러짐, 호흡 없음")
    guardian, ward = _both(result)
    assert ward["layout"]["headline"] == guardian["layout"]["headline"]
    assert "119" in ward["layout"]["headline"]
    assert ward["emergencyWarning"] == guardian["emergencyWarning"] and ward["emergencyWarning"]
    g_server = [x for x in guardian["layout"]["body"] if "놀라셨겠어요" not in x and not x.startswith("증상:")]
    assert ward["layout"]["body"] == g_server
    assert "놀라셨겠어요" not in ward["reply"] and "증상:" not in ward["reply"]
    assert ward["riskLevel"] == guardian["riskLevel"] == "high"


def test_ward_crisis_and_followup_unchanged() -> None:
    crisis = MedicalLlmService.self_harm_response()
    g, w = _both(crisis)
    assert w["reply"] == g["reply"] and w["layout"] == g["layout"] and w["emergencyWarning"] == g["emergencyWarning"]
    follow = MedicalLlmService.followup_question_response("머리가 아파요")
    g, w = _both(follow)
    assert w["reply"] == g["reply"] and w["layout"] == g["layout"]


def test_ward_reservation_answer_untouched() -> None:
    result = {"reply": "어느 병원으로 예약할까요?", "intent": "hospital_reservation", "type": "message", "ui": None, "tool": None,
              "riskLevel": "low", "emergencyWarning": [], "meta": {}}
    _, ward = _both(result)
    assert ward["reply"] == "어느 병원으로 예약할까요?" and ward["layout"] is None
    assert "wardConcise" not in ward["meta"]


# ---- 서비스: role 분기 ---------------------------------------------------------------------------------------


def _spy(env, monkeypatch) -> list[bool]:  # noqa: F811
    seen: list[Any] = []
    real = MedicalLlmService(Settings(), _Loader(**_BACK))

    def generate(message, ctx, history, **kwargs):
        seen.append(kwargs.get("concise", "absent"))
        return real.generate(message, ctx, history, **kwargs)

    monkeypatch.setattr(env.llm, "generate", generate)
    return seen


def _send(env, session: str, **ctx):  # noqa: F811
    payload = ChatRequest(userId="user-1", message="허리가 일주일째 아파요", sessionId=session, context=ChatContext(**ctx))
    return env.svc.process_message(env.db, payload)


@pytest.mark.parametrize("role", ["WARD", "ward", " Ward "])
def test_service_ward_role_is_concise(env, monkeypatch, role: str) -> None:  # noqa: F811
    _router(env, monkeypatch, route="symptom")
    seen = _spy(env, monkeypatch)
    out = _send(env, "s-ward", role=role)
    assert seen == [True]
    assert "힘드셨겠어요" not in out["reply"] and "증상:" not in out["reply"]
    assert out["layout"]["headline"].endswith("정형외과 진료를 권해드려요.")


@pytest.mark.parametrize("ctx", [dict(role="GUARDIAN"), dict(role="GUARDIAN", wardId="ward-9"), dict(role="ADMIN"), {}])
def test_service_non_ward_unchanged(env, monkeypatch, ctx: dict) -> None:  # noqa: F811
    _router(env, monkeypatch, route="symptom")
    seen = _spy(env, monkeypatch)
    out = _send(env, f"s-{len(ctx)}-{ctx.get('role')}", **ctx)
    assert seen == ["absent"]  # 보호자 쪽 호출에는 concise 인자 자체가 없다
    assert "힘드셨겠어요" in out["reply"] and "증상:" in out["reply"]


def test_is_ward_reader() -> None:
    assert ChatService._is_ward_reader({"role": "WARD"})
    assert not ChatService._is_ward_reader({"role": "GUARDIAN"})
    assert not ChatService._is_ward_reader({})
