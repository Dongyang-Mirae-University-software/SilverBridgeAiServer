"""자해 경고 박스 가족용 문구(2026-11-01, 점검 문서 9절 남은 한계 6번 - 사용자 결정).

가족이 주어인 자해 문장이면 본문(_SELF_HARM_REPLY_FAMILY)과 경고 박스(_SELF_HARM_WARNING 의 가족용)가 함께 가족용이다.
고르는 기준은 본문과 같은 about_family 하나다(wardId·proxy_subject 와 상관없음). 화자 주어("제가"·"내가")가 있으면
화자 본인의 위기로 보고 본인용이다. 본인용 문구는 바뀌지 않는다. 답 모양(🚨 119 첫 줄·이모지)은 #151·#154 그대로다."""
from __future__ import annotations

from typing import Any

import pytest

import app.services.medical_llm_service as mls
from app.schemas.chat_schema import ChatContext, ChatRequest
from app.services import chat_keywords as kw
from app.services.chat_layout import split_warning
from app.services.medical_llm_service import MedicalLlmService
from tests.test_chat_crisis_followup import _FlagRouter, _last_meta
from tests.test_chat_router_followup import env  # noqa: F401 - env 는 픽스처

OWN_WARNING = "지금 위험하다고 느끼시면 바로 119에 전화하세요. 자살예방 상담전화 109는 24시간 받습니다."
FAMILY_WARNING = ["가족분이 지금 위험해 보이면 바로 119에 전화하세요.", "자살예방 상담전화 109는 가족도 24시간 상담할 수 있어요."]
OWN_SPLIT = split_warning([OWN_WARNING])


def _send(env, message: str, session: str = "s1", ward_id: str | None = None) -> dict[str, Any]:
    payload = ChatRequest(userId="user-1", message=message, sessionId=session, context=ChatContext(name="홍길동", wardId=ward_id))
    return env.svc.process_message(env.db, payload)


def _assert_shape(out: dict[str, Any]) -> None:
    """답 모양 규칙(#151·#154): 첫 줄 🚨 + 119, 🚨 말고 다른 줄 머리 이모지 없음, 박스 문장에 이모지 없음."""
    lines = out["reply"].split("\n")
    assert lines[0].startswith("🚨") and "119" in lines[0]
    assert sum(line.count("🚨") for line in lines) == 1
    assert all(kw.strip_layout_emoji(line) == line for line in lines[1:])
    assert all(kw.strip_layout_emoji(w) == w for w in out["emergencyWarning"])


# ---- 고정 문장 표 -------------------------------------------------------------------------------------------


def test_own_warning_text_is_byte_identical() -> None:
    assert str(mls._SELF_HARM_WARNING) == OWN_WARNING
    assert mls._SELF_HARM_WARNING.pick(False) == OWN_WARNING


def test_family_warning_in_table() -> None:
    assert mls._SELF_HARM_WARNING.subject == "self"
    assert split_warning([mls._SELF_HARM_WARNING.proxy]) == FAMILY_WARNING


@pytest.mark.parametrize("family, warning, reply", [
    (True, FAMILY_WARNING, mls._SELF_HARM_REPLY_FAMILY),
    (False, OWN_SPLIT, mls._SELF_HARM_REPLY),
])
def test_self_harm_response_picks_box_with_body(family: bool, warning: list[str], reply: str) -> None:
    out = MedicalLlmService.self_harm_response(family=family)
    assert out["reply"] == reply and split_warning(out["emergencyWarning"]) == warning
    assert out["riskLevel"] == "high" and out["meta"]["selfHarmFamily"] is family


# ---- 가족 주어 판정(about_family) ---------------------------------------------------------------------------


@pytest.mark.parametrize("message", [
    "엄마가 죽고 싶다고 해요",
    "아버지께서 죽고 싶다고 하세요",
    "엄마가 죽고 싶다고 저한테 말했어요",  # "저한테"는 주어가 아니다
    "남편이 죽고 싶대요. 제가 어떻게 해야 해요?",  # 할 일을 묻는 화자는 돌보는 사람
    "제가 보기엔 어머니가 자해를 하시는 것 같아요",
])
def test_family_subject(message: str) -> None:
    assert MedicalLlmService.about_family(message) is True


@pytest.mark.parametrize("message", [
    "죽고 싶어요",
    "남편이 때려서 제가 죽고 싶어요",  # 화자 본인의 위기
    "엄마가 잔소리를 해서 저는 죽고 싶어요",
    "아내가 떠나서 나는 죽고 싶다",
    "딸이 죽고 싶대요",  # 가족용 호칭 범위는 넓히지 않았다(대리 판정 범위와 다름)
])
def test_speaker_or_non_family_subject_is_own(message: str) -> None:
    assert MedicalLlmService.about_family(message) is False


# ---- 대화 경로(키워드·폴백·위기 다음 턴·대리 상담 화면) ------------------------------------------------------


@pytest.mark.parametrize("message, ward_id, warning", [
    ("엄마가 죽고 싶다고 해요", None, FAMILY_WARNING),
    ("죽고 싶어요", None, OWN_SPLIT),
    ("남편이 때려서 제가 죽고 싶어요", None, OWN_SPLIT),
    ("엄마가 죽고 싶다고 해요", "ward-1", FAMILY_WARNING),  # 대리 상담 화면 + 가족 주어
    ("죽고 싶대요", "ward-1", OWN_SPLIT),  # 대리 상담 화면이어도 기준은 about_family(본문과 같음)
])
def test_chat_path_box_follows_body(env, message: str, ward_id: str | None, warning: list[str]) -> None:
    out = _send(env, message, ward_id=ward_id)
    family = warning == FAMILY_WARNING
    assert out["emergencyWarning"] == warning
    assert _last_meta(env)["selfHarmFamily"] is family
    body = kw.strip_layout_emoji(out["reply"]).replace("\n", " ")
    assert ("가족도 24시간" in body) is family  # 본문도 같은 쪽
    _assert_shape(out)
    assert env.llm.calls == 0


def test_fallback_path_uses_same_choice(env) -> None:
    out = env.svc._fallback_reply("엄마가 죽고 싶다고 해요", [], emergency=True)
    assert split_warning(out["emergencyWarning"]) == FAMILY_WARNING
    own = env.svc._fallback_reply("죽고 싶어요", [], emergency=True)
    assert split_warning(own["emergencyWarning"]) == OWN_SPLIT


def test_crisis_continuation_judges_this_sentence(env) -> None:
    first = _send(env, "엄마가 죽고 싶다고 해요")
    assert first["emergencyWarning"] == FAMILY_WARNING
    env.router = _FlagRouter(self_harm=True, summary="위기 지속")
    family_turn = _send(env, "엄마가 아직도 너무 힘들어하세요")
    assert family_turn["emergencyWarning"] == FAMILY_WARNING and _last_meta(env)["selfHarmTrigger"] == "router"
    # 이어지는 턴도 이번 문장만 본다 - 주어가 없으면 본문·박스 모두 본인용(본문과 같은 기준)
    plain_turn = _send(env, "너무 무서워요 어떡하죠")
    meta = _last_meta(env)
    assert plain_turn["emergencyWarning"] == OWN_SPLIT and meta["selfHarm"] is True and meta["selfHarmFamily"] is False
    _assert_shape(family_turn)
    _assert_shape(plain_turn)
