"""긴급 말풍선 문장 중복 제거(2026-11-12 화면 확인).

FE 는 긴급 답을 핵심 줄 + 경고 박스 목록(emergencyWarning) + 본문(layout.body) 순으로 그린다. 단계 되묻기 119 규칙 답·생성 실패 대체
답은 본문이 경고 박스 문장 그 자체라 "지금은 응급 상황일 수 있어요."·"혼자 계시면 문을 열어 두고…"가 두 번 보였다.
본문에서 경고 박스와 같은 문장만 뺀다(표시 전용 - reply·경고 박스·핵심 줄·판정은 그대로).
"""
from __future__ import annotations

from app.services import chat_layout
from app.services.medical_llm_service import MedicalLlmService as M
from tests.test_chat_sleeping_pill_breath_lifestyle import _gen
from tests.test_followup_steps import _send, env, flow  # noqa: F401


def _warning_sentences(out: dict) -> set[str]:
    return set(out.get("emergencyWarning") or [])


def test_step_119_body_does_not_repeat_warning(flow) -> None:
    _send(flow, "무릎이 아파요", session="d1")
    out = _send(flow, "가슴이 답답하고 안 나아요", session="d1")
    assert out["riskLevel"] == "high"
    assert len(out["emergencyWarning"]) == 3
    assert not set(out["layout"]["body"]) & _warning_sentences(out)
    assert out["layout"]["headline"] and out["layout"]["urgent"] is True
    # 답 본문(기록·이력)은 그대로 - 경고 문장이 남아 있다.
    assert "혼자 계시면 문을 열어 두고" in out["reply"]


def test_step_119_action_line_stays_in_body() -> None:
    # 증상 유형 행동 줄(가슴 통증 등)은 경고 박스에 없는 문장이라 본문에 남는다.
    result = M.followup_red_flag_emergency("무릎이 아파요 가슴이 조이고 아파요", False)
    chat_layout.apply(result, concise=True, emoji=False)
    extra = [line for line in result["layout"]["body"] if line not in result["emergencyWarning"]]
    assert result["layout"]["body"] == extra
    assert any("운전" in line or "혼자" in line for line in result["layout"]["body"] + [result["reply"]])


def test_generated_emergency_body_unchanged() -> None:
    # 정상 생성 응급 답은 본문이 다른 문장이라 그대로다.
    result = chat_layout.apply(_gen("가슴이 조이고 숨이 차요", intent="emergency_guidance"), concise=True, emoji=False)
    assert result["layout"]["body"]
    assert not set(result["layout"]["body"]) & set(result["emergencyWarning"])


def test_emoji_layout_also_deduped() -> None:
    result = M.followup_red_flag_emergency("무릎이 아파요", False)
    chat_layout.apply(result, concise=True, emoji=True)
    plain = {chat_layout._without_emoji(line).strip() for line in result["layout"]["body"]}
    assert not plain & set(result["emergencyWarning"])


def test_non_emergency_layout_untouched() -> None:
    result = chat_layout.apply(_gen("무릎이 아파요"), concise=True, emoji=False)
    assert "urgent" not in result["layout"]
    assert result["layout"]["body"] or result["layout"]["headline"]
