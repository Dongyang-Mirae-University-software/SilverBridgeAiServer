"""위험 이어짐 (2026-11-15, COMPANION_RISK_CARRY · app/services/companion/risk_carry.py).

평가 세트 검수(6-2b): 위험한 말 바로 다음에 넘기거나 얼버무리는 말("괜찮겠지 뭐, 좀 누워 있을게")도 위험.
위험 판정은 그 턴의 말만 보므로 이 턴들을 놓쳤다. LLM·위험 모델은 스텁이다 (API 키·GPU 없음).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.config import get_settings
from app.routers import companion_router
from app.services.companion import pipeline, risk_carry
from app.services.companion.dialogue_store import store
from app.services.companion.reply_service import SAFETY_REPLY
from app.services.companion.text_emotion_service import TextEmotionResult

SET_PATH = Path(__file__).resolve().parents[1] / "tests" / "companion_eval" / "companion_quality_v1.json"
OPENING = "아침에 일어나셨을 때 몸은 좀 어떠셨어요?"
BODY = "가슴이 꽉 조이고 숨이 차"
MIND = "아무도. 나 같은 건 없어져도 아무도 모를 거야"
RISKY = {BODY, MIND, "아까부터 말이 잘 안 나오고 오른팔에 힘이 없어", "요즘은 그냥 자다가 안 깼으면 좋겠어"}
NORMAL_REPLY = "그러셨군요. 오늘 아침은 무엇을 드셨어요?"


# --- 넘기는 말 판정 -----------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["조금 쉬니까 괜찮아진 것 같기도 하고", "괜찮겠지 뭐, 좀 누워 있을게", "그냥 해 본 소리야",
                                  "별거 아니야", "신경 쓰지 마", "걱정 마, 참을 만해", "됐어"])
def test_downplay_words(text):
    assert risk_carry.is_downplay(text) is True


@pytest.mark.parametrize("text", ["알았어, 딸한테 전화해 볼게", "알았어, 119 부를게", "아들한테 연락했어", "괜찮아, 병원 다녀왔어",
                                  "괜찮은 것 같은데 구급차 불렀어", "다 귀찮아", "밥은 먹었어", "", "   "])
def test_accepting_help_or_unrelated_is_not_downplay(text):
    # 도움을 받겠다는 말은 넘기는 말이 섞여 있어도 이어 가지 않는다. '귀찮'은 '괜찮'이 아니다
    assert risk_carry.is_downplay(text) is False


def test_body_or_mind_wording():
    assert risk_carry.is_body_risk(BODY) and risk_carry.is_body_risk("아까부터 말이 잘 안 나오고 오른팔에 힘이 없어")
    assert not risk_carry.is_body_risk(MIND) and not risk_carry.is_body_risk("요즘은 그냥 자다가 안 깼으면 좋겠어")
    assert risk_carry.is_body_risk("아무 말", body_keyword=True)  # kw4 몸 응급 낱말이면 그대로 몸
    assert "증상" in risk_carry.carry_reply(True) and "마음에 걸려요" in risk_carry.carry_reply(False)
    assert "나아지셨다니" not in risk_carry.carry_reply(True)  # 나아졌다고 하지 않은 말을 넘겨짚지 않는다 (R5)


def test_rule_matches_the_reviewed_labels_in_the_quality_set():
    # 평가 세트의 위험 시나리오에서, 이 규칙으로 이어 간 결과가 재희 님 검수 라벨(risk)과 턴마다 같아야 한다
    data = json.loads(SET_PATH.read_text(encoding="utf-8"))
    checked = 0
    for s in data["scenarios"]:
        prev = False
        for t in s["turns"]:
            if t["elder"] in RISKY:
                assert t["risk"] is True, t["elder"]
                prev = True
                continue
            carried = prev and risk_carry.is_downplay(t["elder"])
            if prev:
                checked += 1
            assert carried == t["risk"], (s["id"], t["elder"])
            prev = carried
    assert checked >= 7  # K1~K4 위험 다음 턴들 + P3 등은 위험 앞 턴이 없어 이어 가지 않음


# --- 라우터에서 턴을 이어 갈 때 ------------------------------------------------------------------------

class SameLlm:
    def generate(self, prompt, **kw):
        return NORMAL_REPLY


def _calm(_t):
    return TextEmotionResult("CALM", 0.9, True, {})


@pytest.fixture
def setup(monkeypatch):
    s = get_settings()
    for k, v in dict(companion_risk_carry=True, companion_reply_history="summary", companion_llm_retry=0,
                     companion_flat_topic_switch=0, companion_avoid_reply=True, companion_risk_first=True,
                     companion_confirm_mode="logprob", companion_confirm_logprob=-0.3).items():
        monkeypatch.setattr(s, k, v)
    monkeypatch.setattr(pipeline, "extract_signals", lambda *a, **k: {"riskFlag": False})
    monkeypatch.setattr(pipeline, "summarize_answer", lambda llm, t: "요약")
    monkeypatch.setattr(pipeline, "_risk_prob_capped", lambda *a, **k: 0.01)
    monkeypatch.setattr(pipeline, "risk_probability", lambda *a, **k: 0.01)
    monkeypatch.setattr(pipeline, "keyword_risk", lambda t: t in RISKY)  # 위험 모델 대신: 정해 둔 말만 위험
    return lambda **kw: [monkeypatch.setattr(s, k, v) for k, v in kw.items()]


NO_FACE = {"label": None, "usable": False}


def _say(d, text):
    r = companion_router._run_turn(d, text, NO_FACE, _calm, SameLlm(), stt_logprob=-0.1)
    return companion_router._turn_response(store.get(d.dialogueId), r, face=NO_FACE)


def test_body_risk_then_downplay_keeps_risk_and_safety(setup):
    d = store.create("ward-carry1", "s-carry1", "09:00", OPENING, 8, None)
    first = _say(d, BODY)
    assert first["riskFlag"] is True and first["riskCarry"] is False
    second = _say(d, "조금 쉬니까 괜찮아진 것 같기도 하고")
    assert second["riskFlag"] is True and second["riskCarry"] is True
    assert second["reply"]["text"] == risk_carry.CARRY_BODY_REPLY
    assert second["reply"]["isFollowUp"] is False
    third = _say(d, "아들한테 연락했어")  # 도움을 받았다는 말 - 이어 가지 않고 평소 대답
    assert third["riskFlag"] is False and third["riskCarry"] is False and third["reply"]["text"] == NORMAL_REPLY
    assert store.get(d.dialogueId).riskSeen is True


def test_internal_source_is_safety_so_topic_switch_and_avoid_skip_it(setup):
    # 출처(replySource)는 응답에 없고 라우터 안에서만 쓴다 - SAFETY 여야 화제 전환·말 피함·되풀이 교체가 건드리지 않는다
    d = store.create("ward-carry0", "s-carry0", "09:00", OPENING, 8, None)
    companion_router._turn_response(store.get(d.dialogueId), companion_router._run_turn(d, BODY, NO_FACE, _calm, SameLlm(), stt_logprob=-0.1), face=NO_FACE)
    r = companion_router._run_turn(d, "괜찮겠지 뭐", NO_FACE, _calm, SameLlm(), stt_logprob=-0.1)
    assert r["replySource"] == "SAFETY" and r["riskCarry"] is True


def test_mind_risk_then_brushing_off_uses_mind_wording(setup):
    d = store.create("ward-carry2", "s-carry2", "18:00", "오늘 누구와 이야기를 나누셨어요?", 8, None)
    _say(d, MIND)
    out = _say(d, "그냥 해 본 소리야")
    assert out["riskCarry"] is True and out["reply"]["text"] == risk_carry.CARRY_REPLY
    assert out["reply"]["text"] != SAFETY_REPLY  # 같은 안내를 되풀이하지 않는다


def test_downplay_chain_and_no_response_do_not_break_carry(setup):
    d = store.create("ward-carry3", "s-carry3", "15:00", "오늘 오후에는 무엇을 하며 지내셨어요?", 8, None)
    _say(d, "아까부터 말이 잘 안 나오고 오른팔에 힘이 없어")
    assert _say(d, "괜찮겠지 뭐, 좀 누워 있을게")["riskCarry"] is True
    _say(d, "")  # 무응답 턴은 직전 위험 상태를 바꾸지 않는다
    assert _say(d, "별거 아니야")["riskCarry"] is True
    assert _say(d, "알았어, 119 부를게")["riskFlag"] is False


def test_no_carry_without_previous_risk_or_when_switched_off(setup):
    d = store.create("ward-carry4", "s-carry4", "09:00", OPENING, 8, None)
    out = _say(d, "괜찮아, 잘 잤어")  # 앞에 위험이 없으면 '괜찮'은 평범한 말
    assert out["riskFlag"] is False and out["riskCarry"] is False
    setup(companion_risk_carry=False)
    d2 = store.create("ward-carry5", "s-carry5", "09:00", OPENING, 8, None)
    _say(d2, BODY)
    off = _say(d2, "조금 쉬니까 괜찮아진 것 같기도 하고")
    assert off["riskFlag"] is False and off["riskCarry"] is False  # 끄면 전과 같다


def test_setting_default_is_on(monkeypatch):
    # 설정 모듈을 다시 불러오지 않는다 (reload 하면 뒤에 도는 다른 시험의 설정이 바뀐다 - 11/15 전체 시험에서 2개 실패로 확인)
    from app.core import config
    src = Path(config.__file__).read_text(encoding="utf-8")
    assert '_as_bool("COMPANION_RISK_CARRY", True)' in src
    monkeypatch.delenv("COMPANION_RISK_CARRY", raising=False)
    assert config._as_bool("COMPANION_RISK_CARRY", True) is True
    monkeypatch.setenv("COMPANION_RISK_CARRY", "false")
    assert config._as_bool("COMPANION_RISK_CARRY", True) is False


# --- 11/20 기능 점검 반영 (H-1 도움 거부 · M-1 부정된 괜찮 · L-1 되묻기 턴 · L-2 '죽겠' 관용구) ----------------

REFUSALS = ["병원 가기 싫어", "119 부르지 마", "전화하지 마 아무한테도", "연락하지 마", "병원 가 봤자 소용없어",
            "딸한테 알리지 마", "병원은 안 갈 거야", "119는 필요 없어"]


@pytest.mark.parametrize("text", REFUSALS)
def test_refusing_help_is_downplay_even_with_accept_words(text):
    assert risk_carry.is_refusal(text) and risk_carry.is_downplay(text)


@pytest.mark.parametrize("text", ["119 부를게", "알았어, 119 부를게", "딸한테 전화해 볼게", "아들한테 연락했어", "병원 다녀왔어",
                                  "구급차 불렀어"])
def test_accepting_help_still_not_carried(text):
    assert not risk_carry.is_refusal(text) and not risk_carry.is_downplay(text)


def test_plain_wording_for_refusal_and_negated_ok():
    for text in ("병원 가기 싫어", "하나도 안 괜찮아", "괜찮지 않아"):
        assert risk_carry.is_downplay(text)
        assert risk_carry.carry_reply(False, text) == risk_carry.CARRY_PLAIN_REPLY
        assert risk_carry.carry_reply(True, text) == risk_carry.CARRY_BODY_PLAIN_REPLY
        assert "괜찮다고 하셔도" not in risk_carry.carry_reply(False, text)
    # 검수한 넘기는 말은 전과 같은 문장 ("괜찮다고 하셔도" 로 받는다)
    assert risk_carry.carry_reply(True, "조금 쉬니까 괜찮아진 것 같기도 하고") == risk_carry.CARRY_BODY_REPLY
    assert risk_carry.carry_reply(False, "그냥 해 본 소리야") == risk_carry.CARRY_REPLY
    for reply in (risk_carry.CARRY_PLAIN_REPLY, risk_carry.CARRY_BODY_PLAIN_REPLY):
        assert "119" in reply  # 안전 안내 표시 (품질 기준 R10)


def test_dying_idiom_with_breath_is_body_but_death_wish_is_mind():
    assert risk_carry.is_body_risk("숨이 막혀 죽겠어")
    assert risk_carry.is_body_risk("가슴이 답답해 죽겠어")
    assert not risk_carry.is_body_risk("숨이 차서 그냥 죽고 싶어")


def test_router_refusal_after_risk_keeps_risk_and_plain_wording(setup):
    d = store.create("ward-carry6", "s-carry6", "09:00", OPENING, 8, None)
    _say(d, BODY)
    out = _say(d, "병원 가기 싫어")
    assert out["riskFlag"] is True and out["riskCarry"] is True
    assert out["reply"]["text"] == risk_carry.CARRY_BODY_PLAIN_REPLY
    assert _say(d, "119 부르지 마")["riskCarry"] is True  # 거부가 이어져도 계속 이어 간다


def test_router_confirm_turn_does_not_clear_previous_risk(setup):
    d = store.create("ward-carry7", "s-carry7", "09:00", OPENING, 8, None)
    _say(d, MIND)
    r = companion_router._run_turn(d, "어 그게 저기 뭐더라", NO_FACE, _calm, SameLlm(), stt_logprob=-2.0)  # 확신도 낮음 → 되묻기
    assert r["replySource"] == "CONFIRM" and not r.get("riskFlag")
    companion_router._turn_response(store.get(d.dialogueId), r, face=NO_FACE)
    assert store.get(d.dialogueId).lastRisk is True
    assert _say(d, "괜찮아, 그냥 해 본 소리야")["riskCarry"] is True
