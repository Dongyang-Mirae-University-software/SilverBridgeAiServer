"""글 대답은 되묻지 않는다 + 멀쩡한 대답을 버리던 check_reply 규칙 보정 (2026-11-13).

DGX(11/13, AI main 460200f 사본) 재현:
  ⑤ 글 대답은 받아쓰기가 없어 logprob 가 늘 None → 낱말 규칙으로 되물었다 ("잘 알아듣지 못했어요…", "혹시 편안 이야기이신가요?").
  ⑥ '대답과 무관'·'과거 가정'이 어르신 말을 다른 낱말로 받은 대답을 버려 대체 문장이 나갔다.
"""
from __future__ import annotations

import logging
import time

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.routers import companion_router
from app.routers.companion_router import get_llm, get_text_classifier, get_transcriber
from app.services.companion import pipeline
from app.services.companion.llm_client import LlmUnavailable
from app.services.companion.reply_service import ReplyResult, check_reply
from app.services.companion.text_emotion_service import TextEmotionResult
from scripts.companion.dev_app import app

KEY = "test-companion-key"
HDR = {"X-API-Key": KEY}
SAD = lambda _t: TextEmotionResult("슬픔", 0.9, True, {})  # noqa: E731 - '대답과 무관' 검사가 도는 기분


@pytest.fixture
def settings(monkeypatch):
    s = get_settings()

    def set_(**kw):
        for k, v in kw.items():
            monkeypatch.setattr(s, k, v)
    return set_


# ---------- ⑥ check_reply: 표의 8건 (통과 6 · 탈락 2) ----------

# (어르신 말, 모델 대답) - 버리면 안 되는 대답
KEEP = [
    ("아니 못잤어.", "못 주무셨다니 안타깝네요. 오늘 아침은 어떻게 시작하셨나요?"),
    ("아니. 요새 통 못했어.", "운동을 못하셨군요. 혹시 건강상의 이유가 있으신가요?"),
    ("몸이 예전같지 않아요.", "어르신 건강이 제일 중요하니까요. 혹시 몸 상태가 좀 나아지셨나요?"),
    ("지금은 무릎이 아파서 못하고 있어요.", "몸이 많이 불편하시겠어요. 특히 어디가 가장 불편하신가요?"),
    ("예전에는 조금만 걸어도 다리가 아프더라고.",
     "몸이 예전같지 않으시다니 안타깝네요. 특별히 불편하신 곳은 없으신가요?"),
    # 앞 턴에서 "몸이 예전같지 않아 피곤"하다고 한 뒤의 턴 - 이번 말에는 '예전'이 없다
    ("감기몸살인가봐 온몸이 쑤시고 아프네",
     "몸이 예전같지 않으셔서 피곤하시겠네요. 요즘 특별히 무리하신 일이 있으셨나요?"),
]
# 그대로 걸러야 하는 대답 (둘째 대답은 표에서 "…"로 줄인 부분을 채운 문장)
DROP = [
    ("어 여기저기 쑤시고 아파서 못했지",
     "잠이 많이 중요하시겠네요. 오늘 하루 중 가장 편안했던 순간은 언제였나요?", "대답과 무관"),
    ("무릎이 아파서 못하고 있어요",
     "예전에 좋아하셨던 활동들을 떠올리시면서 어떻게 하면 무릎에 무리 없이 즐기실 수 있을지 생각해 보셨나요?",
     "과거 가정"),
]


@pytest.mark.parametrize("transcript,reply", KEEP)
def test_reply_that_takes_up_the_answer_is_kept(transcript, reply):
    assert check_reply(reply, False, transcript, None) == []
    assert check_reply(reply, False, transcript, {"usable": True, "turnLabel": "슬픔"}) == []


@pytest.mark.parametrize("transcript,reply,rule", DROP)
def test_reply_that_misses_the_answer_is_still_dropped(transcript, reply, rule):
    assert rule in check_reply(reply, False, transcript, None)


def test_past_rule_still_catches_presumed_past():
    """어르신이 지난 일을 말하지 않았는데 지난 일을 넘겨짚으면 그대로 탈락이다."""
    assert "과거 가정" in check_reply("어제 산책은 다녀오셨어요?", False, "산책을 좋아해요", None)
    assert "과거 가정" in check_reply("예전에는 어떤 일을 하셨어요?", False, "요즘 심심해요", None)
    # 어르신이 같은 낱말·지난 때를 먼저 말했으면 넘겨짚기가 아니다
    assert "과거 가정" not in check_reply("어제 어디 다녀오셨어요?", False, "어제 시장에 갔다 왔어", None)
    assert "과거 가정" not in check_reply("옛날에는 무슨 일을 하셨어요?", False, "젊었을 때는 바빴지", None)
    # 지난 때를 말했어도 '어제'처럼 다른 낱말로 넘겨짚는 것은 그대로 본다
    assert "과거 가정" in check_reply("어제는 어떻게 보내셨어요?", False, "예전에는 바빴지", None)


def test_unrelated_rule_still_catches_off_topic_and_wrong_groups():
    """같은 뜻 묶음은 어르신 말과 대답에 함께 걸릴 때만 통과다 (대답에만 있는 묶음 낱말은 소용없다)."""
    assert "대답과 무관" in check_reply("요즘 식사는 잘 챙겨 드세요?", False, "손주가 다녀갔어", None)
    assert "대답과 무관" in check_reply("잠은 잘 주무세요?", False, "손주가 다녀갔어", None)
    # '잘못'은 '못 함' 묶음이 아니다
    assert "대답과 무관" in check_reply("어떤 게 잘못됐을까요?", False, "아니 못 갔어", None)


# ---------- ⑤ 글 대답은 되묻지 않는다 ----------

class ProbLlm:
    """위험 확률만 준다. 대답·신호 생성은 연결 실패 (실제 모델 없음)."""

    def __init__(self, p_yes: float = 0.0):
        self.p_yes = p_yes

    def generate(self, *_a, **_k):
        raise LlmUnavailable("off")

    def next_token_probs(self, prompt, prefix, candidates, timeout):
        return {"있": self.p_yes, "없": 1 - self.p_yes}

    def message_token_probs(self, *a, **k):
        return {"있": self.p_yes, "없": 1 - self.p_yes}


TEXT_ANSWERS = [
    "우리 아들놈이 허구헌날 술이나 퍼마시고 다녀서 걱정이지라",
    "집안에만 있으니까 답답하고 그런데 막상 나가려면 귀찮고 그러네.",
]


def _turn(transcript, lp=None, kind="voice", llm=None):
    return pipeline.run_turn(transcript=transcript, face={}, classify_text=SAD, llm=llm or ProbLlm(),
                             question="요즘 어떻게 지내세요?", prev_signals=None, is_last_turn=False, turn_no=1,
                             stt_logprob=lp, input_kind=kind)


@pytest.mark.parametrize("text", TEXT_ANSWERS)
def test_text_answer_is_a_normal_turn(settings, text):
    settings(companion_confirm_mode="logprob")
    assert _turn(text)["replySource"] == "CONFIRM"  # 전: 음성처럼 낱말 규칙에 걸렸다 (sensevoice 음성은 지금도 이렇다)
    r = _turn(text, kind="text")
    assert r["replySource"] != "CONFIRM" and r.get("needsCheck") is not True
    assert "textMood" in r and "avoided" in r


def test_text_answer_skips_lexicon_mode_too(settings):
    settings(companion_confirm_mode="lexicon")
    assert _turn("삼기탕을 먹었어요", kind="text")["replySource"] != "CONFIRM"
    assert _turn("삼기탕을 먹었어요", -1.0)["replySource"] == "CONFIRM"  # 음성은 그대로


def test_voice_turn_low_logprob_still_confirms(settings):
    settings(companion_confirm_mode="logprob", companion_confirm_logprob=-0.3)
    assert _turn("점심에 국수를 먹었어요", -2.0)["replySource"] == "CONFIRM"
    # 같은 확률이 와도 글 대답이면 되묻지 않는다 (글에는 확률이 없지만 섞여 들어와도 안전하게)
    assert _turn("점심에 국수를 먹었어요", -2.0, kind="text")["replySource"] != "CONFIRM"


def test_text_answer_risk_and_avoid_still_handled(settings):
    settings(companion_confirm_mode="logprob", companion_avoid_reply=True)
    risky = _turn("이제 그만 죽고 싶어", kind="text")
    assert risky["riskFlag"] is True and risky["replySource"] == "SAFETY"
    avoided = _turn("말하기 싫어", kind="text")
    assert avoided["avoided"] is True and avoided["replySource"] == "AVOID"


@pytest.fixture
def client():
    s = get_settings().model_copy(update={"companion_api_key": KEY, "companion_backend_base_url": ""})
    app.dependency_overrides[get_settings] = lambda: s
    app.dependency_overrides[get_transcriber] = lambda: None
    app.dependency_overrides[get_text_classifier] = lambda: SAD
    app.dependency_overrides[get_llm] = lambda: ProbLlm()
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_text_endpoint_passes_text_kind(client, monkeypatch):
    seen = {}
    real = pipeline.run_turn

    def spy(**kw):
        seen["kind"] = kw.get("input_kind")
        return real(**kw)
    monkeypatch.setattr(companion_router.pipeline, "run_turn", spy)
    body = {"wardId": "u-text-kind", "sessionId": "s1", "slot": "09:00", "question": "요즘 어떻게 지내세요?", "questionId": "q"}
    did = client.post("/companion/dialogues", json=body, headers=HDR).json()["data"]["dialogueId"]
    r = client.post(f"/companion/dialogues/{did}/reply", json={"text": TEXT_ANSWERS[0]}, headers=HDR)
    assert r.status_code == 200 and seen["kind"] == "text"
    assert r.json()["data"]["reply"]["text"] != "잘 알아듣지 못했어요. 한 번 더 말씀해 주시겠어요?"


# ---------- ⑥-c 대체 이유 코드 (원문 없음) ----------

def test_reply_reason_codes():
    r = pipeline.reply_reason
    assert r(ReplyResult("x", True, "LLM", []), False) == "NONE"
    assert r(ReplyResult("x", True, "FALLBACK", ["무응답"]), True) == "TIMEOUT"
    assert r(ReplyResult("x", True, "FALLBACK", ["대답과 무관", "과거 가정"]), False) == "RULE_UNRELATED+RULE_PAST"
    assert r(ReplyResult("x", True, "FALLBACK", ["LLM 실패"]), False) == "LLM_FAIL"
    assert r(ReplyResult("x", True, "SAFETY", ["대답과 무관"]), True) == "SAFETY"
    assert r(ReplyResult("x", True, "FALLBACK", ["새 규칙"]), False) == "OTHER"


def test_reason_log_has_codes_only(settings, caplog):
    settings(companion_confirm_mode="logprob")
    with caplog.at_level(logging.INFO, logger="app.services.companion.pipeline"):
        _turn(TEXT_ANSWERS[0], kind="text")
    lines = [m for m in caplog.messages if "[COMPANION-REPLY-SOURCE]" in m]
    assert lines == ["[COMPANION-REPLY-SOURCE] source=FALLBACK reason=LLM_FAIL timeout=0"]
    assert all("아들" not in m for m in caplog.messages)


class SlowLlm(ProbLlm):
    def generate(self, *_a, **_k):
        time.sleep(0.5)
        raise LlmUnavailable("late")


def test_timeout_is_logged_as_timeout(settings, caplog):
    settings(companion_confirm_mode="logprob", companion_turn_cap_sec=0.1)
    with caplog.at_level(logging.INFO, logger="app.services.companion.pipeline"):
        r = _turn("점심에 국수를 먹었어요", kind="text", llm=SlowLlm())
    assert r["replySource"] == "FALLBACK"
    assert any("reason=TIMEOUT timeout=1" in m for m in caplog.messages)
