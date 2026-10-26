"""대화형 말벗 엔진(companion/dialogue.py) 테스트. 모델·Ollama·서버 없이 돌아간다.

실행 (companion-emotion 폴더에서):
    venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app" / "companion"))
import dialogue as dlg  # noqa: E402

GOOD = "그러셨군요, 손주분이 와서 반가우셨겠어요. 그때 무엇을 함께 하셨나요?"


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make(clock, labels, generate=None, persisted=None):
    seq = iter(labels)

    def analyze(text, face=None):
        label = next(seq)
        return {"textEmotion": None, "label": label, "reason": f"테스트 {label}"}

    return dlg.Dialogue(
        dialogue_id="d1", ward_id="W", session_id="s", slot="09:00", question="오늘 아침 어떠셨어요?",
        analyze=analyze,
        persist=(persisted.append if persisted is not None else (lambda row: None)),
        generate=generate or (lambda q, a, i: {"reply": GOOD, "source": "model"}),
        clock=clock,
    )


# 규칙 검사
@pytest.mark.parametrize("text", [GOOD, "오늘 날씨가 좋네요. 산책은 하셨나요?"])
def test_good_replies_pass(text):
    assert dlg.check_reply(text) == []


@pytest.mark.parametrize("text, problem", [
    ("그렇군요. 무엇을 하셨나요", "물음표"),
    ("그렇군요. 무엇을 하셨나요? 또 뭐 하셨나요?", "물음표"),
    ("무엇을 하셨나요?", "공감 문장"),
    ("그렇군요. 무엇을 하셨어?", "질문 어미"),
    ("그렇군요. 병원에 가 보셨나요?", "의료"),
    ("그렇군요. 지난번에 무엇을 하셨나요?", "과거"),
    ("That is nice. 무엇을 하셨나요?", "영어"),
    ("좋아요 😊 무엇을 하셨나요?", "이모지"),
])
def test_bad_replies_are_caught(text, problem):
    found = dlg.check_reply(text)
    assert found, text
    assert any(problem in p for p in found), found


def test_length_limit():
    text = "그렇군요. " + "정말 " * 40 + "좋으셨나요?"
    assert any("초과" in p for p in dlg.check_reply(text))


# 대화 흐름
def test_normal_conversation_and_session_label():
    clock, saved = Clock(), []
    d = make(clock, ["기쁨", "평온"], persisted=saved)
    r1 = d.reply("잘 잤어요")
    assert r1["done"] is False and r1["reply"] == GOOD and r1["replySource"] == "model"
    r2 = d.reply("손주가 와서 좋았어요")
    assert r2["done"] is False
    out = d.finish()
    assert out["turnCount"] == 2
    assert out["sessionLabel"] == "판단 보류"  # 기쁨 1, 평온 1 → 동점, 확인 필요는 없음
    assert [row["turn"] for row in saved] == [1, 2]


def test_turn_question_follows_previous_reply():
    clock, seen = Clock(), []

    def gen(q, a, i):
        seen.append(q)
        return {"reply": f"공감 문장입니다. 질문 {i}요?", "source": "model"}

    d = make(clock, ["기쁨", "기쁨", "기쁨"], generate=gen)
    d.reply("첫 대답")
    d.reply("둘째 대답")
    assert seen == ["오늘 아침 어떠셨어요?", "공감 문장입니다. 질문 1요?"]


def test_two_minute_limit_ends_dialogue_and_keeps_last_answer():
    clock, saved = Clock(), []
    d = make(clock, ["기쁨", "확인 필요"], persisted=saved)
    d.reply("첫 대답")
    clock.now = 121.0
    r = d.reply("마지막 대답")
    assert r["done"] is True
    assert r["endReason"] == "시간 초과"
    assert r["reply"] is None
    assert len(saved) == 2  # 시간이 지나도 그 대답은 분석해서 저장한다
    with pytest.raises(ValueError):
        d.reply("끝난 뒤의 대답")


def test_seconds_left_counts_down():
    clock = Clock()
    d = make(clock, ["기쁨"])
    clock.now = 30.0
    assert d.seconds_left() == 90
    clock.now = 500.0
    assert d.seconds_left() == 0


def test_max_turns_ends_dialogue():
    clock = Clock()
    d = make(clock, ["기쁨"] * dlg.MAX_TURNS)
    last = None
    for i in range(dlg.MAX_TURNS):
        last = d.reply(f"대답 {i}")
    assert last["done"] is True
    assert last["endReason"] == "최대 턴 수"


def test_finish_before_end_is_allowed():
    clock = Clock()
    d = make(clock, ["확인 필요"])
    d.reply("힘들어요")
    out = d.finish()
    assert out["endReason"] == "사용자 종료"
    assert out["sessionLabel"] == "확인 필요"


# 세션 결과 묶기
@pytest.mark.parametrize("labels, expected", [
    (["기쁨", "기쁨", "평온"], "기쁨"),
    (["기쁨", "확인 필요", "기쁨"], "기쁨"),
    (["기쁨", "기쁨", "확인 필요", "슬픔"], "기쁨"),
    (["확인 필요", "확인 필요", "기쁨", "기쁨"], "확인 필요"),  # 동점 -> 확인 필요
    (["기쁨"] * 19 + ["평온"], "기쁨"),  # 1위
    (["기쁨", "판단 보류"], "판단 보류"),  # 동점 1위, 확인 필요는 없음 -> 판단 보류
    (["평온", "평온", "기쁨"], "평온"),
    (["분석 실패"], "판단 불가"),
    (["분석 실패", "기쁨"], "기쁨"),
    ([], "판단 불가"),
])
def test_combine_label(labels, expected):
    assert dlg.combine_label(labels) == expected


def test_distribution_percentages():
    labels = ["기쁨", "기쁨", "확인 필요", "평온", "분석 실패"]
    assert dlg.distribution(labels) == {"기쁨": 50, "확인 필요": 25, "평온": 25}


# 답변 생성: 모델이 규칙을 못 지키면 미리 정한 문장을 쓴다
def test_generate_falls_back_when_model_fails(monkeypatch):
    def broken(prompt, temp):
        raise OSError("ollama down")

    monkeypatch.setattr(dlg, "_ollama", broken)
    out = dlg.generate_reply("질문", "대답", 0)
    assert out == {"reply": dlg.FALLBACK_REPLIES[0], "source": "fallback"}


def test_generate_rejects_bad_model_output(monkeypatch):
    calls = []

    def bad(prompt, temp):
        calls.append(temp)
        return "병원에 가 보세요"  # 규칙 위반: 물음표·공감·의료 표현

    monkeypatch.setattr(dlg, "_ollama", bad)
    out = dlg.generate_reply("질문", "대답", 1)
    assert out["source"] == "fallback"
    assert out["reply"] == dlg.FALLBACK_REPLIES[1]
    assert len(calls) == 3


def test_generate_accepts_good_model_output(monkeypatch):
    monkeypatch.setattr(dlg, "_ollama", lambda prompt, temp: GOOD)
    out = dlg.generate_reply("질문", "대답", 0)
    assert out == {"reply": GOOD, "source": "model"}


def test_result_is_first_place_even_when_close():
    # 차이 기준은 쓰지 않는다: 1위가 결과다 (기쁨 11 vs 평온 10 -> 기쁨)
    assert dlg.combine_label(["기쁨"] * 11 + ["평온"] * 10) == "기쁨"


def test_summary_lists_all_results_in_order():
    labels = ["기쁨", "기쁨", "확인 필요", "평온", "분석 실패"]
    assert dlg.summarize(labels) == "기쁨 2회 (40%) · 확인 필요 1회 (20%) · 평온 1회 (20%) · 분석 실패 1회 (20%)"
    assert dlg.summarize([]) == ""
    assert dlg.summarize(["판단 불가"]) == "판단 불가 1회 (100%)"
