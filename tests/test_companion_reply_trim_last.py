"""11/23 따로 켜는 대답 스위치: ④ 세 문장 줄이기(COMPANION_REPLY_TRIM) · 마지막 인사 프롬프트 v1b(COMPANION_REPLY_LAST_PROMPT)
· 마지막 인사 검사(COMPANION_REPLY_LAST_CHECK). 모두 기본 끔 - #339 의 STRICT 묶음과 따로 시험·채택하려고 나눴다."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import get_settings
from app.services.companion.reply_service import build_reply_prompt, check_reply, trim_reply

THREE = "반찬 보내 주셔서 다행이네요. 큰 힘이 되셨겠어요. 반찬은 맛있게 드셨나요?"
FUTURE_DONE = "김치찌개로 하루를 마무리하시는 모습이 보기 좋네요. 편안한 저녁 되세요."
VAGUE = "그냥 보내셨군요. 특별히 하고 싶은 활동은 없으셨나요?"


@pytest.fixture
def settings(monkeypatch):
    s = get_settings()

    def set_(**kw):
        for k, v in kw.items():
            monkeypatch.setattr(s, k, v)
    set_(companion_reply_strict=False, companion_reply_strict_plan=False, companion_reply_trim=False,
         companion_reply_last_prompt="v1", companion_reply_last_check=False, companion_reply_history="off")
    return set_


def test_defaults_all_off():
    src = Path(__file__).resolve().parents[1].joinpath("app/core/config.py").read_text(encoding="utf-8")
    assert '_as_bool("COMPANION_REPLY_TRIM", False)' in src
    assert '_as_choice("COMPANION_REPLY_LAST_PROMPT", "v1", ("v1", "v1b"))' in src
    assert '_as_bool("COMPANION_REPLY_LAST_CHECK", False)' in src


def test_trim_switch_alone(settings):
    assert trim_reply(THREE, False) == THREE  # 모두 끔이면 그대로
    settings(companion_reply_trim=True)
    assert trim_reply(THREE, False) == "반찬 보내 주셔서 다행이네요. 반찬은 맛있게 드셨나요?"
    assert trim_reply(THREE, True) == THREE  # 마지막 인사는 줄이지 않는다
    # TRIM 만 켜면 ②막연한 질문·①마지막 인사 검사는 돌지 않는다
    assert "막연한 질문" not in check_reply(VAGUE, False, "그냥 그래")
    assert "앞일을 한 일로" not in check_reply(FUTURE_DONE, True, "이따가 김치찌개 데워 먹으려고")


def test_strict_still_turns_trim_on(settings):
    settings(companion_reply_strict=True)
    assert trim_reply(THREE, False) != THREE


def test_last_check_alone(settings):
    settings(companion_reply_last_check=True)
    assert check_reply(FUTURE_DONE, True, "이따가 김치찌개 데워 먹으려고") == ["앞일을 한 일로"]
    assert check_reply("김치찌개 끓여 드신다니 든든하시겠어요. 편안한 저녁 보내세요.", True, "이따가 김치찌개 데워 먹으려고") == []
    assert "막연한 질문" not in check_reply(VAGUE, False, "그냥 그래")  # 꼬리질문 검사는 켜지 않는다


def test_last_prompt_v1b_in_off_mode(settings):
    p1 = build_reply_prompt("저녁은 뭐 드실 거예요?", "이따가 김치찌개 데워 먹으려고", None, True)
    assert "앞으로의 일로 받습니다" not in p1
    settings(companion_reply_last_prompt="v1b")
    p2 = build_reply_prompt("저녁은 뭐 드실 거예요?", "이따가 김치찌개 데워 먹으려고", None, True)
    assert "앞으로의 일로 받습니다" in p2 and "어르신의 마지막 대답: 이따가 김치찌개 데워 먹으려고" in p2
    assert "{{" not in p2
    # 꼬리질문 프롬프트는 바뀌지 않는다
    assert "앞으로의 일로 받습니다" not in build_reply_prompt("점심 드셨어요?", "국수 먹었어", None, False)


def test_last_prompt_v1b_in_summary_without_history_last(settings):
    settings(companion_reply_history="summary", companion_reply_history_last=False, companion_reply_last_prompt="v1b")
    p = build_reply_prompt("저녁은 뭐 드실 거예요?", "이따가 김치찌개 데워 먹으려고", None, True, (("점심 드셨어요?", "국수"),))
    assert "앞으로의 일로 받습니다" in p and "지금까지의 대화" not in p
    settings(companion_reply_history_last=True)  # 흐름 넣은 v2-last 에는 v1b 를 쓰지 않는다
    assert "지금까지의 대화" in build_reply_prompt("저녁은?", "이따가 국수", None, True, (("점심?", "국수"),))
