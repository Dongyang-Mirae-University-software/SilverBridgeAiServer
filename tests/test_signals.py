from __future__ import annotations

import json

from app.services.companion.signals import (
    PROMPT_PATH,
    PROMPT_VERSION,
    SIGNAL_KEYS,
    build_prompt,
    extract_signals,
    parse_signals,
)

GOOD = {
    "emotion": "negative",
    "activityMotivation": "low",
    "socialContact": "unknown",
    "physicalCondition": "mentioned",
    "futureOrientation": "unknown",
    "responseEvasion": False,
    "riskFlag": False,
}


def _text(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


class FakeLlm:
    def __init__(self, answers: list[str]):
        self.answers = list(answers)
        self.calls = 0

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls += 1
        return self.answers.pop(0)


def test_valid_json_is_parsed_even_with_extra_text():
    raw = "분석 결과입니다.\n" + _text(GOOD) + "\n끝."
    assert parse_signals(raw) == GOOD


def test_all_seven_keys_are_present():
    assert set(parse_signals(_text(GOOD))) == set(SIGNAL_KEYS)
    assert len(SIGNAL_KEYS) == 7


def test_value_outside_the_list_is_rejected():
    assert parse_signals(_text({**GOOD, "activityMotivation": "very-high"})) is None


def test_missing_or_extra_key_is_rejected():
    missing = {k: v for k, v in GOOD.items() if k != "riskFlag"}
    assert parse_signals(_text(missing)) is None
    assert parse_signals(_text({**GOOD, "mood": "good"})) is None


def test_bool_fields_must_be_real_booleans():
    assert parse_signals(_text({**GOOD, "riskFlag": "false"})) is None
    assert parse_signals(_text({**GOOD, "responseEvasion": 0})) is None


def test_non_json_and_non_object_are_rejected():
    assert parse_signals("모르겠습니다") is None
    assert parse_signals("[1, 2, 3]") is None
    assert parse_signals("{broken json") is None
    assert parse_signals("") is None


def test_extract_retries_once_then_succeeds():
    llm = FakeLlm(["죄송합니다", _text(GOOD)])
    assert extract_signals("그냥 아무것도 안 하고 싶네", llm, retry=1) == GOOD
    assert llm.calls == 2


def test_extract_gives_none_after_all_retries():
    llm = FakeLlm(["no", "still no", "never"])
    assert extract_signals("안녕하세요", llm, retry=1) is None
    assert llm.calls == 2  # 처음 1번 + 재시도 1번


def test_prompt_file_is_versioned_and_fills_transcript():
    assert PROMPT_PATH.name == f"{PROMPT_VERSION}.txt"
    prompt = build_prompt("오늘 병원에 다녀왔어요")
    assert "오늘 병원에 다녀왔어요" in prompt
    assert "{{TRANSCRIPT}}" not in prompt
