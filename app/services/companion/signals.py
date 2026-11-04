from __future__ import annotations

import json
from pathlib import Path

from app.core.config import get_settings
from app.services.companion.llm_client import LlmClient, LlmUnavailable

# S5-2 결정 (사용자 객관식, 2026-11-02): 신호 값 목록. 문서 예시와 맞춘다.
#   emotion: 문서 확정값 (positive·neutral·negative·unknown)
#   activityMotivation: 세 단계 + 판단 불가 (사용자 선택)
#   socialContact·physicalCondition·futureOrientation: 언급 여부 (사용자 선택)
SIGNAL_VALUES: dict[str, tuple[str, ...]] = {
    "emotion": ("positive", "neutral", "negative", "unknown"),
    "activityMotivation": ("low", "mid", "high", "unknown"),
    "socialContact": ("mentioned", "unknown"),
    "physicalCondition": ("mentioned", "unknown"),
    "futureOrientation": ("mentioned", "unknown"),
}
SIGNAL_BOOLS: tuple[str, ...] = ("responseEvasion", "riskFlag")
SIGNAL_KEYS: tuple[str, ...] = tuple(SIGNAL_VALUES) + SIGNAL_BOOLS

PROMPT_VERSION = "signals-v1"  # 프롬프트 파일을 고치면 올린다 (로그의 promptVersion 값)
PROMPT_PATH = Path(__file__).parent / "prompts" / f"{PROMPT_VERSION}.txt"


def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def build_prompt(transcript: str) -> str:
    return load_prompt().replace("{{TRANSCRIPT}}", transcript)


def _extract_json_object(text: str) -> str | None:
    """LLM 답에서 첫 번째 가장 바깥 JSON 객체 조각을 꺼낸다. 여러 줄·설명이 섞여도 된다."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    return text[start:end + 1]


def parse_signals(raw: str) -> dict | None:
    """LLM 답을 신호 dict 로 바꾼다. 모양이 하나라도 맞지 않으면 None (S5-5).

    값 목록 밖의 글자, 빠진 키, 타입이 다른 값, 추가 키 모두 실패로 본다.
    """
    chunk = _extract_json_object(raw)
    if chunk is None:
        return None
    try:
        data = json.loads(chunk)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or set(data) != set(SIGNAL_KEYS):
        return None
    for key, allowed in SIGNAL_VALUES.items():
        if data[key] not in allowed:
            return None
    for key in SIGNAL_BOOLS:
        if not isinstance(data[key], bool):
            return None
    return {key: data[key] for key in SIGNAL_KEYS}


def extract_signals(transcript: str, llm: LlmClient, retry: int = 1) -> dict | None:
    """글 한 덩어리에서 7개 신호를 뽑는다. 실패하면 retry 번까지 다시 묻고, 끝내 안 되면 None (S5-5).

    LLM 연결 실패도 같은 방식으로 처리한다. None 은 명세대로 signals:null 로 나간다.
    대화 응답은 따로 만든다 (대화 우선 원칙).
    """
    s = get_settings()
    prompt = build_prompt(transcript)
    for _ in range(retry + 1):
        try:
            raw = llm.generate(prompt, temperature=s.companion_llm_temperature,
                               num_predict=s.companion_llm_num_predict, timeout=s.companion_llm_timeout_sec)
        except LlmUnavailable:
            continue
        signals = parse_signals(raw)
        if signals is not None:
            return signals
    return None
