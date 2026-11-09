"""위험 신호 판정 (노인 자살·자해 신호). 답을 만들지 않고 '있' 글자가 나올 확률을 읽어 판단한다.

확률이 RISK_THRESHOLD 이상이면 위험(riskFlag = true). 말 한마디로 판단하므로 아주 돌려 말한 신호는 놓칠 수 있다.
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import get_settings

RISK_PROMPT_VERSION = "risk-v4"  # 프롬프트 파일을 고치면 올린다
RISK_THRESHOLD = 0.5  # 2026-11-09 측정. 검증 세트 B(개발에 안 쓴 새 문장 위험 20·평범 20): v4 18/20 포착, 헛경보 0/20
_PROMPT = Path(__file__).parent / "prompts" / f"{RISK_PROMPT_VERSION}.txt"
PREFIX = '{"risk": "'  # 모델이 이 뒤에 '있' 또는 '없'을 쓰도록 미리 적어 둔다
CANDIDATES = ("있", "없")


def risk_probability(llm, transcript: str) -> float | None:
    """'있'일 확률. 계산할 수 없으면 None (대화는 계속된다)."""
    probs_fn = getattr(llm, "next_token_probs", None)
    if probs_fn is None or not transcript.strip():
        return None
    prompt = _PROMPT.read_text(encoding="utf-8").replace("{{TRANSCRIPT}}", transcript)
    try:
        probs = probs_fn(prompt, PREFIX, CANDIDATES, timeout=get_settings().companion_llm_timeout_sec)
    except Exception:  # noqa: BLE001 - 위험 판정이 실패해도 대화를 막지 않는다
        return None
    return probs.get("있")


def is_risk(probability: float | None) -> bool:
    return probability is not None and probability >= get_settings().companion_risk_threshold


# 백엔드 결정 E (2026-11-03): 응급·자해 낱말은 모델과 따로, 규칙으로 먼저 잡는다. 모델이 꺼져 있어도 알림이 나간다.
RISK_KEYWORDS = (
    "죽고 싶", "죽고싶", "죽겠", "살기 싫", "살기싫", "살고 싶지", "살고싶지", "사라지고 싶", "사라지고싶",
    "끝내고 싶", "끝내고싶", "목숨", "자살", "자해", "약을 모", "약 모", "따라가고 싶", "따라가고싶",
    "숨이 안 쉬", "숨이 안쉬", "숨을 못", "가슴이 너무 아", "쓰러졌", "119", "꼼짝을 못", "꼼짝 못",
)


def keyword_risk(transcript: str) -> bool:
    """응급·자해 낱말이 하나라도 있으면 True. 띄어쓰기 차이는 무시한다."""
    text = transcript.replace(" ", "")
    return any(k.replace(" ", "") in text for k in RISK_KEYWORDS)
