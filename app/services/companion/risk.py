"""위험 신호 판정 (노인 자살·자해 신호). 답을 만들지 않고 '있' 글자가 나올 확률을 읽어 판단한다.

확률이 RISK_THRESHOLD 이상이면 위험(riskFlag = true). 말 한마디로 판단하므로 아주 돌려 말한 신호는 놓칠 수 있다.
"""

from __future__ import annotations

from pathlib import Path

from app.core.config import get_settings

RISK_PROMPT_VERSION = "risk-v2"  # 프롬프트 파일을 고치면 올린다
RISK_THRESHOLD = 0.99  # 11/06 팀 측정 기준 (헛경보 0%대)
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
