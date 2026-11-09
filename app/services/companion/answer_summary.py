from __future__ import annotations

import re
from pathlib import Path

from app.core.config import get_settings
from app.services.companion.llm_client import LlmClient

SUMMARY_PROMPT_VERSION = "summary-v1"  # 요약 프롬프트 파일을 고치면 올린다
_PROMPT = Path(__file__).parent / "prompts" / "summary-v1.txt"
MAX_CHARS = 40  # 요약 한 줄의 상한 (프롬프트는 30자 이내를 요청한다)
# 11/10: 모델이 가끔(10개 중 1개) 앞에 붙이는 "요약:"·"기록:" 머리말 (COMPANION_SUMMARY_STRIP_PREFIX)
_HEADER = re.compile(r"^(요약|기록)\s*[:：]\s*")


def summarize_answer(llm: LlmClient, transcript: str) -> str | None:
    """어르신 대답을 한 줄로 줄인다. 원문은 돌려주지 않는다. 실패하면 None (대화는 계속된다)."""
    prompt = _PROMPT.read_text(encoding="utf-8").replace("{{TRANSCRIPT}}", transcript)
    try:
        raw = llm.generate(prompt, temperature=0.0, num_predict=60,
                           timeout=get_settings().companion_llm_timeout_sec)
    except Exception:  # noqa: BLE001 - 요약 실패가 대화를 막으면 안 된다
        return None
    text = re.sub(r"\s+", " ", raw).strip().strip("\"'")
    if get_settings().companion_summary_strip_prefix:
        text = _HEADER.sub("", text).strip().strip("\"'")
    if not text:
        return None
    return text[:MAX_CHARS]
